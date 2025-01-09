import json
import random
import argparse
from g_eval import GEvalAPI
from evaluators.fed_evaluate import process_fed_data
from evaluators.tc_usr_evaluate import process_tc_usr_data
import time
import os
import pandas as pd

def plot_scatter(df, output_folder):
    import matplotlib.pyplot as plt
    plt.scatter(df["evaluation_mean"], df["overall_score"], alpha=0.7, color="blue")
    plt.title("Scatter Plot: Evaluation Mean vs Overall Score")
    plt.xlabel("Evaluation Mean")
    plt.ylabel("Overall Score")
    plt.grid(alpha=0.5)
    plt.savefig(os.path.join(output_folder, "scatter_plot.png"))
    plt.close()

def plot_boxplot(df, output_folder):
    import matplotlib.pyplot as plt
    plt.boxplot([df["evaluation_mean"], df["overall_score"]], labels=["Evaluation Mean", "Overall Score"], patch_artist=True)
    plt.title("Distribuzione dei Punteggi")
    plt.ylabel("Valori")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "boxplot.png"))
    plt.close()

def plot_difference_histogram(df, output_folder):
    import matplotlib.pyplot as plt
    df["difference"] = abs(df["evaluation_mean"] - df["overall_score"])
    plt.hist(df["difference"], bins=20, alpha=0.7, color="orange", edgecolor="black")
    plt.title("Distribuzione delle Differenze tra Evaluation Mean e Overall Score")
    plt.xlabel("Differenza Assoluta")
    plt.ylabel("Frequenza")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.savefig(os.path.join(output_folder, "difference_histogram.png"))
    plt.close()

def generate_summary_table(df, results, output_folder):
    import pandas as pd
    summary = {
        "Metric": ["Pearson", "Spearman", "Kendall-Tau"],
        "Value": [results["pearson"], results["spearman"], results["kendalltau"]]
    }
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(output_folder, "summary_table.csv"), index=False)

def plot_heatmap(df, output_folder):
    import seaborn as sns
    import matplotlib.pyplot as plt

    evaluation_df = pd.DataFrame(df["evaluation"].tolist())
    evaluation_df["overall_score"] = df["overall_score"]

    correlation_matrix = evaluation_df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", cbar=True)
    plt.title("Heatmap delle Correlazioni")
    plt.savefig(os.path.join(output_folder, "heatmap.png"))
    plt.close()

def plot_bar_correlation(results, output_folder):
    """
    Crea un grafico a barre per visualizzare le correlazioni (Pearson, Spearman, Kendall-Tau),
    aggiungendo i valori sopra ogni barra.
    """
    import matplotlib.pyplot as plt

    metrics = ["Pearson", "Spearman", "Kendall-Tau"]
    values = [results["pearson"], results["spearman"], results["kendalltau"]]

    # Creazione del grafico a barre
    plt.bar(metrics, values, color=["blue", "orange", "green"])
    plt.title("Correlazioni tra Evaluation Mean e Overall Score")
    plt.ylabel("Valore della Correlazione")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10, color="black")

    plt.savefig(os.path.join(output_folder, "correlation_bar_plot.png"))
    plt.close()

def generate_results(input_file, output_folder):
    """
    Calcola le correlazioni (Pearson, Spearman, Kendall) tra la media dei punteggi di evaluation e overall_score.
    Genera i risultati e li salva nella cartella specificata.
    """
    import os
    import json
    import pandas as pd
    from scipy.stats import spearmanr, pearsonr, kendalltau
    from prettytable import PrettyTable

    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data)

    df["evaluation_mean"] = df["evaluation"].apply(lambda x: sum(x.values()) / len(x))

    pred_scores, human_scores = {}, {}

    for _, row in df.iterrows():
        system_id = row["system"]

        if system_id not in pred_scores:
            pred_scores[system_id] = []
            human_scores[system_id] = []

        # Aggiungi i punteggi
        pred_scores[system_id].append(row["evaluation_mean"])
        human_scores[system_id].append(row["overall_score"])

    results = {'pearson': 0, 'spearman': 0, 'kendalltau': 0}
    valid_systems = 0

    for system_id in pred_scores:
        pred_scores_system = pred_scores[system_id]
        human_scores_system = human_scores[system_id]

        if len(set(human_scores_system)) <= 1 or len(set(pred_scores_system)) <= 1:
            continue

        results['pearson'] += pearsonr(pred_scores_system, human_scores_system)[0]
        results['spearman'] += spearmanr(pred_scores_system, human_scores_system)[0]
        results['kendalltau'] += kendalltau(pred_scores_system, human_scores_system)[0]
        valid_systems += 1

    if valid_systems > 0:
        results = {k: v / valid_systems for k, v in results.items()}

    table = PrettyTable(['Pearson', 'Spearman', 'Kendall'])
    table.add_row([round(results['pearson'], 4), round(results['spearman'], 4), round(results['kendalltau'], 4)])
    print("Correlazioni calcolate:")
    print(table)

    with open(os.path.join(output_folder, "correlations.txt"), "w") as f:
        f.write(str(table))

    plot_scatter(df, output_folder)
    plot_boxplot(df, output_folder)
    plot_difference_histogram(df, output_folder)
    generate_summary_table(df, results, output_folder)
    plot_bar_correlation(results, output_folder)
    print(f"Risultati generati e salvati in {output_folder}")

def load_config():
    import os
    from dotenv import load_dotenv

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key non trovata. Assicurati che OPENAI_API_KEY sia configurata.")
    return api_key


def sample_data(file_path, num_records=None):
    with open(file_path, "r") as f:
        data = json.load(f)
    if num_records:
        return random.sample(data, min(num_records, len(data)))
    return data


def main(mode, input_file, single_template_path, full_template_path, output_file, num_records):
    if mode in ["fed", "tc_usr"]:
        api_key = load_config()
        model = "gpt-4o-mini"
        g_eval = GEvalAPI(api_key=api_key, model=model)
    data = sample_data(input_file, num_records)

    temp_file = "results/temp_test_data.json"
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=4)

    if mode == "results":
        generate_results(input_file, output_file)
    elif mode == "fed":
        process_fed_data(temp_file, g_eval, single_template_path, full_template_path, output_file)
    elif mode == "tc_usr":
        process_tc_usr_data(temp_file, g_eval, single_template_path, output_file)

    else:
        raise ValueError("Modalità non valida. Usa 'fed', 'tc_usr' o 'results'.")

    print(f"Elaborazione completata. Risultati salvati in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elabora i dataset fed, tc_usr o genera risultati.")
    parser.add_argument("--mode", type=str, required=True, choices=["fed", "tc_usr", "results"],
                        help="Modalità: 'fed', 'tc_usr' o 'results'.")
    parser.add_argument("--input_file", type=str, required=True, help="Percorso al file JSON del dataset.")
    parser.add_argument("--single_template_path", type=str, help="Percorso al template per risposte singole.")
    parser.add_argument("--full_template_path", type=str, help="Percorso al template per dialoghi completi (solo per 'fed').")
    parser.add_argument("--output_file", type=str, required=True, help="Percorso per salvare i risultati o la cartella dei risultati ('results').")
    parser.add_argument("--num_records", type=int, default=None, help="Numero di record da elaborare (opzionale).")
    args = parser.parse_args()

    main(
        mode=args.mode,
        input_file=args.input_file,
        single_template_path=args.single_template_path,
        full_template_path=args.full_template_path,
        output_file=args.output_file,
        num_records=args.num_records,
    )