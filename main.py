import json
import random
import argparse
import os
from dotenv import load_dotenv
from g_eval import GEvalAPI
from evaluators.fed_evaluate import process_fed_data
from evaluators.tc_usr_evaluate import process_tc_usr_data
from evaluators.pc_usr_evaluate import process_pc_usr_data
from evaluators.dstc_evaluate import process_dstc_data
from evaluators.convai_evaluate import process_convai_data
from scipy.stats import pearsonr, spearmanr, kendalltau
import pandas as pd

def plot_distance_bars(results, output_folder):
    import matplotlib.pyplot as plt

    metrics = ["Pearson", "Spearman", "Kendall-Tau"]
    values = [results[0], results[1], results[2]]

    plt.bar(metrics, values, color=["blue", "orange", "green"])
    plt.title("Correlazioni tra Evaluation Mean e Overall Score")
    plt.ylabel("Valore della Correlazione")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    for i, v in enumerate(values):
        plt.text(i, v + 0.02, f"{v:.4f}", ha="center", fontsize=10, color="black")

    plt.savefig(os.path.join(output_folder, "distance_correlation_bar_plot.png"))
    plt.close()

    print(f"Plot delle distanze salvato in {output_folder}/distance_correlation_bar_plot.png")

def plot_distance_distribution(df, output_folder):
    """
    Genera un istogramma che mostra la distribuzione delle differenze tra evaluation_mean e overall_score.
    """
    import matplotlib.pyplot as plt
    
    df["distance"] = abs(df["evaluation_mean"] - df["overall_score"])

    plt.bar(["Distanza Media"], [df["distance"].mean()], color="purple")
    plt.title("Distribuzione delle Differenze tra Evaluation Mean e Overall Score")
    plt.ylabel("Valore Medio della Differenza")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Aggiungi l'etichetta del valore sopra la barra
    plt.text(0, df["distance"].mean() + 0.02, f"{df['distance'].mean():.4f}", ha="center", fontsize=10, color="black")

    plt.savefig(os.path.join(output_folder, "distance_bar_plot.png"))
    plt.close()

def load_config():
    """Carica la API key dall'ambiente"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key non trovata. Assicurati che OPENAI_API_KEY sia configurata.")
    return api_key


def sample_data(file_path, num_records=None):
    """Carica e campiona i dati dal file JSON."""
    with open(file_path, "r") as f:
        data = json.load(f)
    return random.sample(data, min(num_records, len(data))) if num_records else data

def calculate_correlations(data):
    """
    Calcola le correlazioni tra overall_score e evaluation["Overall"].
    """
    human_scores = []
    model_scores = []

    for entry in data:
          human_score = entry.get("overall_score")
          model_score = entry.get("evaluation", {}).get("Overall")

          if human_score is not None and model_score is not None:
              human_scores.append(human_score)
              model_scores.append(model_score)

    # Calcolo delle correlazioni
    pearson_corr, _ = pearsonr(human_scores, model_scores)
    spearman_corr, _ = spearmanr(human_scores, model_scores)
    kendall_corr, _ = kendalltau(human_scores, model_scores)

    return pearson_corr, spearman_corr, kendall_corr


def main(mode, input_file, single_template_path, full_template_path, output_file, num_records):
    """
    Esegue l'elaborazione del dataset in base alla modalità selezionata.

    Args:
        mode (str): Modalità di esecuzione (fed, tc_usr, pc_usr, dstc, convai).
        input_file (str): Percorso al file JSON del dataset.
        single_template_path (str): Percorso al template per risposte singole.
        full_template_path (str): Percorso al template per dialoghi completi (solo per 'fed').
        output_file (str): Percorso per salvare i risultati.
        num_records (int): Numero di record da elaborare.
    """
    api_key = load_config()
    model = "gpt-4o-mini"
    g_eval = GEvalAPI(api_key=api_key, model=model)

    data = sample_data(input_file, num_records)
    temp_file = "results/temp_test_data.json"

    # Salva i dati campionati temporaneamente
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=4)

    if mode == "fed":
        results = process_fed_data(temp_file, g_eval, single_template_path, full_template_path, output_file)
    elif mode == "tc_usr":
        results = process_tc_usr_data(temp_file, g_eval, single_template_path, output_file)
    elif mode == "pc_usr":
        results = process_pc_usr_data(temp_file, g_eval, single_template_path, output_file)
    elif mode == "dstc":
        results = process_dstc_data(temp_file, g_eval, full_template_path, output_file)
    elif mode == "convai":
        results = process_convai_data(temp_file, g_eval, full_template_path, output_file)
    elif mode == "result":
        results = json.load(open(input_file, "r"))
        correlation_results = calculate_correlations(results)
        print("\nCorrelazioni calcolate:")
        print(f"Pearson: {correlation_results[0]}")
        print(f"Spearman: {correlation_results[1]}")
        print(f"Kendall-Tau: {correlation_results[2]}")
        plot_distance_bars(correlation_results, os.path.dirname(output_file))
    else:
        raise ValueError("Modalità non valida. Usa 'fed', 'tc_usr', 'pc_usr', 'dstc' o 'convai'.")

    print(f"Elaborazione completata per la modalità '{mode}'. Risultati salvati in {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Esegui la demo per i dataset di valutazione.")
    parser.add_argument("--mode", type=str, required=True, choices=["fed", "tc_usr", "pc_usr", "dstc", "convai", "result"],
                        help="Modalità: 'fed', 'tc_usr', 'pc_usr', 'dstc', 'convai', 'result'.")
    parser.add_argument("--input_file", type=str, required=True, help="Percorso al file JSON del dataset.")
    parser.add_argument("--single_template_path", type=str, help="Percorso al template per risposte singole.")
    parser.add_argument("--full_template_path", type=str, help="Percorso al template per dialoghi completi (solo per 'fed').")
    parser.add_argument("--output_file", type=str, required=True, help="Percorso per salvare i risultati.")
    parser.add_argument("--num_records", type=int, default=2, help="Numero di record da elaborare (opzionale).")
    args = parser.parse_args()

    main(
        mode=args.mode,
        input_file=args.input_file,
        single_template_path=args.single_template_path,
        full_template_path=args.full_template_path,
        output_file=args.output_file,
        num_records=args.num_records,
    )