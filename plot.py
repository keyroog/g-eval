import json
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import pearsonr, spearmanr, kendalltau

def load_dataset(file_path):
    """
    Carica i dati dal file JSON.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_correlations(data):
    """
    Calcola le correlazioni tra overall_score e evaluation["Overall"].
    """
    human_scores = []
    model_scores = []

    for entry in data:
        #evaluate only dialog level
        if entry.get("level") == "dialog-level":
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

def plot_correlations(correlations, output_path):
    """
    Genera un grafico a barre delle correlazioni.
    """
    labels = ['Pearson', 'Spearman', 'Kendall-Tau']
    colors = ['blue', 'orange', 'green']

    plt.figure(figsize=(8, 6))
    bars = plt.bar(labels, correlations, color=colors)
    
    for bar, value in zip(bars, correlations):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height(), f'{value:.4f}', ha='center', va='bottom')

    plt.ylim(0, 1)
    plt.xlabel("Tipi di correlazione")
    plt.ylabel("Valore della correlazione")
    plt.title("Correlazioni tra Evaluation Mean e Overall Score")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig(output_path)
    plt.show()
    print(f"Grafico salvato in: {output_path}")

if __name__ == "__main__":
    # Specifica il percorso del dataset e dell'output
    dataset_path = "results/fed/results_fed_overall.json"
    output_image = "fed_correlation_dialog_level.png"

    # Carica il dataset e calcola le correlazioni
    dataset = load_dataset(dataset_path)
    pearson, spearman, kendall = calculate_correlations(dataset)

    # Genera il grafico
    plot_correlations([pearson, spearman, kendall], output_image)