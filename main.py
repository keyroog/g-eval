import json
import random
import argparse
from g_eval import GEvalAPI
from evaluators.fed_evaluate import process_fed_data
from evaluators.tc_usr_evaluate import process_tc_usr_data


def load_config():
    """
    Carica l'API key da variabili d'ambiente o file .env.
    """
    import os
    from dotenv import load_dotenv

    load_dotenv()  # Carica variabili da .env, se presente
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("API key non trovata. Assicurati che OPENAI_API_KEY sia configurata.")
    return api_key


def sample_data(file_path, num_records=None):
    """
    Carica i dati e restituisce un sottoinsieme casuale o tutti i record.
    Args:
        file_path (str): Percorso al file JSON.
        num_records (int, opzionale): Numero di record da campionare.
    Returns:
        list: Dati campionati o completi.
    """
    with open(file_path, "r") as f:
        data = json.load(f)
    if num_records:
        return random.sample(data, min(num_records, len(data)))
    return data


def main(mode, input_file, single_template_path, full_template_path, output_file, num_records):
    """
    Elabora il dataset in base alla modalità selezionata.
    Args:
        mode (str): Modalità ('fed' o 'tc_usr').
        input_file (str): Percorso al file JSON del dataset.
        single_template_path (str): Percorso al template per risposte singole.
        full_template_path (str): Percorso al template per dialoghi completi.
        output_file (str): Percorso per salvare i risultati.
        num_records (int, opzionale): Numero di record da elaborare.
    """
    # Carica l'API key
    api_key = load_config()
    model = "gpt-4o-mini"
    g_eval = GEvalAPI(api_key=api_key, model=model)

    # Carica i dati
    data = sample_data(input_file, num_records)

    # Salva il sottoinsieme per il debug (se necessario)
    temp_file = "results/temp_test_data.json"
    with open(temp_file, "w") as f:
        json.dump(data, f, indent=4)

    # Elaborazione in base alla modalità
    if mode == "fed":
        process_fed_data(temp_file, g_eval, single_template_path, full_template_path, output_file)
    elif mode == "tc_usr":
        process_tc_usr_data(temp_file, g_eval, single_template_path, output_file)
    else:
        raise ValueError("Modalità non valida. Usa 'fed' o 'tc_usr'.")

    print(f"Elaborazione completata. Risultati salvati in {output_file}")


if __name__ == "__main__":
    # Parsing degli argomenti
    parser = argparse.ArgumentParser(description="Elabora i dataset fed o tc_usr con valutazione automatizzata.")
    parser.add_argument("--mode", type=str, required=True, choices=["fed", "tc_usr"], help="Modalità: 'fed' o 'tc_usr'.")
    parser.add_argument("--input_file", type=str, required=True, help="Percorso al file JSON del dataset.")
    parser.add_argument("--single_template_path", type=str, required=True, help="Percorso al template per risposte singole.")
    parser.add_argument("--full_template_path", type=str, help="Percorso al template per dialoghi completi (solo per 'fed').")
    parser.add_argument("--output_file", type=str, required=True, help="Percorso per salvare i risultati.")
    parser.add_argument("--num_records", type=int, default=None, help="Numero di record da elaborare (opzionale).")
    args = parser.parse_args()

    # Esegui il main
    main(
        mode=args.mode,
        input_file=args.input_file,
        single_template_path=args.single_template_path,
        full_template_path=args.full_template_path,
        output_file="results/" + args.output_file,
        num_records=args.num_records,
    )