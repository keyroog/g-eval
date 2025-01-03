import json
import os
from dotenv import load_dotenv
from g_eval import GEvalAPI
from evaluators.fed_evaluate import process_fed_data
import time


def load_offset(offset_file):
    """
    Carica l'offset dal file salvato.
    Args:
        offset_file (str): Percorso al file di offset.
    Returns:
        int: Offset salvato o 0 se il file non esiste.
    """
    try:
        with open(offset_file, "r") as f:
            return json.load(f).get("offset", 0)
    except FileNotFoundError:
        return 0

def load_results(output_file):
    """
    Carica i risultati esistenti dal file di output.
    Args:
        output_file (str): Percorso al file di output.
    Returns:
        list: Risultati esistenti o lista vuota se il file non esiste.
    """
    try:
        with open(output_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def save_offset(offset, offset_file):
    """
    Salva l'offset corrente.
    Args:
        offset (int): Offset corrente.
        offset_file (str): Percorso al file di offset.
    """
    with open(offset_file, "w") as f:
        json.dump({"offset": offset}, f, indent=4)


def run_test(input_file, single_template_path, full_template_path, output_file, offset_file, max_requests_per_minute=13):
    """
    Esegue il processo sul dataset, rispettando i limiti di rate limit e salvando l'offset.
    Args:
        input_file (str): Percorso al file JSON del dataset completo.
        single_template_path (str): Percorso al template per risposte singole.
        full_template_path (str): Percorso al template per dialoghi completi.
        output_file (str): Percorso per salvare i risultati.
        offset_file (str): Percorso per salvare l'offset.
        max_requests_per_minute (int): Limite massimo di richieste al minuto.
    """
    load_dotenv()  # Carica le variabili dal file .env
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("L'API key non è stata trovata. Assicurati che OPENAI_API_KEY sia definita nel file .env.")

    MODEL = "gpt-4o-mini"
    g_eval = GEvalAPI(api_key=API_KEY, model=MODEL)

    # Carica il dataset
    with open(input_file, "r") as f:
        data = json.load(f)

    # Carica l'offset corrente
    offset = load_offset(offset_file)
    print(f"Ripresa dal record {offset} su {len(data)}.")

    # Carica i risultati esistenti
    results = load_results(output_file)

    # Carica i template
    single_template = g_eval.load_prompt_template(single_template_path)
    full_template = g_eval.load_prompt_template(full_template_path)

    for idx, record in enumerate(data[offset:], start=offset):
        try:
            print(f"Elaborazione record {idx + 1}/{len(data)}...")
            response = record.get("response", "").strip()
            annotations = record.get("annotations", {})
            context = record["context"]
            response = record.get("response", "").strip()

            # Calcolo dell'overall_score come media dei valori in annotations["Overall"]
            overall_scores = annotations.get("Overall", [])
            overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None

             # Scegli il template e genera il prompt
            if response:  # Caso con una risposta singola
                prompt = g_eval.generate_prompt(single_template, context, response)
            else:  # Caso con dialogo completo
                prompt = g_eval.generate_prompt(full_template, context, "")

            # Effettua la richiesta al modello
            evaluations = g_eval.send_request(prompt)

            # Estrai i punteggi di Coherence, Fluency e Relevance con debug
            coherence, fluency, relevance = None, None, None
            for evaluation in evaluations:
                print(f"DEBUG: Valutazione ricevuta: {evaluation}")  # Debug dell'intera valutazione
                lines = evaluation.split("\n")
                for line in lines:
                    print(f"DEBUG: Linea corrente: {line}")  # Debug di ogni linea
                    if "Coherence" in line and ":" in line:
                        try:
                            coherence = int(line.split(":")[1].strip())
                            print(f"DEBUG: Coherence estratta: {coherence}")  # Debug Coherence
                        except ValueError:
                            print(f"Errore nel parsing di Coherence: {line}")
                    if "Fluency" in line and ":" in line:
                        try:
                            fluency = int(line.split(":")[1].strip())
                            print(f"DEBUG: Fluency estratta: {fluency}")  # Debug Fluency
                        except ValueError:
                            print(f"Errore nel parsing di Fluency: {line}")
                    if "Relevance" in line and ":" in line:
                        try:
                            relevance = int(line.split(":")[1].strip())
                            print(f"DEBUG: Relevance estratta: {relevance}")  # Debug Relevance
                        except ValueError:
                            print(f"Errore nel parsing di Relevance: {line}")

            # Salva i risultati
            results.append({
                "context": context,
                "response": response,
                "overall_score": overall_score,
                "prompt": prompt,
                "evaluation": [
                    f"- Coherence: {coherence}",
                    f"- Fluency: {fluency}",
                    f"- Relevance: {relevance}"
                ]
            })

            # Salva risultati parziali
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            # Salva l'offset corrente
            save_offset(idx + 1, offset_file)

            # Rispetta il limite di richieste
            time.sleep(60 / max_requests_per_minute)

        except Exception as e:
            if "rate limit" in str(e).lower():
                print(f"Rate limit raggiunto. Salvataggio stato e uscita...")
                save_offset(idx, offset_file)
                break
            else:
                print(f"Errore durante l'elaborazione del record {idx}: {e}")
                save_offset(idx, offset_file)
                break

    print(f"Elaborazione completata o interrotta. Risultati salvati in {output_file}")


if __name__ == "__main__":
    # Configura i file e parametri
    INPUT_FILE = "datasets/fed_data.json"
    OUTPUT_FILE = "results_fed.json"
    OFFSET_FILE = "offset_fed.json"
    SINGLE_TEMPLATE_PATH = "prompts/fed_single_response.txt"
    FULL_TEMPLATE_PATH = "prompts/fed_full_dialogue.txt"

    # Esegui il processo
    run_test(
        input_file=INPUT_FILE,
        single_template_path=SINGLE_TEMPLATE_PATH,
        full_template_path=FULL_TEMPLATE_PATH,
        output_file=OUTPUT_FILE,
        offset_file=OFFSET_FILE
    )