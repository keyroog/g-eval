import json
import os
from dotenv import load_dotenv
from g_eval import GEvalAPI
from evaluators.fed_evaluate import process_fed_data
import time

def load_api_keys(api_keys_file):
    """
    Carica le API key dal file JSON.
    """
    with open(api_keys_file, "r") as f:
        data = json.load(f)
    return data


def save_api_keys(api_keys_file, data):
    """
    Salva lo stato delle API key nel file JSON.
    """
    with open(api_keys_file, "w") as f:
        json.dump(data, f, indent=4)


def get_current_api_key(api_keys_data):
    """
    Restituisce l'API key corrente.
    """
    index = api_keys_data["current_index"]
    return api_keys_data["keys"][index]


def rotate_api_key(api_keys_data):
    """
    Cambia l'API key corrente con la prossima nella lista.
    """
    api_keys_data["current_index"] = (api_keys_data["current_index"] + 1) % len(api_keys_data["keys"])
    return api_keys_data

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


def run_test(input_file, single_template_path, full_template_path, output_file, offset_file, max_requests_per_minute=10):
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

    api_keys_data = load_api_keys("api_keys.json")
    API_KEY = get_current_api_key(api_keys_data)

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

    for idx, instance in enumerate(data[offset:], start=offset):
        try:
            print(f"Elaborazione record {idx + 1}/{len(data)}...")
            response = instance.get("response", "").strip()
            annotations = instance.get("annotations", {})
            context = instance["context"]
            response = instance.get("response", "").strip()
            annotations = instance.get("annotations", {})
            system = instance.get("system", "")

            # Calcolo dell'overall_score come media dei valori in annotations["Overall"]
            overall_scores = annotations.get("Overall", [])
            overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None

            # Normalizzazione del contesto
            conversation = context.split("\n")
            conversation = [line.replace("User: ", "").replace("System: ", "").strip() for line in conversation]

            # Logica per turn-level e dialog-level
            if response:  # Caso turn-level
                full_conversation = " ".join(conversation) + " " + response.replace("System: ", "").strip()
                prompt = g_eval.generate_prompt(single_template, full_conversation, response.replace("System: ", "").strip())
            else:  # Caso dialog-level
                full_conversation = " ".join(conversation)
                prompt = g_eval.generate_prompt(full_template, full_conversation, "")

            # Effettua la richiesta al modello
            evaluations = g_eval.send_request(prompt)

            print(evaluations)
            # Estrai i punteggi di Coherence, Fluency e Relevance con debug
            overall = None
            for evaluation in evaluations:
              if "Overall" in evaluation and ":" in evaluation:
                try:
                  overall = int(evaluation.split(":")[1].strip())
                except ValueError:
                  print(f"Errore nel parsing di Overall: {evaluation}")

            # Salva i risultati
            results.append({
                "context": full_conversation,
                "response": response if response else None,
                "system": system,
                "overall_score": overall_score,
                "prompt": prompt,
                "evaluation": {
                    "Overall": overall,
                },
                "level": "turn-level" if response else "dialog-level"
            })

            # Salva risultati parziali
            with open(output_file, "w") as f:
                json.dump(results, f, indent=4)

            # Salva l'offset corrente
            save_offset(idx + 1, offset_file)

            # Rispetta il limite di richieste
            time.sleep(60 / max_requests_per_minute)

        except Exception as e:
            print(f"Errore durante l'elaborazione del record {idx}: {e}")

            # Controlla se l'errore è legato all'API key
            if "rate limit" in str(e).lower() or "invalid api key" in str(e).lower():
                print("API key esaurita o non valida. Cambiando chiave...")
                api_keys_data = rotate_api_key(api_keys_data)
                save_api_keys("api_keys.json", api_keys_data)

                # Aggiorna l'API key
                API_KEY = get_current_api_key(api_keys_data)
                g_eval = GEvalAPI(api_key=API_KEY, model=MODEL)
                print(f"Nuova API key selezionata: {API_KEY}")
            else:
                save_offset(idx, offset_file)
                time.sleep(2)
                break

    #riprova
    if(idx < len(data)):
        run_test(input_file, single_template_path, full_template_path, output_file, offset_file, max_requests_per_minute)
    else:
        print(f"Elaborazione completata. Risultati salvati in {output_file}")

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