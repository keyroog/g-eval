import json
import os
import tqdm
import time
from dotenv import load_dotenv
from g_eval import GEvalAPI


def load_api_keys(api_keys_file):
    """
    Carica le API key dal file JSON.
    """
    with open(api_keys_file, "r") as f:
        return json.load(f)


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
    """
    try:
        with open(offset_file, "r") as f:
            return json.load(f).get("offset", 0)
    except FileNotFoundError:
        return 0


def save_offset(offset, offset_file):
    """
    Salva l'offset corrente.
    """
    with open(offset_file, "w") as f:
        json.dump({"offset": offset}, f, indent=4)


def load_results(output_file):
    """
    Carica i risultati esistenti dal file di output.
    """
    try:
        with open(output_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def run_test_convai2(input_file, full_template_path, output_file, offset_file, api_keys_file, max_requests_per_minute=10):
    """
    Esegue il processo sul dataset rispettando i limiti di rate limit, salvando l'offset e ruotando le API key.
    """
    api_keys_data = load_api_keys(api_keys_file)
    API_KEY = get_current_api_key(api_keys_data)

    MODEL = "gpt-4o-mini"
    g_eval = GEvalAPI(api_key=API_KEY, model=MODEL)

    # Carica il dataset
    with open(input_file, "r") as f:
        convai2_data = json.load(f)

    # Carica l'offset corrente
    offset = load_offset(offset_file)
    print(f"Ripresa dal record {offset} su {len(convai2_data)}.")

    # Carica i risultati esistenti
    results = load_results(output_file)
    full_template = g_eval.load_prompt_template(full_template_path)

    # Elaborazione dei dialoghi
    for dialog_id, example in enumerate(tqdm.tqdm(convai2_data[offset:], desc="Elaborazione dialoghi"), start=offset):
        try:
            dialog = example['dialog']
            if len(dialog) > 1 and example['eval_score'] is not None:
                print(f"Elaborazione dialogo {dialog_id + 1}/{len(convai2_data)}...")
                # Costruzione del contesto del dialogo
                dialog_text = "\n".join([turn['text'] for turn in dialog])
                conversation = " ".join(dialog_text.split("\n"))

                annotation = int(example['eval_score'])

                # Generazione del prompt
                prompt = g_eval.generate_prompt(full_template, conversation, "")

                # Invio richiesta al modello
                evaluations = g_eval.send_request(prompt)

                overall = None
                for evaluation in evaluations:
                    if "Overall" in evaluation and ":" in evaluation:
                        try:
                            overall = int(evaluation.split(":")[1].strip())
                        except ValueError:
                            print(f"Errore nel parsing di Overall: {evaluation}")

                # Salvataggio del risultato
                results.append({
                    "dialog_id": dialog_id,
                    "context": conversation,
                    "overall_score": annotation,
                    "prompt": prompt,
                    "evaluation": {
                        "Overall": overall,
                    },
                    "level": "dialog-level"
                })

                # Salva risultati parziali
                with open(output_file, "w") as f:
                    json.dump(results, f, indent=4)

                # Salva l'offset corrente
                save_offset(dialog_id + 1, offset_file)

                # Rispetta il limite di richieste
                time.sleep(60 / max_requests_per_minute)

        except Exception as e:
            print(f"Errore durante l'elaborazione del dialog {dialog_id}: {e}")

            if "rate limit" in str(e).lower() or "invalid api key" in str(e).lower():
                print("API key esaurita o non valida. Cambiando chiave...")
                api_keys_data = rotate_api_key(api_keys_data)
                save_api_keys(api_keys_file, api_keys_data)

                # Aggiorna l'API key
                API_KEY = get_current_api_key(api_keys_data)
                g_eval = GEvalAPI(api_key=API_KEY, model=MODEL)
                print(f"Nuova API key selezionata: {API_KEY}")
                break
            else:
                save_offset(dialog_id, offset_file)
                break

    if(dialog_id < len(convai2_data)):
        run_test_convai2(input_file, full_template_path, output_file, offset_file, api_keys_file, max_requests_per_minute)
    else:
        print("Elaborazione completata con successo.")


if __name__ == "__main__":
    # Configura i file e parametri
    INPUT_FILE = "datasets/convai2_data.json"
    OUTPUT_FILE = "results_convai2.json"
    OFFSET_FILE = "offset_convai2.json"
    FULL_TEMPLATE_PATH = "prompts/fed_full_dialogue.txt"
    API_KEYS_FILE = "api_keys.json"
    MAX_REQUESTS_PER_MINUTE = 10

    run_test_convai2(
        input_file=INPUT_FILE,
        full_template_path=FULL_TEMPLATE_PATH,
        output_file=OUTPUT_FILE,
        offset_file=OFFSET_FILE,
        api_keys_file=API_KEYS_FILE,
        max_requests_per_minute=MAX_REQUESTS_PER_MINUTE
    )