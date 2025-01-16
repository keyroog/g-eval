import json
import os
import tqdm
import time
from dotenv import load_dotenv
from g_eval import GEvalAPI


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


def run_test(input_file, full_template_path, output_file, offset_file, max_requests_per_minute=10):
    """
    Esegue il processo sul dataset rispettando i limiti di rate limit e salvando l'offset.
    """
    # Carica l'API key
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("L'API key non è stata trovata. Assicurati che OPENAI_API_KEY sia definita nel file .env.")

    # Inizializza l'API GEval
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
                # Costruzione del contesto del dialogo
                dialog_text = "\n".join([turn['text'] for turn in dialog])
                conversation = " ".join(dialog_text.split("\n"))

                annotation = int(example['eval_score'])

                prompt = g_eval.generate_prompt(full_template, conversation, "")

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
            save_offset(dialog_id, offset_file)
            break

    print(f"Elaborazione completata. Risultati salvati in {output_file}")


if __name__ == "__main__":
    # Configura i file e parametri
    INPUT_FILE = "datasets/convai2_data.json"
    OUTPUT_FILE = "results_convai2.json"
    OFFSET_FILE = "offset_convai2.json"
    FULL_TEMPLATE_PATH = "prompts/fed_full_dialogue.txt"
    MAX_REQUESTS_PER_MINUTE = 10

    # Esegui il processo
    run_test(
        input_file=INPUT_FILE,
        full_template_path=FULL_TEMPLATE_PATH,
        output_file=OUTPUT_FILE,
        offset_file=OFFSET_FILE,
        max_requests_per_minute=MAX_REQUESTS_PER_MINUTE
    )