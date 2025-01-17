import json
import os
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
            return json.load(f)
    except FileNotFoundError:
        return {"record": 0, "response": 0}


def save_offset(record_idx, response_idx, offset_file):
    """
    Salva l'offset corrente.
    """
    with open(offset_file, "w") as f:
        json.dump({"record": record_idx, "response": response_idx}, f, indent=4)


def load_results(output_file):
    """
    Carica i risultati esistenti dal file di output.
    """
    try:
        with open(output_file, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def run_test_pc(input_file, single_template_path, output_file, offset_file, api_keys_file, max_requests_per_minute=10):
    """
    Esegue il test sul dataset PC_USR, rispettando i limiti di rate limit e salvando l'offset.
    """
    api_keys_data = load_api_keys(api_keys_file)
    API_KEY = get_current_api_key(api_keys_data)

    MODEL = "gpt-4o-mini"
    g_eval = GEvalAPI(api_key=API_KEY, model=MODEL)

    with open(input_file, "r") as f:
        data = json.load(f)

    offset = load_offset(offset_file)
    print(f"Resuming from record {offset['record']} and response {offset['response']} of {len(data)}.")

    results = load_results(output_file)
    single_template = g_eval.load_prompt_template(single_template_path)

    for record_idx, instance in enumerate(data[offset["record"]:], start=offset["record"]):
        try:
            print(f"Processing record {record_idx + 1}/{len(data)}...")
            context = instance["context"]
            fact = instance["fact"]
            responses = instance["responses"]

            for response_idx, response_entry in enumerate(responses[offset["response"]:], start=offset["response"]):
                try:
                    print(f"Processing response {response_idx + 1}/{len(responses)}...")
                    response = response_entry["response"]
                    system = response_entry["model"]
                    overall_scores = response_entry.get("Overall", [])
                    overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None

                    conversation = context.split("\n")
                    conversation = [line.replace("User: ", "").replace("System: ", "").strip() for line in conversation]
                    full_conversation = " ".join(conversation) + " " + response.replace("System: ", "").strip()

                    prompt = g_eval.generate_prompt(single_template, full_conversation, response, fact)
                    evaluations = g_eval.send_request(prompt)

                    overall = None
                    for evaluation in evaluations:
                        if "Overall" in evaluation and ":" in evaluation:
                            try:
                                overall = int(evaluation.split(":")[1].strip())
                            except ValueError:
                                print(f"Errore nel parsing di Overall: {evaluation}")

                    results.append({
                        "context": full_conversation,
                        "response": response,
                        "system": system,
                        "overall_score": overall_score,
                        "prompt": prompt,
                        "evaluation": {
                            "Overall": overall,
                        },
                        "level": "turn-level"
                    })

                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=4)

                    save_offset(record_idx, response_idx + 1, offset_file)
                    time.sleep(60 / max_requests_per_minute)

                except Exception as e:
                    print(f"Error processing response {response_idx} in record {record_idx}: {e}")

                    if "rate limit" in str(e).lower() or "invalid api key" in str(e).lower():
                        print("API key esaurita o non valida. Cambiando chiave...")
                        api_keys_data = rotate_api_key(api_keys_data)
                        save_api_keys(api_keys_file, api_keys_data)

                        # Aggiorna l'API key
                        API_KEY = get_current_api_key(api_keys_data)
                        g_eval = GEvalAPI(api_key=API_KEY, model=MODEL)
                        print(f"Nuova API key selezionata: {API_KEY}")
                    else:
                        save_offset(record_idx, response_idx, offset_file)
                        raise e

            offset["response"] = 0  # Reset per il prossimo record

        except Exception as e:
            print(f"Error processing record {record_idx}: {e}")
            save_offset(record_idx, offset["response"], offset_file)
            break

    if record_idx == len(data) - 1:
        print("Processing completed successfully.")
    else:
        #retry
        run_test_pc(input_file, single_template_path, output_file, offset_file, api_keys_file, max_requests_per_minute)

if __name__ == "__main__":
    INPUT_FILE = "datasets/pc_usr_data.json"
    OUTPUT_FILE = "results_pc_usr.json"
    OFFSET_FILE = "offset_pc_usr.json"
    SINGLE_TEMPLATE_PATH = "prompts/tc_usr_single_response.txt"
    API_KEYS_FILE = "api_keys.json"

    run_test_pc(
        input_file=INPUT_FILE,
        single_template_path=SINGLE_TEMPLATE_PATH,
        output_file=OUTPUT_FILE,
        offset_file=OFFSET_FILE,
        api_keys_file=API_KEYS_FILE
    )