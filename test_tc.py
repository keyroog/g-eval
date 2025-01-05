import json
import os
from dotenv import load_dotenv
from g_eval import GEvalAPI
import time


def load_offset(offset_file):
    """
    Carica l'offset dal file salvato.
    Restituisce un dizionario con il record corrente e la risposta corrente.
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


def run_test_tc(input_file, single_template_path, output_file, offset_file, max_requests_per_minute=10):
    """
    Esegue il test sul dataset TC_USR, rispettando i limiti di rate limit e salvando l'offset.
    """
    load_dotenv()
    API_KEY = os.getenv("OPENAI_API_KEY")
    if not API_KEY:
        raise ValueError("API key not found. Set OPENAI_API_KEY in your .env file.")
    
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
                    
                    fluency, consistency, coherence, relevance = None, None, None, None
                    for evaluation in evaluations:
                        lines = evaluation.split("\n")
                        for line in lines:
                            if "Fluency" in line and ":" in line:
                                fluency = int(line.split(":")[1].strip())
                            if "Consistency" in line and ":" in line:
                                consistency = int(line.split(":")[1].strip())
                            if "Coherence" in line and ":" in line:
                                coherence = int(line.split(":")[1].strip())
                            if "Relevance" in line and ":" in line:
                                relevance = int(line.split(":")[1].strip())
                    
                    results.append({
                        "context": full_conversation,
                        "response": response if response else None,
                        "system": system,
                        "overall_score": overall_score,
                        "prompt": prompt,
                        "evaluation": {
                            "Fluency": fluency,
                            "Consistency": consistency,
                            "Coherence": coherence,
                            "Relevance": relevance
                        },
                        "level": "turn-level"
                    })
                    
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=4)
                    
                    save_offset(record_idx, response_idx + 1, offset_file)
                    time.sleep(60 / max_requests_per_minute)

                except Exception as e:
                    print(f"Error processing response {response_idx} in record {record_idx}: {e}")
                    save_offset(record_idx, response_idx, offset_file)
                    raise e

            offset["response"] = 0  # Reset the response offset for the next record

        except Exception as e:
            print(f"Error processing record {record_idx}: {e}")
            save_offset(record_idx, offset["response"], offset_file)
            break
    
    print(f"Processing completed or interrupted. Results saved in {output_file}")


if __name__ == "__main__":
    INPUT_FILE = "datasets/tc_usr_data.json"
    OUTPUT_FILE = "results_tc_usr.json"
    OFFSET_FILE = "offset_tc_usr.json"
    SINGLE_TEMPLATE_PATH = "prompts/tc_usr_single_response.txt"
    
    run_test_tc(
        input_file=INPUT_FILE,
        single_template_path=SINGLE_TEMPLATE_PATH,
        output_file=OUTPUT_FILE,
        offset_file=OFFSET_FILE
    )