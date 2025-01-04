import json
from g_eval import GEvalAPI
import time

def process_tc_usr_data(file_path, g_eval, single_template_path, output_path):
    """
    Processa il dataset TC_USR, calcola l'overall score per ogni risposta e ottiene le valutazioni dal modello.
    Args:
        file_path (str): Percorso al file JSON del dataset.
        g_eval (GEvalAPI): Istanza di GEvalAPI.
        single_template_path (str): Percorso al template per risposte singole.
        output_path (str): Percorso per salvare i risultati.
    """
    # Carica il template
    single_template = g_eval.load_prompt_template(single_template_path)

    with open(file_path, "r") as f:
        data = json.load(f)

    results = []

    for instance in data:
        context = instance["context"]
        fact = instance["fact"]
        responses = instance["responses"]

        for response_entry in responses:
            response = response_entry["response"]
            system = response_entry["model"]
            overall_scores = response_entry.get("Overall", [])
            
            # Calcolo dell'overall_score
            overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None

            conversation = context.split("\n")
            conversation = [line.replace("User: ", "").replace("System: ", "").strip() for line in conversation]

            full_conversation = " ".join(conversation) + " " + response.replace("System: ", "").strip()
            # Genera il prompt
            prompt = g_eval.generate_prompt(single_template, full_conversation, response, fact)

            # Effettua la richiesta al modello
            evaluations = g_eval.send_request(prompt)

            # Estrai i punteggi di Coherence, Fluency e Relevance con debug
            fluency, consistency, coherence, relevance = None, None, None, None
            for evaluation in evaluations:
                lines = evaluation.split("\n")
                for line in lines:
                    if "Fluency" in line and ":" in line:
                        try:
                            fluency = int(line.split(":")[1].strip())
                        except ValueError:
                            print(f"Errore nel parsing di Fluency: {line}")
                    if "Consistency" in line and ":" in line:
                        try:
                            consistency = int(line.split(":")[1].strip())
                        except ValueError:
                            print(f"Errore nel parsing di Coherence: {line}")
                    if "Coherence" in line and ":" in line:
                        try:
                            coherence = int(line.split(":")[1].strip())
                        except ValueError:
                            print(f"Errore nel parsing di Coherence: {line}")
                    if "Relevance" in line and ":" in line:
                        try:
                            relevance = int(line.split(":")[1].strip())
                        except ValueError:
                            print(f"Errore nel parsing di Relevance: {line}")

            # Salva i risultati
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
                "level": "turn-level" if response else "dialog-level"
            })
            max_request_per_minute = 10
            time.sleep(60 / max_request_per_minute)

    # Salva i risultati in un file JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Risultati salvati in {output_path}")