import json
from g_eval import GEvalAPI

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
        responses = instance["responses"]

        for response_entry in responses:
            response = response_entry["response"]
            overall_scores = response_entry.get("Overall", [])
            
            # Calcolo dell'overall_score
            overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None

            # Genera il prompt
            prompt = g_eval.generate_prompt(single_template, context, response)

            # Effettua la richiesta al modello
            evaluations = g_eval.send_request(prompt)

            # Estrai i punteggi di Coherence, Fluency e Relevance
            coherence, fluency, relevance = None, None, None
            for evaluation in evaluations:
                lines = evaluation.split("\n")
                for line in lines:
                    if "Coherence" in line and ":" in line:
                        coherence = int(line.split(":")[1].strip())
                    elif "Fluency" in line and ":" in line:
                        fluency = int(line.split(":")[1].strip())
                    elif "Relevance" in line and ":" in line:
                        relevance = int(line.split(":")[1].strip())

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

    # Salva i risultati in un file JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Risultati salvati in {output_path}")