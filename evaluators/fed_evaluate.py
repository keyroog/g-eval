import json
from g_eval import GEvalAPI

def process_fed_data(file_path, g_eval, single_template_path, full_template_path, output_path):
    """
    Processa il dataset FED, calcola l'overall score dalle annotations e ottiene i valori di valutazione dal modello.
    Args:
        file_path (str): Percorso al file JSON del dataset.
        g_eval (GEvalAPI): Istanza di GEvalAPI.
        single_template_path (str): Percorso al template per risposte singole.
        full_template_path (str): Percorso al template per dialoghi completi.
        output_path (str): Percorso per salvare i risultati.
    """
    # Carica i template
    single_template = g_eval.load_prompt_template(single_template_path)
    full_template = g_eval.load_prompt_template(full_template_path)

    with open(file_path, "r") as f:
        data = json.load(f)

    results = []

    for instance in data:
        context = instance["context"]
        response = instance.get("response", "").strip()
        annotations = instance.get("annotations", {})

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
            lines = evaluation.split("\n")
            for line in lines:
                if "Coherence" in line and ":" in line:
                    try:
                        coherence = int(line.split(":")[1].strip())
                    except ValueError:
                        print(f"Errore nel parsing di Coherence: {line}")
                if "Fluency" in line and ":" in line:
                    try:
                        fluency = int(line.split(":")[1].strip())
                    except ValueError:
                        print(f"Errore nel parsing di Fluency: {line}")
                if "Relevance" in line and ":" in line:
                    try:
                        relevance = int(line.split(":")[1].strip())
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

    # Salva i risultati in un file JSON
    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Risultati salvati in {output_path}")