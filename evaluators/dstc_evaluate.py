import json
from g_eval import GEvalAPI
import time

def process_dstc_data(file_path, g_eval, single_template_path, output_path):
    """
    Processa il dataset DSTC per valutare le risposte fornite nei dialoghi.
    
    Args:
        file_path (str): Percorso del file JSON di input.
        g_eval (GEvalAPI): Oggetto GEvalAPI per la generazione e valutazione.
        single_template_path (str): Percorso del template per la valutazione.
        output_path (str): Percorso per salvare i risultati.
    """
    single_template = g_eval.load_prompt_template(single_template_path)

    with open(file_path, "r") as f:
        data = json.load(f)

    results = []

    for i, instance in enumerate(data['contexts']):
        context = " ".join(instance)
        response = data["responses"][i]
        overall_score = data["scores"][i]

        prompt = g_eval.generate_prompt(single_template, context, response)
        evaluations = g_eval.send_request(prompt)

        overall = None
        for evaluation in evaluations:
            if "Overall" in evaluation and ":" in evaluation:
                try:
                    overall = int(evaluation.split(":")[1].strip())
                except ValueError:
                    print(f"Errore nel parsing di Overall: {evaluation}")

        results.append({
            "dialog_id": i,
            "context": context,
            "response": response,
            "overall_score": overall_score,
            "prompt": prompt,
            "evaluation": {
                "Overall": overall,
            },
            "level": "turn-level"
        })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Risultati DSTC salvati in {output_path}")

    return results