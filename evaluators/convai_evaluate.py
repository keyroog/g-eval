import json
from g_eval import GEvalAPI
import time

def process_convai_data(file_path, g_eval, single_template_path, output_path):
    """
    Processa il dataset ConvAI2 per valutare la qualità delle risposte nei dialoghi.

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

    for dialog_id, example in enumerate(data):
        dialog = example['dialog']
        eval_score = example['eval_score']

        if len(dialog) > 1 and eval_score is not None:
            dialog_text = "\n".join([turn['text'] for turn in dialog])
            prompt = g_eval.generate_prompt(single_template, dialog_text, "")

            evaluations = g_eval.send_request(prompt)

            overall = None
            for evaluation in evaluations:
                if "Overall" in evaluation and ":" in evaluation:
                    try:
                        overall = int(evaluation.split(":")[1].strip())
                    except ValueError:
                        print(f"Errore nel parsing di Overall: {evaluation}")

            results.append({
                "dialog_id": dialog_id,
                "context": dialog_text,
                "overall_score": eval_score,
                "prompt": prompt,
                "evaluation": {
                    "Overall": overall,
                },
                "level": "dialog-level"
            })

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Risultati ConvAI2 salvati in {output_path}")

    return results