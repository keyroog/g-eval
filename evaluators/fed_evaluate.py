import json
from g_eval import GEvalAPI

def process_fed_data(file_path, g_eval, single_template_path, full_template_path, output_path):
    single_template = g_eval.load_prompt_template(single_template_path)
    full_template = g_eval.load_prompt_template(full_template_path)

    with open(file_path, "r") as f:
        data = json.load(f)

    results = []

    for instance in data:
        context = instance["context"]
        response = instance.get("response", "").strip()
        annotations = instance.get("annotations", {})
        system = instance.get("system", "")

        overall_scores = annotations.get("Overall", [])
        overall_score = sum(overall_scores) / len(overall_scores) if overall_scores else None

        conversation = context.split("\n")
        conversation = [line.replace("User: ", "").replace("System: ", "").strip() for line in conversation]

        if response:
            full_conversation = " ".join(conversation) + " " + response.replace("System: ", "").strip()
            prompt = g_eval.generate_prompt(single_template, full_conversation, response.replace("System: ", "").strip())
        else:
            full_conversation = " ".join(conversation)
            prompt = g_eval.generate_prompt(full_template, full_conversation, "")

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
            "response": response if response else None,
            "system": system,
            "overall_score": overall_score,
            "prompt": prompt,
            "evaluation": {
                "Overall": overall,
            },
            "level": "turn-level" if response else "dialog-level"
        })
        max_request_per_minute = 10
        time.sleep(60 / max_request_per_minute)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Risultati salvati in {output_path}")