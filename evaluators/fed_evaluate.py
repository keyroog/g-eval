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

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"Risultati salvati in {output_path}")