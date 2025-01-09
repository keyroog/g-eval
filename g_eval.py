import openai
import time

class GEvalAPI:
    def __init__(self, api_key, model="gpt-4"):
        self.client = openai.OpenAI(api_key=api_key, base_url="https://api.gpt4-all.xyz/v1")
        self.model = model

    def load_prompt_template(self, file_path):
        with open(file_path, "r") as f:
            return f.read()

    def generate_prompt(self, template, context, response,fact = None):
        to_return = template.replace("{{context}}", context).replace("{{response}}", response)
        if fact:
            to_return = to_return.replace("{{fact}}", fact)
        return to_return
        
    def send_request(self, prompt, n=1, max_tokens=50, temperature=0):
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "system", "content": prompt}],
                    n=n,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0,
                )
                evaluations = [choice.message.content for choice in response.choices]
                return evaluations
            except Exception as e:
                print(f"Errore API: {e}")
                if "limit" in str(e).lower():
                    print("Raggiunto limite di richiesta, attesa in corso...")
                    raise
                else:
                    raise
