import requests

class Generator:
    def __init__(self):
        self.api_url = "http://127.0.0.1:1234/v1/chat/completions"
        self.system_prompt = (
            "You are a helpful assistant. Use only the provided context to answer the user's question. "
            "If you don’t know the answer, say: 'I don’t know based on the provided content.'"
        )

    def format_prompt(self, context_chunks, user_question):
        context = "\n\n".join([f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(context_chunks)])
        prompt = f"Context:\n{context}\n\nQuestion: {user_question}\n\nAnswer:"
        return prompt

    def generate_answer(self, context_chunks, user_question):
        prompt = self.format_prompt(context_chunks, user_question)

        payload = {
            "model": "mistral",  # LM Studio uses 'mistral' as the model ID
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "stream": False,
            "max_tokens": 500
        }

        response = requests.post(self.api_url, json=payload)
        return response.json()['choices'][0]['message']['content']
