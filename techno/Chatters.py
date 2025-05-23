import openai  
import tiktoken
from config import OPENAI_API_KEY

openai.api_key = OPENAI_API_KEY


class OpenAIChatter:
    tl = {
        "gpt-4": 8192,
        "gpt-4-0613": 8192,
        "gpt-4-32k": 32768,
        "gpt-4-32k-0613": 32768,
        "gpt-4-1106-preview": 128000,  
        "gpt-4-turbo": 128000,        
        "gpt-3.5-turbo": 4096,
        "gpt-3.5-turbo-0613": 4096,
        "gpt-3.5-turbo-1106": 16385,
    }

    def __init__(self, context, query, chat_model):
        self.model = chat_model
        self.token_limit = self.tl.get(self.model, 8192)
        self.prompt = f"""
        You are a helpful assistant. Use the context below 
        to answer the question.

        Context:
        {context}

        Question:
        {query}

        If the context does not provide enough information, respond:
        "I'm sorry, I couldn't find an answer in the documents."
        """

    def check_prompt_length(self):
        """
        Checks that the prompt is within the token limit for the model specified
        """
        encoding = tiktoken.encoding_for_model(self.model)
        tokens = len(encoding.encode(self.prompt))
        if tokens > self.token_limit:
            raise ValueError("Prompt is too long, maybe due to the context containing large tables with numbers. Try with a smaller value for the --top_k option")
    
    def ask_question(self):
        """
        Returns the answer of the LLM 
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": self.prompt}]
        )
        return response.choices[0].message.content


