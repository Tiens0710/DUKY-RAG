from google import genai
import os
import time
from dotenv import load_dotenv

load_dotenv()

class DukyRAG:
    def __init__(self, api_key=None, model_name="gemini-3-flash-preview", embed_model="gemini-embedding-2-preview"):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or arguments")
        
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = model_name
        self.embed_model = embed_model

    def get_embedding(self, text):
        response = self.client.models.embed_content(
            model=self.embed_model,
            contents=text,
            config={'task_type': 'RETRIEVAL_QUERY'}
        )
        return response.embeddings[0].values

    def generate_answer_stream(self, question, context):
        prompt = f"""Bạn là trợ lý hỗ trợ người dùng Duky AI.
Dựa vào thông tin sau để trả lời câu hỏi. Chỉ dùng thông tin được cung cấp, không bịa thêm.

=== THÔNG TIN ===
{context}

=== CÂU HỎI ===
{question}

=== TRẢ LỜI ==="""
        
        return self.client.models.generate_content_stream(
            model=self.model_name,
            contents=prompt
        )

    def call_with_retry(self, fn, *args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1: raise e
                print(f"⚠️ Đang thử lại... ({attempt+1}/3)")
                time.sleep(5)
