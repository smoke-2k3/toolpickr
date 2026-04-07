from typing import List, Optional
from toolpickr.embeddings.base import EmbeddingProvider

try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError("google-genai is required. Please install it using 'pip install google-genai'")

class GeminiEmbeddings(EmbeddingProvider):
    def __init__(self, api_key: Optional[str] = None, model: str = 'gemini-embedding-001'):
        """
        Initializes the Gemini Embedding provider. 
        If api_key is not passed, it automatically looks for the GEMINI_API_KEY environment variable.
        """
        self.client = genai.Client(api_key=api_key)
        self.model = model

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single string into a vector."""
        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_DOCUMENT"
            )
        )
        # response.embeddings is a list. We take the first one and return its values.
        return response.embeddings[0].values

    def embed_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """Embeds a list of strings efficiently by processing them in manageable batches."""
        all_embeddings = []
        
        # Iterate through the texts in increments of batch_size
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            response = self.client.models.embed_content(
                model=self.model,
                contents=batch,
                config=types.EmbedContentConfig(
                    task_type="RETRIEVAL_DOCUMENT"
                )
            )
            
            # Extract the actual values list out of each embedding object and append to results
            batch_embeddings = [emb.values for emb in response.embeddings]
            all_embeddings.extend(batch_embeddings)
        #print("ALL embeddings: ", all_embeddings)
        return all_embeddings