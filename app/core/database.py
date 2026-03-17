import chromadb
import os
from dotenv import load_dotenv

load_dotenv()

class DukyDatabase:
    def __init__(self, path=None, collection_name="dukyai_knowledge"):
        self.path = path or os.getenv("CHROMA_DB_PATH", "./chroma_db")
        self.client = chromadb.PersistentClient(path=self.path)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def upsert_chunks(self, ids, documents, metadatas, embeddings):
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
            embeddings=embeddings
        )

    def query(self, query_embeddings, n_results=3):
        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )

    def count(self):
        return self.collection.count()

    def get_existing_ids(self):
        return set(self.collection.get()['ids'])
