import faiss
import pickle
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_file="vectordb/index.faiss", meta_file="vectordb/chunk_text.pkl"):
        print("🔄 Loading FAISS index and chunk metadata...")

        # Load embedding model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

        # Load FAISS index
        self.index = faiss.read_index(index_file)

        # Load stored chunk texts
        with open(meta_file, 'rb') as f:
            self.chunk_texts = pickle.load(f)

    def search(self, query, top_k=3):
        # Convert query to embedding
        query_embedding = self.model.encode([query])

        # Search for nearest neighbors
        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for i, idx in enumerate(indices[0]):
            result = {
                "chunk_id": idx,
                "text": self.chunk_texts[idx],
                "score": float(distances[0][i])
            }
            results.append(result)

        return results

# Example usage
if __name__ == "__main__":
    retriever = Retriever()
    query = input("❓ Enter your question: ")
    results = retriever.search(query)

    print("\n🔍 Top matching chunks:")
    for res in results:
        print(f"\n🔹 Chunk #{res['chunk_id']} (Score: {res['score']:.2f})")
        print(res['text'])
