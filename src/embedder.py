import os
import faiss
import pickle
from sentence_transformers import SentenceTransformer

# Set up the embedding model
MODEL_NAME = "all-MiniLM-L6-v2"
embedder = SentenceTransformer(MODEL_NAME)

def get_chunks_from_folder(folder_path):
    """Read and return all text chunks from the specified folder."""
    chunk_list = []
    for fname in sorted(os.listdir(folder_path)):
        path = os.path.join(folder_path, fname)
        with open(path, "r", encoding="utf-8") as f:
            chunk_list.append(f.read())
    return chunk_list

def generate_embeddings(texts):
    """Turn a list of texts into vector embeddings."""
    return embedder.encode(texts)

def save_index_and_metadata(vectors, index_path, meta_path, raw_texts):
    """Save vectors to FAISS and store raw text as metadata."""
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    
    faiss.write_index(index, index_path)

    with open(meta_path, "wb") as meta_out:
        pickle.dump(raw_texts, meta_out)

def build_faiss_index():
    # Input/output paths
    chunk_dir = "chunks"
    output_dir = "vectordb"
    index_path = os.path.join(output_dir, "index.faiss")
    meta_path = os.path.join(output_dir, "chunk_text.pkl")

    # Read chunks
    chunks = get_chunks_from_folder(chunk_dir)
    print(f"📄 Found {len(chunks)} chunks.")

    # Get vector representations
    vectors = generate_embeddings(chunks)
    print(f"📈 Generated embeddings with shape: {vectors.shape}")

    # Save everything
    os.makedirs(output_dir, exist_ok=True)
    save_index_and_metadata(vectors, index_path, meta_path, chunks)
    print(f"✅ All done! Files saved in '{output_dir}/'")

if __name__ == "__main__":
    build_faiss_index()
