import os
import re
import fitz  # PyMuPDF
import spacy

# Load spaCy's English tokenizer, tagger, parser, NER, and word vectors
nlp = spacy.load("en_core_web_sm")

def read_pdf_text(path):
    """Extract text from all pages in a PDF file."""
    with fitz.open(path) as doc:
        full_text = ""
        for page in doc:
            full_text += page.get_text()
    return full_text

def simplify_whitespace(text):
    """Remove excessive whitespace and newlines."""
    return re.sub(r'\s+', ' ', text).strip()

def split_into_chunks(text, max_words=200):
    """Break text into chunks without exceeding max word count per chunk."""
    parsed = nlp(text)
    sentences = [sent.text.strip() for sent in parsed.sents]

    chunks = []
    current_chunk = []
    words_in_chunk = 0

    for sent in sentences:
        words_in_sent = len(sent.split())
        if words_in_chunk + words_in_sent <= max_words:
            current_chunk.append(sent)
            words_in_chunk += words_in_sent
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            words_in_chunk = words_in_sent

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

def export_chunks(chunks, output_dir):
    """Save each chunk as a separate .txt file."""
    os.makedirs(output_dir, exist_ok=True)
    for idx, chunk in enumerate(chunks):
        filename = f"chunk_{idx}.txt"
        path = os.path.join(output_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(chunk)

def run_pipeline(pdf_path, save_dir):
    print("🔧 Processing PDF...")
    raw_text = read_pdf_text(pdf_path)
    cleaned = simplify_whitespace(raw_text)
    parts = split_into_chunks(cleaned)
    export_chunks(parts, save_dir)
    print(f"✅ Done! {len(parts)} chunks written to '{save_dir}'.")

if __name__ == "__main__":
    run_pipeline("data/document.pdf", "chunks")
