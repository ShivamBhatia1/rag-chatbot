                          #  RAG Chatbot with Streaming Responses – Legal Document QA

An AI-powered chatbot built using Retrieval-Augmented Generation (RAG) to answer questions based on custom documents (e.g., legal terms, privacy policy). Uses FAISS for semantic retrieval and Mistral-7B-Instruct (GGUF) running locally via LM Studio, with real-time streaming responses in a Streamlit interface.

This is an end-to-end Retrieval-Augmented Generation (RAG) chatbot that answers user queries using a legal PDF document (e.g., Terms & Conditions). It uses local embeddings for retrieval and runs a fine-tuned LLM (**Mistral-7B-Instruct**) via **LM Studio** for generation. The chatbot supports **token-by-token streaming** in a friendly **Streamlit UI**.

demo Video link :- https://drive.google.com/file/d/1G3AjVHRdzEzjm_0C6dlaz1NkZCNQR0aP/view?usp=sharing
---

##  Project Architecture

```
           --------------------
          |  PDF Document      |
           ---------+----------
                    |
                    v
      ------------------------------- 
     |   Chunker (SpaCy + PyMuPDF)   |
      -------------------------------
                    |
                    v
      --------------------------------------
     |   Embeddings (MiniLM) + FAISS index  |
      --------------------------------------
                    |
                    v
           --------------------- 
          |   Retriever (FAISS) |
           --------------------- 
                    |
                    v
    --------------------------------------------- 
   | Generator (Prompt + Mistral LLM via API)    |
    ---------------------------------------------
                    |
                    v
           ----------------------
          |  Streamlit Chat App  |
           ----------------------
```

---

##  How to Run the Project

###  1. Clone the Project

```bash
git clone <your_repo_url>
cd rag-chatbot
```

---

###  2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate  # For Windows
```

---

###  3. Install Requirements

Create a file named `requirements.txt` with the following content:

```txt
streamlit
faiss-cpu
PyMuPDF
spacy
sentence-transformers
requests
```

Then run:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

###  4. Preprocess PDF → Chunks

Place your PDF in the `data/` folder as `document.pdf`.

Then run:

```bash
python src/chunker.py
```

---

### ✅ 5. Create Embeddings + FAISS Index

```bash
python src/embedder.py
```

---

### ✅ 6. Run the Mistral Model (Locally)

- Download [Mistral-7B-Instruct GGUF Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF)
- Load it into [LM Studio](https://lmstudio.ai)
- Go to **Chat tab** → Select model → ✅ Enable API Server
- It should say:  
  `API Server running at http://localhost:1234`

---

### ✅ 7. Launch the Streamlit Chat App

```bash
streamlit run app.py
```

Then go to `http://localhost:8501`

---

## 🤖 Sample Queries

Sample Queries below includes example questions, their respective answers, and the source chunk number.

1	What is the main purpose of the User Agreement?( success)

Bot:   The main purpose of the User Agreement is to set out the terms on which eBay offers you access to and use of their Services. It includes an overview of their policies, the Mobile Application Terms of Use, and additional terms posted on and in their sites, applications, tools, and services. The User Agreement outlines the terms that users must comply with when accessing or using eBay's Services.
    
sourceChunks :	0, 58, 15 ( success)


2	What happens if a user violates eBay's policies?	( success)

Bot:    If a user violates eBay's policies, they may face various consequences such as having their account suspended or terminated, being charged additional fees, and/or having legal action taken against them. It is important for users to carefully read and understand eBay's policies before using the platform to avoid any potential issues.

sourceChunks :	34, 56, 57


3	What happens if a seller tries to sell outside of eBay?	( success)

Bot:  If we believe you are violating our policies prohibiting Offers to buy or sell outside of eBay, you may be subject to a range of actions, including limits on your buying and selling privileges, restrictions on listings and account features, suspension of your account, application of fees, and recovery of expenses for policy monitoring and enforcement. Additionally, as provided below in the Fees and Taxes section, you may be charged final value fees.

sourceChunks :	34, 61, 56 ( success)

---

##  Model and Embedding Choices

| Component        | Model/Tool                     | Reason Chosen                            |
|------------------|--------------------------------|-------------------------------------------|
| Embedding Model  | `all-MiniLM-L6-v2`             | Lightweight, fast, good semantic quality  |
| Vector Database  | `FAISS`                        | Fast and reliable similarity search       |
| LLM Generator    | `Mistral-7B-Instruct` (GGUF)   | High-quality answers, local, private and best for my laptop ram and gpu.     |
| Tokenization     | `spaCy (en_core_web_sm)`       | Accurate sentence splitting               |

---

##  Features Implemented

- ✅ PDF parsing and chunking (100–300 words)
- ✅ Sentence-aware chunk splitting using SpaCy
- ✅ Embedding with MiniLM + FAISS vector DB
- ✅ Retrieval of top-k relevant chunks
- ✅ Prompt templating with citations
- ✅ Streaming responses from local LLM
- ✅ Streamlit chatbot with chat history and chunk sources

---

## 🖼️ Screenshots

![alt text](<Screenshot 2025-07-11 214837-1.jpg>)  
![alt text](<Screenshot 2025-07-11 2149503333333-1.jpg>) 
![alt text](<Screenshot 2025-07-11 2151014-1.jpg>) 
![alt text](<Screenshot 2025-07-11 21513455-1.jpg>)

All screenshots are located inn z_screenshots Folder.

---

## 🎥 Demo Video

> 🎬 Watch the Demo Video here -   https://drive.google.com/file/d/1G3AjVHRdzEzjm_0C6dlaz1NkZCNQR0aP/view?usp=sharing    
---

GitHub repository URL with complete source code:-


## 🙋 Author

**Name:** Shivam Bhatia
**Email:** shivambhatia800@gmail.com
**Mobile No.** +91-9729288984
**Project:** Junior AI Engineer Assignment  
**Company:** Amlgo Labs  
**Date:** 11th July 2025

