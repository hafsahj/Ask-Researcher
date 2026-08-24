---
title: Ask My Research
emoji: 📄
colorFrom: blue
colorTo: purple
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
license: mit
---

# Ask-Researcher

Upload PDFs of research papers, ask questions in plain English, get answers grounded in the actual text, with the source file and page cited for every retrieved passage.

## How it works

1. **Ingest** (`src/ingest.py`): extract text per page from uploaded PDFs, split into overlapping chunks.
2. **Embed** (`src/embeddings.py`): each chunk gets a dense vector via `sentence-transformers/all-MiniLM-L6-v2`, running locally rather than through an API.
3. **Index** (`src/vector_store.py`): chunk vectors go into a FAISS index for cosine-similarity search.
4. **Retrieve**: a question is embedded the same way and matched against the index to pull the most relevant chunks.
5. **Generate** (`src/generate.py`): retrieved chunks plus the question are sent to an instruction-tuned model via the Hugging Face Inference API, which answers using only that context, citing source and page.

## Structure

```
├── app.py               # Streamlit UI
├── src/
│   ├── ingest.py          # PDF text extraction + chunking
│   ├── embeddings.py       # dense embeddings (sentence-transformers)
│   ├── vector_store.py      # FAISS index build/search
│   └── generate.py           # HF Inference API call for grounded answers
├── requirements.txt
├── LICENSE
└── README.md
```

## Running locally

```bash
git clone https://github.com/hafsahj/ask-researcher.git
cd ask-researcher
pip install -r requirements.txt
export HF_TOKEN=your_huggingface_token   # free, from huggingface.co/settings/tokens
streamlit run app.py
```

The free Hugging Face Inference tier rotates which models are available. If generation errors out with a model-not-found message, set `ASK_MY_RESEARCH_MODEL` to a currently available instruction-tuned model (under 10B parameters) instead of the default.

## Deploying

This repo's README.md doubles as Hugging Face Spaces config (the YAML block at the top). Push it to a new Space with the Streamlit SDK selected, add `HF_TOKEN` as a Space secret, and it deploys directly, no separate config file needed.

## License

MIT, see [LICENSE](LICENSE).
