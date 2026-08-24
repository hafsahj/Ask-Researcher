"""Ask My Research: upload papers, ask questions, get answers grounded in them."""
import streamlit as st

from src.embeddings import embed_texts
from src.generate import answer_question
from src.ingest import process_pdf
from src.vector_store import VectorStore

st.set_page_config(page_title="Ask My Research", page_icon="📄")
st.title("Ask My Research")
st.caption("Upload PDFs, ask questions, get answers grounded in the actual text with page citations.")

if "store" not in st.session_state:
    st.session_state.store = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

uploaded_files = st.file_uploader("Upload research papers (PDF)", type="pdf", accept_multiple_files=True)

if uploaded_files:
    new_files = [f for f in uploaded_files if f.name not in st.session_state.processed_files]
    if new_files:
        with st.spinner(f"Processing {len(new_files)} new file(s)..."):
            all_chunks = []
            for f in new_files:
                all_chunks.extend(process_pdf(f, f.name))

            if all_chunks:
                embeddings = embed_texts([c.text for c in all_chunks])
                if st.session_state.store is None:
                    st.session_state.store = VectorStore(dim=embeddings.shape[1])
                st.session_state.store.add(all_chunks, embeddings)

            st.session_state.processed_files.update(f.name for f in new_files)

if st.session_state.store:
    st.success(f"{len(st.session_state.store)} chunks indexed from {len(st.session_state.processed_files)} file(s).")

    question = st.text_input("Ask a question about the uploaded papers")
    top_k = st.slider("Passages to retrieve", min_value=1, max_value=10, value=5)

    if question:
        with st.spinner("Retrieving relevant passages..."):
            query_embedding = embed_texts([question])[0]
            results = st.session_state.store.search(query_embedding, k=top_k)

        with st.spinner("Generating answer..."):
            try:
                answer = answer_question(question, results)
            except RuntimeError as e:
                answer = None
                st.error(str(e))

        if answer:
            st.markdown("### Answer")
            st.write(answer)

        if results:
            with st.expander("Retrieved passages"):
                for r in results:
                    st.markdown(f"**{r.chunk.source}, page {r.chunk.page}** (similarity: {r.score:.2f})")
                    st.write(r.chunk.text)
                    st.divider()
else:
    st.info("Upload at least one PDF to get started.")
