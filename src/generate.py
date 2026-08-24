"""Answer generation via the Hugging Face Inference API.

Requires an HF_TOKEN environment variable (a free account token from
huggingface.co/settings/tokens is enough). The free serverless tier rotates
which models are available, so DEFAULT_MODEL may need swapping out over time
- if generation starts failing with a model-not-found error, that's almost
always why. Check https://huggingface.co/models?other=text-generation for a
currently-available instruction-tuned model under ~10B parameters.
"""
import os

from huggingface_hub import InferenceClient

from .vector_store import SearchResult

DEFAULT_MODEL = os.environ.get("ASK_MY_RESEARCH_MODEL", "Qwen/Qwen2.5-7B-Instruct")

SYSTEM_PROMPT = (
    "You are a research assistant. Answer the question using only the provided "
    "excerpts from the user's uploaded papers. If the excerpts don't contain "
    "enough information to answer, say so directly rather than guessing. Cite "
    "which source and page each part of your answer comes from."
)


def build_context(results: list[SearchResult]) -> str:
    blocks = []
    for r in results:
        blocks.append(f"[{r.chunk.source}, page {r.chunk.page}]\n{r.chunk.text}")
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, results: list[SearchResult], model: str | None = None) -> str:
    if not results:
        return "No relevant passages were found in the uploaded documents for this question."

    context = build_context(results)
    user_prompt = (
        f"Excerpts from uploaded papers:\n\n{context}\n\n---\n\nQuestion: {question}"
    )

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set. Generate a free token at "
            "huggingface.co/settings/tokens and set it before running the app."
        )

    client = InferenceClient(token=token)
    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        model=model or DEFAULT_MODEL,
        max_tokens=500,
        temperature=0.2,
    )
    return response.choices[0].message.content
