from __future__ import annotations

from typing import Any

import requests

from app.config import GENERATION_MODEL, INSUFFICIENT_EVIDENCE_DISTANCE, OLLAMA_URL


def _ollama_generate(prompt: str) -> str:
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": GENERATION_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 300},
        },
        timeout=90,
    )
    response.raise_for_status()
    payload = response.json()
    return payload.get("response", "").strip()


def build_evidence_prompt(question: str, chunks: list[dict[str, Any]]) -> str:
    """Build a prompt that instructs the model to answer only from supplied evidence."""
    if not chunks:
        return (
            "You are a careful enterprise knowledge assistant. There is no relevant retrieved evidence. "
            "Do not guess. Say clearly that the indexed documents do not provide enough evidence to answer the question. "
            f"Question: {question}"
        )

    evidence_lines = []
    for index, chunk in enumerate(chunks, start=1):
        source_label = f"[{chunk.get('source_name', 'unknown')} p.{int(chunk.get('page_number', 1))}]"
        evidence_lines.append(f"{index}. {source_label}: {chunk.get('text', '')}")

    return (
        "You are a careful enterprise knowledge assistant. Use only the evidence below. "
        "Do not invent facts. If the evidence does not support the answer, say that the indexed documents do not provide enough evidence. "
        "Cite every factual answer using the exact source labels shown in brackets, for example [document.pdf p.5]. "
        "When the evidence is insufficient, explicitly say the answer cannot be supported by the available retrieved passages.\n\n"
        f"Evidence:\n{chr(10).join(evidence_lines)}\n\nQuestion: {question}\n\nAnswer:"
    )


def generate_answer(question: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """Return an answer and the source citations for the current question."""
    if not chunks:
        return {
            "answer": "I could not find enough evidence in the indexed documents to answer this question reliably.",
            "citations": [],
            "status": "insufficient_evidence",
        }

    best_distance = float(chunks[0].get("distance", 1.0))
    if best_distance > INSUFFICIENT_EVIDENCE_DISTANCE:
        return {
            "answer": "I could not find enough evidence in the indexed documents to answer this question reliably.",
            "citations": [],
            "status": "insufficient_evidence",
        }

    prompt = build_evidence_prompt(question, chunks)
    answer = _ollama_generate(prompt)
    citations = [
        {
            "source_name": chunk.get("source_name", "unknown"),
            "page_number": int(chunk.get("page_number", 1)),
            "distance": float(chunk.get("distance", 0.0)),
        }
        for chunk in chunks
    ]
    return {"answer": answer, "citations": citations, "status": "answer_ready"}
