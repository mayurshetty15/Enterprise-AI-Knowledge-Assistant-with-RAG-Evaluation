# Evaluation datasets

`synthetic-v1.json` is a small, versioned, synthetic corpus intended for local
smoke evaluation. Its facts and citations are fabricated for tests; they are not
claims about a real organization.

## Adding real-document cases

1. Create a new dataset version rather than editing a dataset already used by a
   saved run.
2. Only use documents for which you have permission to create evaluation data.
   Human-review each question, reference answer, source document ID, and page
   number against the original PDF before adding it.
3. Add the relevant page text to the local evaluation corpus. Keep document IDs
   stable and non-sensitive; do not use confidential names or secrets in IDs.
4. Use `relevant_chunk_ids` only when the exact stable chunk IDs have been
   independently verified. Leave it empty to evaluate source-document/page
   relevance instead.
5. Mark unanswerable cases with `answerable: false`, `reference_answer: null`,
   and an empty `expected_sources` list.
6. Do not commit private document text or ground truth to a public repository.
   Keep private datasets outside version control and evaluate them locally.

The dataset schema rejects missing required fields, duplicate case/document IDs,
invalid pages, and paths in corpus filenames. Reports intentionally omit source
text, questions, reference answers, and generated answer text.
