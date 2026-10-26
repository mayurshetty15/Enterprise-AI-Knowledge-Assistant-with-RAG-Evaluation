from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CorpusPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page: int = Field(gt=0)
    text: str = Field(min_length=1)


class CorpusDocument(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    filename: str = Field(min_length=1)
    pages: list[CorpusPage] = Field(min_length=1)

    @model_validator(mode="after")
    def filename_is_basename(self) -> "CorpusDocument":
        if Path(self.filename).name != self.filename:
            raise ValueError("Corpus filenames must be simple filenames, not paths.")
        if len({page.page for page in self.pages}) != len(self.pages):
            raise ValueError("Corpus page numbers must be unique within a document.")
        return self


class ExpectedSource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(min_length=1)
    page: int = Field(gt=0)


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    reference_answer: str | None = None
    expected_sources: list[ExpectedSource] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    category: str = Field(min_length=1)
    answerable: bool

    @model_validator(mode="after")
    def answerability_matches_references(self) -> "EvaluationCase":
        if self.answerable and (not self.reference_answer or not self.expected_sources):
            raise ValueError(
                "Answerable cases require a reference_answer and expected_sources."
            )
        if not self.answerable and self.expected_sources:
            raise ValueError("Unanswerable cases must not declare expected_sources.")
        if not self.answerable and self.reference_answer is not None:
            raise ValueError("Unanswerable cases must not declare a reference_answer.")
        if len(self.relevant_chunk_ids) != len(set(self.relevant_chunk_ids)):
            raise ValueError("relevant_chunk_ids must not contain duplicates.")
        return self


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    corpus: list[CorpusDocument] = Field(min_length=1)
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def references_resolve_to_corpus(self) -> "EvaluationDataset":
        document_pages = {
            document.document_id: {page.page for page in document.pages}
            for document in self.corpus
        }
        if len(document_pages) != len(self.corpus):
            raise ValueError("Corpus document IDs must be unique.")
        filenames = [document.filename for document in self.corpus]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Corpus filenames must be unique for citation attribution.")
        case_ids = [case.case_id for case in self.cases]
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("Case IDs must be unique.")
        for case in self.cases:
            for source in case.expected_sources:
                if source.document_id not in document_pages:
                    raise ValueError(
                        f"Case {case.case_id!r} references an unknown document."
                    )
                if source.page not in document_pages[source.document_id]:
                    raise ValueError(
                        f"Case {case.case_id!r} references an unknown source page."
                    )
        return self
