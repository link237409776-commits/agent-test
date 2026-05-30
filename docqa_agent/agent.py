from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

from .answerer import answer_with_evidence
from .chunker import build_chunks
from .models import Answer, PdfProfile
from .parser import parse_pdf
from .pdf_analyzer import analyze_pdf
from .vector_store import TfidfStore


class DocumentQaAgent:
    def __init__(self, pdf_path: str | Path | Iterable[str | Path] = "doc") -> None:
        self.pdf_paths = self._resolve_pdf_paths(pdf_path)
        if not self.pdf_paths:
            raise FileNotFoundError(f"No PDF files found in: {pdf_path}")

        self.pdf_path = self.pdf_paths[0]
        self.profile: PdfProfile | None = None
        self.profiles: list[PdfProfile] = []
        self.store: TfidfStore | None = None

    @staticmethod
    def _resolve_pdf_paths(pdf_path: str | Path | Iterable[str | Path]) -> list[Path]:
        if isinstance(pdf_path, (str, Path)):
            path = Path(pdf_path)
            if path.is_dir():
                return sorted(path.glob("*.pdf"))
            return [path]

        paths: list[Path] = []
        for item in pdf_path:
            path = Path(item)
            if path.is_dir():
                paths.extend(sorted(path.glob("*.pdf")))
            else:
                paths.append(path)
        return paths

    def build(self) -> PdfProfile:
        all_chunks = []
        self.profiles = []

        for pdf_path in self.pdf_paths:
            profile = analyze_pdf(pdf_path)
            pages = parse_pdf(pdf_path, profile)
            all_chunks.extend(build_chunks(pages))
            self.profiles.append(profile)

        self.profile = self.profiles[0]
        chunks = all_chunks
        self.store = TfidfStore(chunks)
        return self.profile

    def retrieve(self, question: str, top_k: int = 5) -> list[Evidence]:
        if self.store is None:
            self.build()
        assert self.store is not None
        return self.store.search(question, top_k=top_k)

    def ask(self, question: str, top_k: int = 5) -> Answer:
        evidences = self.retrieve(question, top_k=top_k)
        return answer_with_evidence(question, evidences)

    def save_index(self, path: str | Path) -> None:
        if self.store is None:
            self.build()
        assert self.store is not None
        self.store.save(path)


def chat(pdf_path: str | Path | Iterable[str | Path] = "doc", top_k: int = 5) -> None:
    agent = DocumentQaAgent(pdf_path)
    print("Loading PDF documents...")
    agent.build()
    print(f"Loaded {len(agent.pdf_paths)} PDF document(s). Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break
        if not question:
            continue

        answer = agent.ask(question, top_k=top_k)
        print(f"\nAssistant: {getattr(answer, 'answer', answer)}")
