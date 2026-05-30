from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import scrolledtext

from .agent import DocumentQaAgent
from .llm_config import load_llm_config


class DocumentQaWindow:
    def __init__(self, pdf_path: str | Path = "doc", top_k: int = 5) -> None:
        self.pdf_path = pdf_path
        self.top_k = top_k
        self.agent: DocumentQaAgent | None = None
        self.messages: queue.Queue[tuple[str, str]] = queue.Queue()

        self.root = tk.Tk()
        self.root.title("Document QA")
        self.root.geometry("820x560")

        self.output = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, state=tk.DISABLED)
        self.output.pack(fill=tk.BOTH, expand=True, padx=12, pady=(12, 8))

        input_frame = tk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=12, pady=(0, 12))

        self.input = tk.Entry(input_frame)
        self.input.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input.bind("<Return>", self._on_send)

        self.send_button = tk.Button(input_frame, text="Send", command=self._on_send, state=tk.DISABLED)
        self.send_button.pack(side=tk.RIGHT, padx=(8, 0))

        self._append("System", "Loading PDF documents. Scanned PDFs may take a while on the first OCR run...")
        self._append("System", self._llm_status_message())
        threading.Thread(target=self._load_agent, daemon=True).start()
        self.root.after(100, self._drain_messages)

    def run(self) -> None:
        self.root.mainloop()

    def _load_agent(self) -> None:
        try:
            agent = DocumentQaAgent(self.pdf_path)
            agent.build()
        except Exception as exc:
            self.messages.put(("System", f"Failed to load PDF documents: {exc}"))
            return

        self.agent = agent
        self.messages.put(("System", f"Loaded {len(agent.pdf_paths)} PDF document(s). You can ask now."))

    def _on_send(self, event: tk.Event | None = None) -> None:
        question = self.input.get().strip()
        if not question or self.agent is None:
            return

        self.input.delete(0, tk.END)
        self.send_button.config(state=tk.DISABLED)
        self._append("You", question)
        threading.Thread(target=self._answer, args=(question,), daemon=True).start()

    def _answer(self, question: str) -> None:
        assert self.agent is not None
        try:
            answer = self.agent.ask(question, top_k=self.top_k)
            text = getattr(answer, "answer", str(answer))
            if hasattr(answer, "self_check"):
                used_llm = answer.self_check.get("used_llm")
                llm_status = answer.self_check.get("llm_status")
                score = answer.self_check.get("max_retrieval_score")
                text += f"\n\n[状态] 大模型: {'已调用' if used_llm else '未调用'}; LLM 状态: {llm_status}; 最高检索分数: {score}"
        except Exception as exc:
            text = f"Failed to answer: {exc}"
        self.messages.put(("Assistant", text))

    def _drain_messages(self) -> None:
        while not self.messages.empty():
            speaker, text = self.messages.get()
            self._append(speaker, text)
            if self.agent is not None:
                self.send_button.config(state=tk.NORMAL)
        self.root.after(100, self._drain_messages)

    def _append(self, speaker: str, text: str) -> None:
        self.output.config(state=tk.NORMAL)
        self.output.insert(tk.END, f"{speaker}: {text}\n\n")
        self.output.see(tk.END)
        self.output.config(state=tk.DISABLED)

    def _llm_status_message(self) -> str:
        config = load_llm_config()
        if not config.is_configured:
            return "LLM is not configured. Set DOCQA_LLM_BASE_URL and DOCQA_LLM_MODEL to enable generation."
        return f"LLM configured: model={config.model}, base_url={config.base_url}"


def main() -> None:
    DocumentQaWindow().run()


if __name__ == "__main__":
    main()
