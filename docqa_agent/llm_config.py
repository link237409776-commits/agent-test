from __future__ import annotations

import os
import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_SYSTEM_PROMPT = (
    "你是一个严谨的文档问答助手，专注于分析给定文档中的内容。"
    "只能基于用户提供的文档证据回答问题，不要编造证据中没有的信息。"
    "不要依赖外部知识或常识推断。若证据不足，请明确回答：‘无法根据当前文档证据确认此问题。’"
    "回答应先给结论，再列依据，并在关键结论后标注来源页码，例如（来源：第 3 页）。"
)

DEFAULT_USER_PROMPT_TEMPLATE = (
    "问题：{question}\n\n"
    "以下是从文档中检索到的证据，请仅基于这些证据回答，不要使用其他信息：\n"
    "{context}"
)


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    api_key: str
    temperature: float
    timeout: float
    system_prompt: str
    user_prompt_template: str

    @property
    def endpoint(self) -> str:
        endpoint = self.base_url.rstrip("/")
        if not endpoint.endswith("/chat/completions"):
            endpoint = f"{endpoint}/chat/completions"
        return endpoint

    @property
    def is_configured(self) -> bool:
        return bool(self.base_url and self.model)

    @property
    def has_api_key(self) -> bool:
        return bool(self.api_key)

    def build_user_prompt(self, question: str, context: str) -> str:
        return self.user_prompt_template.format(question=question, context=context)

    def build_messages(self, question: str, context: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self.build_user_prompt(question, context)},
        ]


def load_llm_config() -> LLMConfig:
    # Try local settings file first
    settings_path = Path(__file__).parent / "llm_settings.json"
    if settings_path.exists():
        try:
            raw = json.loads(settings_path.read_text(encoding="utf-8"))
            return LLMConfig(
                base_url=raw.get("base_url", ""),
                model=raw.get("model", ""),
                api_key=raw.get("api_key", ""),
                temperature=float(raw.get("temperature", 0.1)),
                timeout=float(raw.get("timeout", 60)),
                system_prompt=raw.get("system_prompt", os.getenv("DOCQA_LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)),
                user_prompt_template=raw.get("user_prompt_template", os.getenv("DOCQA_LLM_USER_PROMPT", DEFAULT_USER_PROMPT_TEMPLATE)),
            )
        except Exception:
            # fall back to environment
            pass

    return LLMConfig(
        base_url=os.getenv("DOCQA_LLM_BASE_URL", ""),
        model=os.getenv("DOCQA_LLM_MODEL", ""),
        api_key=os.getenv("DOCQA_LLM_API_KEY", ""),
        temperature=float(os.getenv("DOCQA_LLM_TEMPERATURE", "0.1")),
        timeout=float(os.getenv("DOCQA_LLM_TIMEOUT", "60")),
        system_prompt=os.getenv("DOCQA_LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT),
        user_prompt_template=os.getenv("DOCQA_LLM_USER_PROMPT", DEFAULT_USER_PROMPT_TEMPLATE),
    )


def save_llm_settings(data: dict) -> None:
    """Save LLM settings to docqa_agent/llm_settings.json."""
    settings_path = Path(__file__).parent / "llm_settings.json"
    payload = {
        "base_url": data.get("base_url", ""),
        "model": data.get("model", ""),
        "api_key": data.get("api_key", ""),
        "temperature": float(data.get("temperature", 0.1)),
        "timeout": float(data.get("timeout", 60)),
        "system_prompt": data.get("system_prompt", os.getenv("DOCQA_LLM_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)),
        "user_prompt_template": data.get("user_prompt_template", os.getenv("DOCQA_LLM_USER_PROMPT", DEFAULT_USER_PROMPT_TEMPLATE)),
    }
    settings_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
