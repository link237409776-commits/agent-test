from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from .llm_config import load_llm_config
from .models import Answer, Evidence


def filter_useful_evidence(evidences: list[Evidence], min_score: float = 0.05) -> list[Evidence]:
    return [item for item in evidences if item.score >= min_score]


def build_prompt_messages(question: str, evidences: list[Evidence]) -> list[dict[str, str]]:
    config = load_llm_config()
    return config.build_messages(question, build_evidence_context(evidences))


def answer_with_evidence(question: str, evidences: list[Evidence], min_score: float = 0.05) -> Answer:
    useful = filter_useful_evidence(evidences, min_score)
    if not useful:
        text = "无法根据当前文档证据回答该问题。"
        return Answer(
            question,
            text,
            evidences,
            self_check(question, text, evidences, refused=True, used_llm=False, llm_status="no_evidence"),
        )

    llm_answer, llm_status = generate_with_llm(question, useful)
    if llm_answer:
        return Answer(
            question,
            llm_answer,
            useful,
            self_check(question, llm_answer, useful, refused=False, used_llm=True, llm_status=llm_status),
        )

    fallback = build_extract_answer(question, useful)
    return Answer(
        question,
        fallback,
        useful,
        self_check(question, fallback, useful, refused=False, used_llm=False, llm_status=llm_status),
    )


def generate_with_llm(question: str, evidences: list[Evidence]) -> tuple[str | None, str]:
    config = load_llm_config()
    if not config.is_configured:
        return None, "disabled: set DOCQA_LLM_BASE_URL and DOCQA_LLM_MODEL"

    endpoint = config.endpoint
    prompt_messages = config.build_messages(question, build_evidence_context(evidences[:8]))
    payload = {
        "model": config.model,
        "temperature": config.temperature,
        "messages": prompt_messages,
    }
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if config.has_api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    try:
        request = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(request, timeout=config.timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        return None, f"http_error: {exc.code} {detail}"
    except (OSError, urllib.error.URLError) as exc:
        return None, f"request_error: {exc}"
    except json.JSONDecodeError as exc:
        return None, f"bad_json: {exc}"

    content = result.get("choices", [{}])[0].get("message", {}).get("content")
    if not isinstance(content, str) or not content.strip():
        return None, "empty_response"
    return content.strip(), "ok"


def build_evidence_context(evidences: list[Evidence]) -> str:
    lines: list[str] = []
    for index, item in enumerate(evidences, start=1):
        source_label = f"[证据 {index}] 页码: {item.page}; 条款: {item.clause_id or '无'}; 类型: {item.kind}"
        lines.append(f"{source_label}; 检索分数: {item.score}\n{item.text}")
    return "\n\n".join(lines)


def build_extract_answer(question: str, evidences: list[Evidence]) -> str:
    bullets = []
    for item in evidences[:5]:
        snippet = best_snippet(question, item.text)
        label = f"第 {item.page} 页"
        if item.clause_id:
            label += f"，{item.clause_id}"
        bullets.append(f"- {snippet}（来源：{label}，分数：{item.score}）")
    return "根据检索到的文档证据：\n" + "\n".join(bullets)


def best_snippet(question: str, text: str, max_chars: int = 260) -> str:
    sentences = re.split(r"(?<=[。！？；;])\s*|\n+", text)
    query_terms = set(re.findall(r"[\u4e00-\u9fff]{1,4}|[A-Za-z0-9_]+", question.lower()))
    ranked = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        terms = set(re.findall(r"[\u4e00-\u9fff]{1,4}|[A-Za-z0-9_]+", sentence.lower()))
        phrase_hit = 1 if compact(question) and compact(question) in compact(sentence) else 0
        ranked.append((len(query_terms.intersection(terms)) + phrase_hit * 5, sentence))
    ranked.sort(key=lambda item: item[0], reverse=True)
    snippet = ranked[0][1] if ranked else text.strip()
    if len(snippet) > max_chars:
        return snippet[: max_chars - 1] + "..."
    return snippet


def compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def self_check(
    question: str,
    answer: str,
    evidences: list[Evidence],
    refused: bool,
    used_llm: bool,
    llm_status: str,
) -> dict[str, object]:
    has_evidence = bool(evidences)
    max_score = max((item.score for item in evidences), default=0.0)
    grounded = has_evidence and max_score >= 0.05 and not refused
    needs_refusal = refused or not grounded
    hallucination_risk = "low" if grounded and len(evidences) >= 2 else "medium" if grounded else "high"

    return {
        "has_evidence": has_evidence,
        "max_retrieval_score": round(max_score, 4),
        "grounded": grounded,
        "hallucination_risk": hallucination_risk,
        "needs_refusal": needs_refusal,
        "used_llm": used_llm,
        "llm_status": llm_status,
        "policy": "仅基于返回证据作答；证据不足时拒答或提示需要人工复核。",
    }
