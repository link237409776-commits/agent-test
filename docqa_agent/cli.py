from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .agent import DocumentQaAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="最小可运行的智能文档问答 Agent")
    parser.add_argument("pdf", help="待问答的 PDF 文件路径")
    parser.add_argument("-q", "--question", help="直接提问；不传则进入交互模式")
    parser.add_argument("--top-k", type=int, default=5, help="检索证据条数")
    parser.add_argument("--save-index", help="保存可复现索引 JSON")
    args = parser.parse_args()

    agent = DocumentQaAgent(args.pdf)
    profile = agent.build()
    print("PDF 类型判断：")
    print(json.dumps(asdict(profile), ensure_ascii=False, indent=2))

    if args.save_index:
        agent.save_index(Path(args.save_index))
        print(f"索引已保存：{args.save_index}")

    if args.question:
        evidences = agent.retrieve(args.question, top_k=args.top_k)
        print_evidence(evidences)
        print_answer(agent.ask(args.question, top_k=args.top_k))
        return

    print("\n进入问答模式，输入 exit 退出。")
    while True:
        question = input("\n问题> ").strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if question:
            evidences = agent.retrieve(question, top_k=args.top_k)
            print_evidence(evidences)
            print_answer(agent.ask(question, top_k=args.top_k))


def print_evidence(evidences) -> None:
    print("\n检索到的证据：")
    if not evidences:
        print("- 未检索到符合条件的证据。")
        return
    for item in evidences:
        clause = f"，条款：{item.clause_id}" if item.clause_id else ""
        print(f"- {item.chunk_id} | 第 {item.page} 页{clause} | score={item.score}")
        print(f"  {item.text[:180].replace(chr(10), ' ')}")


def print_answer(answer) -> None:
    print("\n答案：")
    print(answer.answer)
    print("\n来源证据：")
    for item in answer.evidences:
        clause = f"，条款：{item.clause_id}" if item.clause_id else ""
        print(f"- {item.chunk_id} | 第 {item.page} 页{clause} | score={item.score}")
        print(f"  {item.text[:180].replace(chr(10), ' ')}")
    print("\n自检：")
    print(json.dumps(answer.self_check, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
