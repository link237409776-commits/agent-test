from docqa_agent.answerer import answer_with_evidence
from docqa_agent.chunker import build_chunks
from docqa_agent.models import Evidence, PageContent
from docqa_agent.vector_store import TfidfStore


def test_clause_and_table_chunks_are_created():
    pages = [
        PageContent(
            page=1,
            text="第一条 付款条件\n甲方应在验收后10日内付款。\n第二条 违约责任\n逾期每日按千分之一支付违约金。",
            tables=[[["项目", "金额"], ["服务费", "10000元"]]],
        )
    ]

    chunks = build_chunks(pages)

    assert any(chunk.clause_id == "第一条" for chunk in chunks)
    assert any(chunk.kind == "table" and "服务费" in chunk.text for chunk in chunks)


def test_retrieval_returns_page_evidence():
    chunks = build_chunks([PageContent(page=2, text="第三条 保密义务\n双方应对商业秘密承担保密义务。")])
    store = TfidfStore(chunks)

    results = store.search("保密义务是什么")

    assert results
    assert results[0].page == 2
    assert "保密" in results[0].text


def test_answer_refuses_without_evidence():
    answer = answer_with_evidence("董事会成员是谁？", [])

    assert answer.self_check["needs_refusal"] is True
    assert "无法根据当前文档证据回答" in answer.answer


def test_answer_includes_source_in_text():
    evidence = Evidence(chunk_id="p1-t1", page=1, text="第一条 付款条件。甲方应在验收后10日内付款。", score=0.2, clause_id="第一条")

    answer = answer_with_evidence("付款条件是什么？", [evidence])

    assert "第 1 页" in answer.answer
    assert "第一条" in answer.answer
    assert answer.self_check["grounded"] is True
