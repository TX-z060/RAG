import json
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from openai import APITimeoutError
import httpx

import evidence_gate
import hybrid_retrieval
import main
from main import app


client = TestClient(app)


def test_rrf_fusion_matches_evaluation_case():
    dense = [
        {"id": 10, "text": "d10", "score": 0.889},
        {"id": 8, "text": "d8", "score": 0.882},
        {"id": 2, "text": "d2", "score": 0.878},
    ]
    tfidf = [
        {"id": 1, "text": "t1", "score": 0.375},
        {"id": 2, "text": "t2", "score": 0.268},
        {"id": 10, "text": "t10", "score": 0.155},
    ]

    results = hybrid_retrieval.reciprocal_rank_fusion(dense, tfidf)

    assert [item["id"] for item in results] == [10, 2, 1]
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["text"] == "d10"


def test_rrf_fusion_tie_break_uses_dense_rank():
    dense = [
        {"id": 5, "text": "d5", "score": 0.9},
        {"id": 2, "text": "d2", "score": 0.8},
    ]
    tfidf = [
        {"id": 2, "text": "t2", "score": 0.5},
        {"id": 7, "text": "t7", "score": 0.4},
    ]

    results = hybrid_retrieval.reciprocal_rank_fusion(dense, tfidf)

    assert [item["id"] for item in results] == [2, 5, 7]


def test_rrf_retrieval_stops_when_tfidf_below_gate():
    chunks = [
        {"id": 1, "text": "HTTP是一种应用层协议。"},
        {"id": 2, "text": "MySQL事务使用COMMIT提交。"},
    ]

    with (
        patch(
            "hybrid_retrieval.tfidf_candidates",
            return_value=[{"id": 1, "text": chunks[0]["text"], "score": 0.05}],
        ),
        patch("hybrid_retrieval.build_dense_index") as fake_build,
    ):
        results = hybrid_retrieval.retrieve_rrf("任意问题", chunks)

    assert results == []
    fake_build.assert_not_called()


def test_rrf_branch_in_ask_filters_and_saves():
    chunk = {"id": 3, "text": "虚拟环境隔离依赖。", "score": 0.02}
    chunk_with_tfidf = dict(chunk, score=0.4)

    with (
        patch.dict(
            os.environ,
            {
                "RAG_RETRIEVER": "rrf",
                "RAG_RRF_K": "60",
                "RAG_RRF_TFIDF_GATE": "0.1",
            },
        ),
        patch(
            "main.get_materials_chunks",
            return_value={"count": 1, "chunks": [chunk]},
        ),
        patch(
            "hybrid_retrieval.build_dense_index",
            return_value=(object(), object()),
        ),
        patch("hybrid_retrieval.retrieve_dense", return_value=[chunk]),
        patch(
            "hybrid_retrieval.tfidf_candidates",
            return_value=[chunk_with_tfidf],
        ),
        patch("main.model") as fake_model,
        patch("main.save_qa_history", return_value=42) as fake_save,
    ):
        fake_model.invoke.return_value.content = "使用虚拟环境。[片段3]"

        response = client.post(
            "/ask", json={"question": "怎样隔离依赖？"}
        )

    assert response.status_code == 200
    assert response.json()["history_id"] == 42
    assert response.json()["retrieved_chunks"][0]["id"] == 3
    fake_save.assert_called_once()


def test_invalid_rrf_config_returns_500():
    with (
        patch.dict(os.environ, {"RAG_RETRIEVER": "rrf", "RAG_RRF_K": "abc"}),
        patch(
            "main.get_materials_chunks",
            return_value={"count": 1, "chunks": [{"id": 1, "text": "资料"}]},
        ),
    ):
        response = client.post("/ask", json={"question": "问题"})

    assert response.status_code == 500
    assert response.json()["detail"] == "RRF 参数必须是数字"


def test_gate_insufficient_returns_fixed_answer_without_model_call():
    chunk = {"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}

    with (
        patch.dict(os.environ, {"RAG_EVIDENCE_GATE": "true"}),
        patch("main.retrieve_for_question", return_value={"results": [chunk]}),
        patch("main.check_evidence", return_value=(False, {"reason": "缺少要点"})),
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        response = client.post("/ask", json={"question": "HTTP是什么？"})

    assert response.status_code == 200
    assert response.json()["answer"] == "资料中未找到相关信息。"
    assert response.json()["history_id"] is None
    assert response.json()["retrieved_chunks"] == []
    fake_model.invoke.assert_not_called()
    fake_save.assert_not_called()


def test_gate_sufficient_continues_to_model():
    chunk = {"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}

    with (
        patch.dict(os.environ, {"RAG_EVIDENCE_GATE": "true"}),
        patch("main.retrieve_for_question", return_value={"results": [chunk]}),
        patch("main.check_evidence", return_value=(True, {"evidence_ids": [1]})),
        patch("main.model") as fake_model,
        patch("main.save_qa_history", return_value=7),
    ):
        fake_model.invoke.return_value.content = "HTTP是应用层协议。[片段1]"
        response = client.post("/ask", json={"question": "HTTP是什么？"})

    assert response.status_code == 200
    assert response.json()["answer"] == "HTTP是应用层协议。[片段1]"
    fake_model.invoke.assert_called_once()


def test_gate_defaults_to_off_for_backward_compatibility():
    chunk = {"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}

    with (
        patch.dict(os.environ, {}, clear=False),
        patch("main.retrieve_for_question", return_value={"results": [chunk]}),
        patch("main.check_evidence") as fake_gate,
        patch("main.model") as fake_model,
        patch("main.save_qa_history", return_value=7),
    ):
        os.environ.pop("RAG_EVIDENCE_GATE", None)
        fake_model.invoke.return_value.content = "HTTP是应用层协议。[片段1]"
        response = client.post("/ask", json={"question": "HTTP是什么？"})

    assert response.status_code == 200
    fake_gate.assert_not_called()


def test_parse_gate_response_handles_fenced_json():
    content = '```json\n{"sufficient": true, "evidence_ids": [1, 2], "reason": "充分"}\n```'

    detail = evidence_gate.parse_gate_response(content)

    assert detail["sufficient"] is True
    assert detail["evidence_ids"] == [1, 2]


@pytest.mark.parametrize("content", [
    "不是 JSON",
    '{"sufficient": "yes"}',
    "{}",
    "",
])
def test_parse_gate_response_returns_none_on_bad_output(content):
    assert evidence_gate.parse_gate_response(content) is None


def test_gate_fail_open_on_model_error():
    chunk = {"id": 1, "text": "资料", "score": 0.9}
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")

    with patch("main.model") as fake_model:
        fake_model.invoke.side_effect = APITimeoutError(request=request)
        sufficient, detail = evidence_gate.check_evidence(
            "问题", [chunk], fake_model
        )

    assert sufficient is True
    assert detail["gate"] == "unavailable"
