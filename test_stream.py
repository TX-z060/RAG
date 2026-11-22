"""流式问答 /ask/stream 的接口测试。"""
import os
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient
from openai import APITimeoutError

import main
from main import app

client = TestClient(app)


class FakeChunk:
    def __init__(self, text):
        self.content = text


def parse_events(response):
    """把 SSE 响应解析为 (event, payload) 列表。"""
    events = []
    for frame in response.text.split("\n\n"):
        name, payload = "", ""
        for line in frame.splitlines():
            if line.startswith("event:"):
                name = line[6:].strip()
            elif line.startswith("data:"):
                payload += line[5:].strip()
        if name and payload:
            events.append((name, __import__("json").loads(payload)))
    return events


def test_stream_emits_meta_then_delta_then_done():
    chunk = {"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}
    with (
        patch("main.retrieve_for_question", return_value={"results": [chunk]}),
        patch("main.save_qa_history", return_value=11) as fake_save,
        patch("main.model") as fake_model,
    ):
        fake_model.stream.return_value = iter([
            FakeChunk("HTTP是"), FakeChunk("应用层"), FakeChunk("协议。[片段1]")
        ])
        response = client.post("/ask/stream", json={"question": "HTTP是什么？"})

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = parse_events(response)
    names = [name for name, _ in events]
    assert names == ["meta", "delta", "delta", "delta", "done"]
    assert events[0][1]["chunks"] == [chunk]
    answer = "".join(payload["text"] for name, payload in events if name == "delta")
    assert answer == "HTTP是应用层协议。[片段1]"
    assert events[-1][1]["history_id"] == 11
    fake_save.assert_called_once_with(
        question="HTTP是什么？",
        answer=answer,
        source="course.txt",
        retrieved_chunks=[chunk],
    )


def test_stream_no_material_shortcuts_to_answer_without_model():
    with (
        patch("main.retrieve_for_question", return_value={"results": []}),
        patch("main.save_qa_history") as fake_save,
        patch("main.model") as fake_model,
    ):
        response = client.post("/ask/stream", json={"question": "资料里没有的问题"})

    events = parse_events(response)
    assert events[0][1]["early"] is True
    answer = "".join(payload["text"] for name, payload in events if name == "delta")
    assert answer == "没有找到相关资料"
    assert events[-1][0] == "done"
    fake_model.stream.assert_not_called()
    fake_save.assert_not_called()


def test_stream_gate_insufficient_does_not_call_model():
    chunk = {"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}
    with (
        patch.dict(os.environ, {"RAG_EVIDENCE_GATE": "true"}),
        patch("main.retrieve_for_question", return_value={"results": [chunk]}),
        patch("main.check_evidence", return_value=(False, {"reason": "缺少要点"})),
        patch("main.save_qa_history") as fake_save,
        patch("main.model") as fake_model,
    ):
        response = client.post("/ask/stream", json={"question": "HTTP是什么？"})

    events = parse_events(response)
    answer = "".join(payload["text"] for name, payload in events if name == "delta")
    assert answer == "资料中未找到相关信息。"
    fake_model.stream.assert_not_called()
    fake_save.assert_not_called()


def test_stream_model_timeout_emits_error_without_saving():
    chunk = {"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}
    request = httpx.Request("POST", "https://api.deepseek.com/chat/completions")
    with (
        patch("main.retrieve_for_question", return_value={"results": [chunk]}),
        patch("main.save_qa_history") as fake_save,
        patch("main.model") as fake_model,
    ):
        fake_model.stream.side_effect = APITimeoutError(request=request)
        response = client.post("/ask/stream", json={"question": "HTTP是什么？"})

    events = parse_events(response)
    assert [name for name, _ in events] == ["meta", "error"]
    assert events[-1][1]["message"] == "模型响应超时，请稍后重试。"
    fake_save.assert_not_called()


def test_stream_db_failure_emits_error_after_deltas():
    chunk = {"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}
    with (
        patch("main.retrieve_for_question", return_value={"results": [chunk]}),
        patch("main.save_qa_history", side_effect=main.MySQLError("db down")),
        patch("main.model") as fake_model,
    ):
        fake_model.stream.return_value = iter([FakeChunk("HTTP是协议。[片段1]")])
        response = client.post("/ask/stream", json={"question": "HTTP是什么？"})

    events = parse_events(response)
    assert [name for name, _ in events] == ["meta", "delta", "error"]
    assert "保存记录失败" in events[-1][1]["message"]


def test_stream_blank_question_rejected_before_stream():
    with patch("main.model") as fake_model:
        response = client.post("/ask/stream", json={"question": "   "})
    assert response.status_code == 400
    assert response.json()["detail"] == "问题不能为空"
    fake_model.stream.assert_not_called()


def test_stream_uses_configured_retriever():
    chunk = {"id": 3, "text": "虚拟环境隔离依赖。", "score": 0.02}
    with (
        patch.dict(os.environ, {"RAG_RETRIEVER": "rrf"}),
        patch("main.retrieve_for_question", return_value={"results": [chunk]}) as fake_retrieve,
        patch("main.save_qa_history", return_value=5),
        patch("main.model") as fake_model,
    ):
        fake_model.stream.return_value = iter([FakeChunk("回答。[片段3]")])
        response = client.post("/ask/stream", json={"question": "怎样隔离依赖？"})

    assert response.status_code == 200
    fake_retrieve.assert_called_once_with("怎样隔离依赖？")
