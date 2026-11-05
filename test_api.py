from unittest.mock import patch
from fastapi.testclient import TestClient
from main import app
from pymysql import OperationalError
import httpx
from openai import APITimeoutError,APIConnectionError
client = TestClient(app)


def test_blank_question_is_rejected():
    with (
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        response = client.post("/ask", json={"question": "   "})

        assert response.status_code == 400
        assert response.json()["detail"] == "问题不能为空"
        fake_model.invoke.assert_not_called()
        fake_save.assert_not_called()

def test_missing_question_is_rejected():
    with (
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        response = client.post("/ask", json={})

        assert response.status_code == 422
        fake_model.invoke.assert_not_called()
        fake_save.assert_not_called()

def test_question_uses_retrieved_material():
    question = "HTTP是什么？"
    chunk = {
        "id": 1,
        "text": "HTTP是一种应用层协议。",
        "score": 0.9,
    }
    expected_answer = "HTTP是一种应用层协议。[片段1]"

    with (
        patch("main.retrieve_material") as fake_retrieve,
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        fake_retrieve.return_value = {"results": [chunk]}
        fake_model.invoke.return_value.content = expected_answer
        fake_save.return_value = 17

        response = client.post("/ask", json={"question": question})

        assert response.status_code == 200

        body = response.json()
        assert body["answer"] == expected_answer
        assert body["retrieved_chunks"] == [chunk]
        assert body["history_id"] == 17

        fake_save.assert_called_once_with(
            question=question,
            answer=expected_answer,
            source="course.txt",
            retrieved_chunks=[chunk],
        )

        fake_retrieve.assert_called_once_with(question)
        fake_model.invoke.assert_called_once()

        messages = fake_model.invoke.call_args.args[0]
        assert chunk["text"] in str(messages)


def test_no_matching_material_does_not_save_history():
    with (
        patch("main.retrieve_material") as fake_retrieve,
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        fake_retrieve.return_value = {"results": []}

        response = client.post("/ask", json={"question": "资料中没有的问题"})

        assert response.status_code == 200
        assert response.json()["history_id"] is None
        assert response.json()["retrieved_chunks"] == []
        fake_model.invoke.assert_not_called()
        fake_save.assert_not_called()


def test_database_failure_returns_service_unavailable():
    chunk = {"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}
    with (
        patch("main.retrieve_material") as fake_retrieve,
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        fake_retrieve.return_value = {"results": [chunk]}
        fake_model.invoke.return_value.content = "HTTP是一种应用层协议。[片段1]"
        fake_save.side_effect = OperationalError(2003, "测试用：数据库暂不可用")

        response = client.post("/ask", json={"question": "HTTP是什么？"})

        assert response.status_code == 503
        assert response.json()["detail"] == "回答已生成，但保存记录失败，请检查数据库连接。"
        fake_save.assert_called_once()


def test_model_timeout_does_not_save_history():
    chunk = {
        "id": 1,
        "text": "HTTP是一种应用层协议。",
        "score": 0.9,
    }

    with (
        patch("main.retrieve_material") as fake_retrieve,
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        fake_retrieve.return_value = {
            "results": [chunk],
        }

        request = httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions",
        )

        fake_model.invoke.side_effect = APITimeoutError(
            request=request,
        )

        response = client.post(
            "/ask",
            json={"question": "HTTP是什么？"},
        )

        assert response.status_code == 504
        assert response.json()["detail"] == (
            "模型响应超时，请稍后重试。"
        )

        fake_model.invoke.assert_called_once()
        fake_save.assert_not_called()


def test_model_connection_failure_does_not_save_history():
    chunk = {
        "id": 1,
        "text": "HTTP是一种应用层协议。",
        "score": 0.9,
    }

    with (
        patch("main.retrieve_material") as fake_retrieve,
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        fake_retrieve.return_value = {"results": [chunk]}

        request = httpx.Request(
            "POST",
            "https://api.deepseek.com/chat/completions",
        )

        fake_model.invoke.side_effect = APIConnectionError(
            request=request,
        )

        response = client.post(
            "/ask",
            json={"question": "HTTP是什么？"},
        )

        assert response.status_code == 502
        assert response.json()["detail"] == (
            "无法连接模型服务，请检查网络。"
        )

        fake_model.invoke.assert_called_once()
        fake_save.assert_not_called()

def test_too_long_question_is_rejected():
    with (
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        response = client.post(
            "/ask",
            json={"question": "问" * 1001},
        )

        assert response.status_code == 422
        fake_model.invoke.assert_not_called()
        fake_save.assert_not_called()

def test_oversized_upload_is_rejected():
    large_content = b"a" * (1024 * 1024 + 1)

    with patch("main.Path.write_text") as fake_write:
        response = client.post(
            "/materials/upload",
            files={
                "file": (
                    "large.txt",
                    large_content,
                    "text/plain",
                ),
            },
        )

        assert response.status_code == 413
        assert response.json()["detail"] == (
            "文件大小不能超过1 MiB。"
        )
        fake_write.assert_not_called()

def test_retrieval_includes_neighbors_without_duplicates():
    chunks = [
        {"id": 1, "text": "先进行热身。"},
        {"id": 2, "text": "力量训练A：椅子深蹲。"},
        {"id": 3, "text": "力量训练A：墙面俯卧撑、臀桥。"},
        {"id": 4, "text": "死虫式：左右交替伸展手脚。"},
        {"id": 5, "text": "合理饮食，规律睡眠。"},
    ]

    with patch(
        "main.get_materials_chunks",
        return_value={
            "count": len(chunks),
            "chunks": chunks,
        },
    ):
        response = client.get(
            "/retrieve",
            params={"question": "力量训练A"},
        )

    assert response.status_code == 200

    results = response.json()["results"]
    ids = [item["id"] for item in results]

    assert ids == [1, 2, 3, 4]
    assert any("死虫式" in item["text"] for item in results)