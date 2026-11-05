from datetime import datetime
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from pymysql import OperationalError

from main import app


client = TestClient(app)


def test_history_list_supports_pagination():
    page = {
        "total": 3,
        "limit": 2,
        "offset": 1,
        "items": [
            {"id": 2, "question": "HTTP是什么？", "source": "course.txt",
             "created_at": datetime(2026, 11, 3, 10, 0, 0)},
        ],
    }
    with patch("main.list_qa_history", return_value=page) as fake_list:
        response = client.get("/history", params={"limit": 2, "offset": 1})

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 3
    assert body["limit"] == 2
    assert body["offset"] == 1
    assert body["items"][0]["id"] == 2
    assert body["items"][0]["created_at"] == "2026-11-03T10:00:00"
    fake_list.assert_called_once_with(limit=2, offset=1)


def test_empty_history_uses_default_pagination():
    page = {"total": 0, "limit": 10, "offset": 0, "items": []}
    with patch("main.list_qa_history", return_value=page) as fake_list:
        response = client.get("/history")

    assert response.status_code == 200
    assert response.json() == page
    fake_list.assert_called_once_with(limit=10, offset=0)


@pytest.mark.parametrize("url", [
    "/history?limit=0", "/history?limit=101", "/history?offset=-1",
    "/history?limit=abc", "/history/0", "/history/abc",
    "/history/18446744073709551616",
])
def test_invalid_history_parameters_do_not_query_database(url):
    with patch("main.list_qa_history") as fake_list, patch("main.get_qa_history") as fake_get:
        response = client.get(url)

    assert response.status_code == 422
    fake_list.assert_not_called()
    fake_get.assert_not_called()


def test_history_detail_returns_answer_and_chunks():
    record = {
        "id": 17, "question": "HTTP是什么？", "answer": "应用层协议。[片段1]",
        "source": "course.txt",
        "retrieved_chunks": [{"id": 1, "text": "HTTP是一种应用层协议。", "score": 0.9}],
        "created_at": datetime(2026, 11, 3, 10, 0, 0),
    }
    with patch("main.get_qa_history", return_value=record) as fake_get:
        response = client.get("/history/17")

    assert response.status_code == 200
    assert response.json()["answer"] == record["answer"]
    assert response.json()["retrieved_chunks"] == record["retrieved_chunks"]
    fake_get.assert_called_once_with(17)


def test_missing_history_returns_not_found():
    with patch("main.get_qa_history", return_value=None):
        response = client.get("/history/17")

    assert response.status_code == 404
    assert response.json()["detail"] == "问答记录不存在"


@pytest.mark.parametrize("url, target", [
    ("/history", "main.list_qa_history"),
    ("/history/17", "main.get_qa_history"),
])
def test_history_database_failure_returns_service_unavailable(url, target):
    with patch(target, side_effect=OperationalError(2003, "测试用：数据库暂不可用")):
        response = client.get(url)

    assert response.status_code == 503
    assert response.json()["detail"] == "查询记录失败，请检查数据库连接。"
