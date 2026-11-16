import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from main import app


client = TestClient(app)


def test_dense_question_filters_and_saves_evidence():
    question = "怎样隔离Python项目的依赖？"

    relevant = {
        "id": 1,
        "text": "Python虚拟环境用于隔离不同项目的依赖。",
        "score": 0.92,
    }
    irrelevant = {
        "id": 2,
        "text": "MySQL事务使用COMMIT提交。",
        "score": 0.86,
    }

    answer = "使用Python虚拟环境隔离依赖。[片段1]"

    with (
        patch.dict(
            os.environ,
            {
                "RAG_RETRIEVER": "dense",
                "RAG_DENSE_MIN_SCORE": "0.88",
            },
        ),
        patch(
            "main.get_materials_chunks",
            return_value={
                "count": 2,
                "chunks": [relevant, irrelevant],
            },
        ),
        patch(
            "dense_retrieval.build_dense_index",
            return_value=(object(), object()),
        ),
        patch(
            "dense_retrieval.retrieve_dense",
            return_value=[relevant, irrelevant],
        ) as fake_dense,
        patch("main.retrieve_material") as fake_tfidf,
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        fake_model.invoke.return_value.content = answer
        fake_save.return_value = 25

        response = client.post(
            "/ask",
            json={"question": question},
        )

        assert response.status_code == 200

        body = response.json()
        assert body["answer"] == answer
        assert body["history_id"] == 25
        assert body["retrieved_chunks"] == [relevant]

        fake_dense.assert_called_once()
        fake_tfidf.assert_not_called()
        fake_model.invoke.assert_called_once()

        messages = fake_model.invoke.call_args.args[0]
        assert relevant["text"] in str(messages)
        assert irrelevant["text"] not in str(messages)

        fake_save.assert_called_once_with(
            question=question,
            answer=answer,
            source="course.txt",
            retrieved_chunks=[relevant],
        )


@pytest.mark.parametrize(
    "scores",
    [
        [],
        [0.87],
        [0.88],
    ],
)
def test_dense_without_qualified_evidence_stops(scores):
    candidates = [
        {
            "id": index,
            "text": "不合格的候选片段",
            "score": score,
        }
        for index, score in enumerate(scores, start=1)
    ]

    with (
        patch.dict(
            os.environ,
            {
                "RAG_RETRIEVER": "dense",
                "RAG_DENSE_MIN_SCORE": "0.88",
            },
        ),
        patch(
            "main.get_materials_chunks",
            return_value={
                "count": 1,
                "chunks": [{"id": 1, "text": "测试资料"}],
            },
        ),
        patch(
            "dense_retrieval.build_dense_index",
            return_value=(object(), object()),
        ),
        patch(
            "dense_retrieval.retrieve_dense",
            return_value=candidates,
        ),
        patch("main.retrieve_material") as fake_tfidf,
        patch("main.model") as fake_model,
        patch("main.save_qa_history") as fake_save,
    ):
        response = client.post(
            "/ask",
            json={"question": "资料没有覆盖的问题"},
        )

        assert response.status_code == 200

        body = response.json()
        assert body["answer"] == "没有找到相关资料"
        assert body["history_id"] is None
        assert body["retrieved_chunks"] == []

        fake_tfidf.assert_not_called()
        fake_model.invoke.assert_not_called()
        fake_save.assert_not_called()