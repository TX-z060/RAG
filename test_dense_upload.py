import os
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient

import dense_retrieval
from main import app


client = TestClient(app)


def test_upload_replaces_dense_material(tmp_path):
    python_text = "Python虚拟环境用于隔离不同项目的依赖。"
    mysql_text = "MySQL事务使用ROLLBACK撤销尚未提交的修改。"

    python_question = "怎样隔离Python项目的依赖？"
    mysql_question = "怎样撤销MySQL事务修改？"

    # 用确定的向量模拟编码，避免下载或加载真实模型。
    def encode_texts(texts, **kwargs):
        vectors = []

        for text in texts:
            if "Python" in text:
                vectors.append([1.0, 0.0])
            elif "MySQL" in text:
                vectors.append([0.0, 1.0])
            else:
                raise AssertionError(f"未预期的编码内容：{text}")

        return np.array(vectors, dtype=np.float32)

    fake_encoder = Mock()
    fake_encoder.encode.side_effect = encode_texts

    def upload(text):
        response = client.post(
            "/materials/upload",
            files={
                "file": (
                    "test.txt",
                    text.encode("utf-8"),
                    "text/plain",
                ),
            },
        )
        assert response.status_code == 200

    def ask_question(question):
        response = client.post(
            "/ask",
            json={"question": question},
        )
        assert response.status_code == 200
        return response.json()

    # 避免其他测试留下的缓存影响本次检查。
    dense_retrieval._build_dense_index_cached.cache_clear()

    try:
        with (
            patch.dict(
                os.environ,
                {
                    "RAG_RETRIEVER": "dense",
                    "RAG_DENSE_MIN_SCORE": "0.88",
                },
            ),
            patch(
                "main.Path",
                return_value=tmp_path / "main.py",
            ),
            patch(
                "dense_retrieval.get_dense_model",
                return_value=fake_encoder,
            ),
            patch("main.model") as fake_model,
            patch("main.save_qa_history") as fake_save,
        ):
            fake_model.invoke.return_value.content = (
                "测试回答。[片段1]"
            )
            fake_save.return_value = 30

            # 上传 Python 资料，首次提问应找到它。
            upload(python_text)
            first = ask_question(python_question)
            assert first["retrieved_chunks"][0]["text"] == python_text

            # 内容未变，再次提问应复用资料向量。
            repeated = ask_question(python_question)
            assert repeated["retrieved_chunks"] == first["retrieved_chunks"]

            # 上传 MySQL 资料，旧的 Python 问题应无匹配结果。
            upload(mysql_text)
            missing = ask_question(python_question)

            assert missing["answer"] == "没有找到相关资料"
            assert missing["retrieved_chunks"] == []
            assert missing["history_id"] is None

            # 这次无匹配结果，不应新增模型调用或保存记录。
            assert fake_model.invoke.call_count == 2
            assert fake_save.call_count == 2

            # MySQL 问题应使用新资料。
            updated = ask_question(mysql_question)
            assert updated["retrieved_chunks"][0]["text"] == mysql_text

            saved_chunks = fake_save.call_args.kwargs["retrieved_chunks"]
            assert saved_chunks == updated["retrieved_chunks"]

            # 收集资料编码调用，排除每次问题的编码调用。
            passage_batches = [
                call.args[0]
                for call in fake_encoder.encode.call_args_list
                if call.args[0][0].startswith("passage: ")
            ]

            assert passage_batches == [
                ["passage: " + python_text],
                ["passage: " + mysql_text],
            ]

            assert (tmp_path / "course.txt").read_text(
                encoding="utf-8"
            ) == mysql_text

    finally:
        # 即使断言失败，也清理本次测试留下的缓存。
        dense_retrieval._build_dense_index_cached.cache_clear()