"""手动验证POST /ask到MySQL的保存流程，会新增一条标记的测试记录。

只替换模型回复；使用实际资料检索、接口处理、保存函数和数据库。
"""

import json
from pathlib import Path
import sys
from unittest.mock import patch

from fastapi.testclient import TestClient

from db import get_connection
from main import app


def main():
    material = Path(__file__).with_name("course.txt").read_text(encoding="utf-8")
    first_line = next((line.strip() for line in material.splitlines() if line.strip()), "")
    if not first_line:
        raise RuntimeError("请先上传一份非空资料，再运行接口保存验证。")
    question = "【接口保存验证】" + first_line[:30]
    expected_answer = "【接口保存验证】这是一条验证接口写入数据库的测试回复。"

    with patch("main.model") as fake_model:
        fake_model.invoke.return_value.content = expected_answer
        response = TestClient(app).post("/ask", json={"question": question})
        assert response.status_code == 200, response.text
        body = response.json()
        assert isinstance(body["history_id"], int)
        assert body["retrieved_chunks"]
        fake_model.invoke.assert_called_once()

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT question, answer, source, retrieved_chunks "
                "FROM qa_history WHERE id = %s",
                (body["history_id"],),
            )
            stored = cursor.fetchone()

    assert stored is not None
    assert stored["question"] == body["question"]
    assert stored["answer"] == expected_answer
    assert stored["source"] == body["source"]
    assert json.loads(stored["retrieved_chunks"]) == body["retrieved_chunks"]
    print(f"接口保存及独立连接读回验证通过，记录编号：{body['history_id']}")
    print(f"已核对引用片段数量：{len(body['retrieved_chunks'])}")
    print("测试记录已保留，模型回复使用测试替身。")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
