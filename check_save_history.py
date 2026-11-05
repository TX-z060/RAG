"""手动验证保存函数。每次运行会新增一条明确标记的测试记录。"""

import json
import sys

from db import get_connection, save_qa_history


def main():
    question = "【保存函数验证】Python's 参数能正确保存吗？"
    answer = "这是一条用于检查中文、单引号和JSON引用片段的测试记录。"
    source = "保存函数验证"
    chunks = [{"id": 1, "text": "参数化SQL将语句与数据分开传递。", "score": 0.9}]

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT DATABASE() AS database_name, @@port AS port")
            target = cursor.fetchone()
    assert target["database_name"] == "course_assistant"
    assert target["port"] == 3307

    history_id = save_qa_history(question, answer, source, chunks)

    # 使用新的连接读回，验证数据已经提交，而不只是停留在原连接中。
    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT id, question, answer, source, retrieved_chunks "
                "FROM qa_history WHERE id = %s",
                (history_id,),
            )
            stored = cursor.fetchone()

    assert stored is not None
    assert stored["question"] == question
    assert stored["answer"] == answer
    assert stored["source"] == source
    assert json.loads(stored["retrieved_chunks"]) == chunks
    print(f"保存并读回验证通过，测试记录编号：{history_id}")
    print("测试记录已保留在qa_history表中。")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
