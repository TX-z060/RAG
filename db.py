from pathlib import Path
from dotenv import dotenv_values
import json
import pymysql

config_path = Path(__file__).parent / ".mysql" / "app.env"
config = dotenv_values(config_path)

def get_connection():
    return pymysql.connect(
        host=config["MYSQL_HOST"],
        port=int(config["MYSQL_PORT"]),
        user=config["MYSQL_USER"],
        password=config["MYSQL_PASSWORD"],
        database=config["MYSQL_DATABASE"],
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
        connect_timeout=5,
    )


def save_qa_history(question, answer, source, retrieved_chunks):
    """保存一次问答及其引用片段，提交后返回新记录的编号。"""
    chunks_json = json.dumps(
        retrieved_chunks, ensure_ascii=False, allow_nan=False
    )
    sql="""
        INSERT INTO qa_history 
            (question, answer, source, retrieved_chunks)
        VALUES 
            (%s, %s, %s, %s)
    """

    with get_connection() as connection:
        try:
            with connection.cursor() as cursor:
                cursor.execute(sql, (question, answer, source, chunks_json))
                history_id = cursor.lastrowid

            connection.commit()
            return history_id
        except Exception:
            connection.rollback()
            raise

#查询历史的函数
def list_qa_history(limit=10, offset=0):
    sql = """
        SELECT id, question, source, created_at
        FROM qa_history
        ORDER BY id DESC
        LIMIT %s OFFSET %s
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) AS total FROM qa_history")
            total = cursor.fetchone()["total"]

            cursor.execute(sql, (limit, offset))
            items = list(cursor.fetchall())

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "items": items,
    }

#按编号查询一条问答记录的函数
def get_qa_history(history_id):
    sql = """
        SELECT id, question, answer, source,
               retrieved_chunks, created_at
        FROM qa_history
        WHERE id = %s
    """

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(sql, (history_id,))
            record = cursor.fetchone()

    if record is not None:
        record["retrieved_chunks"] = json.loads(
            record["retrieved_chunks"]
        )

    return record