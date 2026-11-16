import os
from functools import lru_cache
from pathlib import Path

base_dir = Path(__file__).resolve().parent
cache_dir = base_dir / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(cache_dir)

from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"


@lru_cache(maxsize=1)
def get_dense_model():
    """缓存已加载的模型实例，供后续调用复用。"""
    print("正在加载 E5 模型……", flush=True)

    return SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(cache_dir),
        local_files_only=True,
    )


@lru_cache(maxsize=1)
def _build_dense_index_cached(chunk_items):
    """按片段编号、文字和顺序缓存最近一份资料向量。"""
    model = get_dense_model()

    print("正在重新计算资料向量……", flush=True)

    texts = [
        "passage: " + text
        for chunk_id, text in chunk_items
    ]

    passage_vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return model, passage_vectors


def build_dense_index(chunks):
    """把当前资料快照转换成缓存键，再取得对应的模型和向量。"""
    chunk_items = tuple(
        (chunk["id"], chunk["text"])
        for chunk in chunks
    )

    return _build_dense_index_cached(chunk_items)


def retrieve_dense(
    question,
    chunks,
    model,
    passage_vectors,
):
    """计算问题向量，返回相似度最高的三个片段。"""
    question_vector = model.encode(
        ["query: " + question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0]

    scores = passage_vectors @ question_vector
    indices = scores.argsort()[::-1][:3]

    results = []

    for index in indices:
        results.append({
            "id": chunks[index]["id"],
            "text": chunks[index]["text"],
            "score": float(scores[index]),
        })

    return results
