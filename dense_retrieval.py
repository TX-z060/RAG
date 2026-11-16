import os
from pathlib import Path

base_dir = Path(__file__).resolve().parent
cache_dir = base_dir / ".cache" / "huggingface"
os.environ["HF_HOME"] = str(cache_dir)

from sentence_transformers import SentenceTransformer


MODEL_NAME = "intfloat/multilingual-e5-small"


def build_dense_index(chunks):
    """加载模型，并提前计算所有资料片段的向量。"""
    model = SentenceTransformer(
        MODEL_NAME,
        device="cpu",
        cache_folder=str(cache_dir),
        local_files_only=True,
    )

    texts = [
        "passage: " + chunk["text"]
        for chunk in chunks
    ]

    passage_vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return model, passage_vectors


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