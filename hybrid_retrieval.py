"""RRF 混合检索：融合 E5 稠密检索与 TF-IDF 检索的排序结果。

配置来源：evals/python_venv_retrieval_holdout_comparison.json 中的
rrf_tfidf_gate_010 方案，以及 butterfly_retrieval_holdout_rrf_top3.json。
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dense_retrieval import build_dense_index, retrieve_dense


DEFAULT_RRF_K = 60
DEFAULT_TFIDF_GATE = 0.1
DEFAULT_TOP_K = 3


def tfidf_candidates(question, chunks, top_k=DEFAULT_TOP_K):
    """返回 TF-IDF 正分数候选项，参数与 main.retrieve_material 保持一致。"""
    texts = [chunk["text"] for chunk in chunks]
    if not texts or not question:
        return []

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
    )
    matrix = vectorizer.fit_transform(texts)
    query = vectorizer.transform([question])
    scores = cosine_similarity(query, matrix).flatten()

    indices = [
        int(index)
        for index in scores.argsort()[::-1][:top_k]
        if scores[index] > 0
    ]
    return [
        {
            "id": chunks[index]["id"],
            "text": chunks[index]["text"],
            "score": float(scores[index]),
        }
        for index in indices
    ]


def reciprocal_rank_fusion(
    dense_results,
    tfidf_results,
    k=DEFAULT_RRF_K,
    top_k=DEFAULT_TOP_K,
):
    """按 RRF 分数融合两路结果。

    每个片段的分数为两路排名倒数之和：1/(k + 名次)，名次从 1 开始。
    只在单路出现的片段按该路名次计分。
    并列时依次按 E5 名次、片段编号排序，保证输出稳定。
    """
    dense_rank = {
        item["id"]: rank
        for rank, item in enumerate(dense_results, start=1)
    }
    tfidf_rank = {
        item["id"]: rank
        for rank, item in enumerate(tfidf_results, start=1)
    }
    by_id = {}
    for item in list(dense_results) + list(tfidf_results):
        by_id.setdefault(item["id"], item)

    def fusion_score(chunk_id):
        score = 0.0
        if chunk_id in dense_rank:
            score += 1.0 / (k + dense_rank[chunk_id])
        if chunk_id in tfidf_rank:
            score += 1.0 / (k + tfidf_rank[chunk_id])
        return score

    def sort_key(chunk_id):
        return (
            -fusion_score(chunk_id),
            dense_rank.get(chunk_id, float("inf")),
            chunk_id,
        )

    ordered_ids = sorted(by_id, key=sort_key)[:top_k]
    results = []
    for chunk_id in ordered_ids:
        item = by_id[chunk_id]
        results.append({
            "id": chunk_id,
            "text": item["text"],
            "score": fusion_score(chunk_id),
        })
    return results


def retrieve_rrf(question, chunks, k=DEFAULT_RRF_K,
                 tfidf_gate=DEFAULT_TFIDF_GATE,
                 top_k=DEFAULT_TOP_K):
    """RRF 检索入口：TF-IDF 最佳分低于门槛时返回空，抑制无答案误召回。"""
    tfidf_results = tfidf_candidates(question, chunks, top_k=top_k)
    if not tfidf_results or tfidf_results[0]["score"] < tfidf_gate:
        return []

    model, passage_vectors = build_dense_index(chunks)
    dense_results = retrieve_dense(
        question, chunks, model, passage_vectors,
    )
    return reciprocal_rank_fusion(
        dense_results, tfidf_results, k=k, top_k=top_k,
    )
