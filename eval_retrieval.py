#这段代码的任务是： 用固定的问题测试检索函数，把“应该找到的片段和“实际找到的片段”进行比较，再统计结果”
import argparse
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from dense_retrieval import build_dense_index, retrieve_dense
from main import retrieve_material

def format_rate(total,count):
    """把计算结果显示为百分比"""
    if count == 0:
        return "不适用"
    return f"{total/count:.1%}"
def reciprocal_rank_at_3(results, expected_ids):
    """返回前三个结果中，第一个正确片段的排名倒数。"""
    ranked_results = sorted(
        results,
        key=lambda item: item["score"],
        reverse=True,
    )

    for rank, item in enumerate(
        ranked_results[:3],
        start=1,
    ):
        if item["id"] in expected_ids:
            return 1.0 / rank

    return 0.0
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--no-neighbors",
        action="store_true",
        help="评测时关闭相邻片段补充",
    )

    parser.add_argument(
        "--min-score",
        type=float,
        default=0.0,
        help="只保留分数大于此值的候选片段",
    )
    parser.add_argument(
        "--dataset",
        default="evals/retrieval_dataset.json",
        help="评测数据文件的路径",
    )
    parser.add_argument(
        "--retriever",
        choices=["tfidf", "dense"],
        default="tfidf",
        help="选择检索方法",
    )
    parser.add_argument(
        "--dense-min-score",
        type=float,
        default=None,
        help="向量检索的分数门槛，不指定时不过滤",
    )

    args = parser.parse_args()

    if not 0.0 <= args.min_score <= 1.0:
        parser.error("--min-score 必须在 0 到 1 之间")

    if args.dense_min_score is not None:
        if args.retriever != "dense":
            parser.error("--dense-min-score 只能用于 dense 检索")
        if not -1.0 <= args.dense_min_score <= 1.0:
            parser.error("--dense-min-score 必须在 -1 到 1 之间")

    include_neighbors = not args.no_neighbors
    if args.retriever == "dense":
        if include_neighbors or args.min_score != 0.0:
            parser.error(
                "当前向量基线请使用 --no-neighbors，"
                "并且不要设置非零 --min-score"
            )

    dataset_path = (
        Path(__file__).parent
        / args.dataset
    )

    data = json.loads(
        dataset_path.read_text(encoding="utf-8-sig")
    )
    chunks = data["chunks"]
    model = None
    passage_vectors = None
    if args.retriever == "dense":
        print("正在加载模型并计算资料向量……", flush=True)
        model, passage_vectors = build_dense_index(chunks)

    positive_count = 0 #资料中有标准依据的问题数量
    negative_count = 0 #资料中没有标准依据的问题数量
    recall_sum = 0     #各道有答案问题的召回率之和
    precision_sum = 0  #各道有答案问题的精确率之和
    reciprocal_rank_sum = 0.0
    correct_empty_count = 0 #无答案问题中，检索确实返回空结果的数量
    case_results = []

    #临时让检索函数读取固定片段
    #检索算法仍然使用项目中的真实实现
    with patch(
        "main.get_materials_chunks",
        #return_value 的意思是：临时替代函数被调用时，返回指定的数据。
        return_value={ 
            "count": len(chunks),
            "chunks": chunks,
        },
    ):
        for case in data["cases"]:
            if args.retriever == "dense":
                results = retrieve_dense(
                    case["question"],
                    chunks,
                    model,
                    passage_vectors,
                )
                # 先取前三个候选，再按向量分数门槛过滤。
                if args.dense_min_score is not None:
                    results = [
                        item
                        for item in results
                        if item["score"] > args.dense_min_score
                    ]
            else:
                results = retrieve_material(
                    case["question"],
                    include_neighbors=include_neighbors,
                )["results"]

                results = [
                    item
                    for item in results
                    if item["score"] > args.min_score
                ]

            expected_ids = set(
                case["expected_chunk_ids"]
            )
            actual_ids = {
                item["id"] for item in results
            }
            rr = reciprocal_rank_at_3(
                results,
                expected_ids,
            )

            case_results.append({
                "id": case["id"],
                "question": case["question"],
                "expected_chunk_ids": sorted(expected_ids),
                "reciprocal_rank_at_3": (
                    rr if expected_ids else None
                ),
                "retrieved_chunk_ids": sorted(actual_ids),
                "ranked_results": [
                    {
                        "id": item["id"],
                        "score": float(item["score"]),
                    }
                    for item in sorted(
                        results,
                        key=lambda item: item["score"],
                        reverse=True,
                    )
                ],
            })

            print(f"\n{case['id']}:{case['question']}")
            print("标准片段：", sorted(expected_ids))
            print("实际片段：", sorted(actual_ids))
            ranked_results = sorted(
                results,
                key=lambda item: item["score"],
                reverse=True,
            )

            print("候选片段分数：")

            for item in ranked_results:
                label = (
                    "标准答案片段"
                    if item["id"] in expected_ids
                    else "非标准答案片段"
                )

                print(
                    f"  片段 {item['id']}："
                    f"{item['score']:.4f}，{label}"
                )

            if expected_ids:
                positive_count += 1
                reciprocal_rank_sum += rr
                print(f"本题 RR@3：{rr:.4f}")

                #交集表示实际找到的正确片段
                hit_count = len(expected_ids & actual_ids)

                recall = hit_count / len(expected_ids)
                precision = (
                    hit_count / len(actual_ids)
                    if actual_ids else 0.0
                )
                recall_sum += recall
                precision_sum += precision

                print(
                    f"证据召回率：{recall:.1%},"
                    f"上下文精确率：{precision:.1%}"
                )
            else:
                negative_count += 1

                if not actual_ids:
                    correct_empty_count += 1
                    print("结果：正确返回空结果")
                else:
                    print("结果：资料未覆盖，但仍召回了片段")

    print("\n===== 评测汇总 =====")
    print("有答案问题数：", positive_count)
    print("无答案问题数：", negative_count)
    print(
        "平均证据召回率：",
        format_rate(recall_sum, positive_count),
    )
    print(
        "平均上下文精确率：",
        format_rate(precision_sum, positive_count),
    )
    print(
        "无答案问题空检索率：",
        format_rate(correct_empty_count, negative_count),
    )
    if args.retriever == "dense":
        settings = {
            "model": "intfloat/multilingual-e5-small",
            "normalize_embeddings": True,
            "query_prefix": "query: ",
            "passage_prefix": "passage: ",
            "top_k": 3,
            "neighbor_window": 0,
            "score_filter": (
                f"> {args.dense_min_score:g}"
                if args.dense_min_score is not None
                else None
            ),
        }
    else:
        settings = {
            "analyzer": "char",
            "ngram_range": [1, 3],
            "top_k": 3,
            "neighbor_window": 1 if include_neighbors else 0,
            "score_filter": f"> {args.min_score:g}",
        }
    mean_rr = (
        reciprocal_rank_sum / positive_count
        if positive_count else None
    )

    print(
        "有答案问题 MRR@3：",
        f"{mean_rr:.4f}"
        if mean_rr is not None
        else "不适用",
    )
    report = {
        "algorithm": args.retriever,
        "settings": settings,
        "dataset_sha256": hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest(),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "metrics": {
            "answerable_mrr_at_3": mean_rr,
            "mean_evidence_recall": (
                recall_sum / positive_count
                if positive_count else None
            ),
            "mean_context_precision": (
                precision_sum / positive_count
                if positive_count else None
            ),
            "unanswerable_empty_rate": (
                correct_empty_count / negative_count
                if negative_count else None
            ),
        },
        "cases": case_results,
    }

    neighbor_mode = (
        "with_neighbors"
        if include_neighbors
        else "no_neighbors"
    )
    if args.retriever == "dense":
        filter_suffix = (
            "top3"
            if args.dense_min_score is None
            else f"top3_min_score_{args.dense_min_score:g}"
        )
        report_name = (
            f"{dataset_path.stem}"
            f"_dense_no_neighbors_{filter_suffix}.json"
        )
    else:
        report_name = (
            f"{dataset_path.stem}"
            f"_tfidf_{neighbor_mode}"
            f"_min_score_{args.min_score:g}.json"
        )
    report_path = dataset_path.with_name(report_name)
    report_path.write_text(
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n报告已保存：", report_path)

if __name__ == "__main__":
    main()
