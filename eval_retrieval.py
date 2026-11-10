#这段代码的任务是： 用固定的问题测试检索函数，把“应该找到的片段和“实际找到的片段”进行比较，再统计结果”
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from main import retrieve_material

def format_rate(total,count):
    """把计算结果显示为百分比"""
    if count == 0:
        return "不适用"
    return f"{total/count:.1%}"

def main():
    #根据脚本位置找到固定数据集
    dataset_path = (
        Path(__file__).parent
        /"evals"
        /"retrieval_dataset.json"
    )

    data = json.loads(
        dataset_path.read_text(encoding="utf-8-sig")
    )
    chunks = data["chunks"]
    positive_count = 0 #资料中有标准依据的问题数量
    negative_count = 0 #资料中没有标准依据的问题数量
    recall_sum = 0     #各道有答案问题的召回率之和
    precision_sum = 0  #各道有答案问题的精确率之和
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
            results = retrieve_material(
                case["question"]
            )["results"]

            expected_ids = set(
                case["expected_chunk_ids"]
            )
            actual_ids = {
                item["id"] for item in results
            }
            case_results.append({
                "id": case["id"],
                "question": case["question"],
                "expected_chunk_ids": sorted(expected_ids),
                "retrieved_chunk_ids": sorted(actual_ids),
            })

            print(f"\n{case['id']}:{case['question']}")
            print("标准片段：", sorted(expected_ids))
            print("实际片段：", sorted(actual_ids))

            if expected_ids:
                positive_count += 1

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
    report = {
        "algorithm": "tfidf",
        "settings": {
            "analyzer": "char",
            "ngram_range": [1, 3],
            "top_k": 3,
            "neighbor_window": 1,
            "score_filter": "> 0",
        },
        "dataset_sha256": hashlib.sha256(
            dataset_path.read_bytes()
        ).hexdigest(),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "metrics": {
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

    report_path = dataset_path.with_name(
        "tfidf_baseline.json"
    )
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