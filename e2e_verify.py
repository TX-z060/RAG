"""生产链路 E2E 验证：对比 E5 / RRF / RRF+门禁在 holdout 集上的表现。

每组配置启动独立的 Uvicorn 进程，通过环境变量注入检索参数，
走真实的上传 -> 检索 -> DeepSeek 生成 -> MySQL 持久化链路。

用法：
    python e2e_verify.py                      # 跑全部三组
    python e2e_verify.py --arms dense rrf_gate  # 只跑指定组
    python e2e_verify.py --base-port 8200     # 自定义起始端口

注意：会临时替换 course.txt，脚本结束（含异常）后自动恢复原文件。
"""

import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from urllib.parse import urljoin

import requests


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "evals" / "butterfly_retrieval_holdout.json"
MATERIAL_PATH = BASE_DIR / "evals" / "manual" / "butterfly_beginner_guide.txt"
COURSE_PATH = BASE_DIR / "course.txt"
REPORT_PATH = BASE_DIR / "evals" / "production_e2e_comparison.json"

PYTHON = BASE_DIR / ".venv" / "Scripts" / "python.exe"

ARMS = {
    "dense": {
        "RAG_RETRIEVER": "dense",
        "RAG_DENSE_MIN_SCORE": "0.88",
        "RAG_EVIDENCE_GATE": "false",
    },
    "rrf": {
        "RAG_RETRIEVER": "rrf",
        "RAG_RRF_K": "60",
        "RAG_RRF_TFIDF_GATE": "0.1",
        "RAG_EVIDENCE_GATE": "false",
    },
    "rrf_gate": {
        "RAG_RETRIEVER": "rrf",
        "RAG_RRF_K": "60",
        "RAG_RRF_TFIDF_GATE": "0.1",
        "RAG_EVIDENCE_GATE": "true",
    },
}

TIMEOUT_SECONDS = 180
STARTUP_TIMEOUT = 40
REQUEST_TIMEOUT = 120


def assert_port_free(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        try:
            probe.connect(("127.0.0.1", port))
        except OSError:
            return
    raise SystemExit(
        f"端口 {port} 已被占用。请用 --base-port 指定空闲端口后重试。"
    )


def wait_for_server(base_url, timeout=STARTUP_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(
                urljoin(base_url, "/"), timeout=2,
            )
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


def start_server(port, env_overrides):
    environment = os.environ.copy()
    environment.update(env_overrides)
    environment["PYTHONUNBUFFERED"] = "1"

    log_path = BASE_DIR / f".e2e_server_{port}.log"
    log_file = log_path.open("w", encoding="utf-8")

    process = subprocess.Popen(
        [
            str(PYTHON), "-m", "uvicorn", "main:app",
            "--host", "127.0.0.1", "--port", str(port),
            "--log-level", "warning",
        ],
        cwd=str(BASE_DIR),
        env=environment,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    return process, log_file, log_path


def stop_server(process, log_file, log_path):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
    log_file.close()
    if log_path.exists():
        log_path.unlink()


def run_arm(arm_name, env_overrides, port, dataset, material):
    base_url = f"http://127.0.0.1:{port}/"
    print(f"\n=== 启动 {arm_name} (端口 {port}) ===", flush=True)

    process, log_file, log_path = start_server(port, env_overrides)
    try:
        if not wait_for_server(base_url):
            raise RuntimeError(
                f"{arm_name} 服务启动超时，日志：{log_path.read_text(encoding='utf-8')[-2000:]}"
            )

        with MATERIAL_PATH.open("rb") as handle:
            upload = requests.post(
                urljoin(base_url, "/materials/upload"),
                files={"file": ("guide.txt", handle, "text/plain")},
                timeout=REQUEST_TIMEOUT,
            )
        upload.raise_for_status()
        print("资料上传：", upload.json(), flush=True)

        cases = []
        for case in dataset["cases"]:
            expected = set(case["expected_chunk_ids"])
            started = time.monotonic()
            response = requests.post(
                urljoin(base_url, "/ask"),
                json={"question": case["question"]},
                timeout=REQUEST_TIMEOUT,
            )
            elapsed_ms = round((time.monotonic() - started) * 1000, 1)
            response.raise_for_status()
            body = response.json()

            retrieved_ids = {
                chunk["id"] for chunk in body["retrieved_chunks"]
            }
            dismissed = (
                body["answer"] == "没有找到相关资料"
                or body["answer"].startswith("资料中未找到相关信息")
            )

            record = {
                "id": case["id"],
                "category": "answerable" if expected else "unanswerable",
                "question": case["question"],
                "expected": sorted(expected),
                "retrieved": sorted(retrieved_ids),
                "answer": body["answer"],
                "history_id": body["history_id"],
                "correct_abstention": dismissed,
                "elapsed_ms": elapsed_ms,
            }

            history_ok = None
            snapshot_equal = None
            if body["history_id"] is not None:
                history = requests.get(
                    urljoin(base_url, f"/history/{body['history_id']}"),
                    timeout=REQUEST_TIMEOUT,
                )
                history_ok = history.status_code == 200
                if history_ok:
                    detail = history.json()
                    snapshot_equal = (
                        detail["answer"] == body["answer"]
                        and {
                            chunk["id"]
                            for chunk in detail["retrieved_chunks"]
                        } == retrieved_ids
                    )
            record["history_ok"] = history_ok
            record["snapshot_equal"] = snapshot_equal

            cases.append(record)
            print(
                f"  {case['id']}: expected={sorted(expected)} "
                f"retrieved={sorted(retrieved_ids)} "
                f"abstain={dismissed} {elapsed_ms}ms",
                flush=True,
            )

        return {"arm": arm_name, "config": env_overrides, "cases": cases}
    finally:
        stop_server(process, log_file, log_path)


def summarize(result):
    cases = result["cases"]
    answerable = [case for case in cases if case["category"] == "answerable"]
    unanswerable = [case for case in cases if case["category"] == "unanswerable"]

    recall_sum = 0.0
    mrr_sum = 0.0
    complete_evidence = 0
    for case in answerable:
        expected = set(case["expected"])
        retrieved = set(case["retrieved"])
        recall_sum += len(expected & retrieved) / len(expected)
        ranks = sorted(retrieved)
        for rank, chunk_id in enumerate(ranks, start=1):
            if chunk_id in expected:
                mrr_sum += 1.0 / rank
                break
        if expected <= retrieved:
            complete_evidence += 1

    return {
        "arm": result["arm"],
        "evidence_recall": (
            recall_sum / len(answerable) if answerable else None
        ),
        "mrr_at_3": mrr_sum / len(answerable) if answerable else None,
        "complete_evidence_count": complete_evidence,
        "answerable_count": len(answerable),
        "correct_abstentions": sum(
            1 for case in unanswerable if case["correct_abstention"]
        ),
        "unanswerable_count": len(unanswerable),
        "history_verified": sum(
            1 for case in cases if case["history_ok"] and case["snapshot_equal"]
        ),
        "history_expected": sum(
            1 for case in cases if case["history_id"] is not None
        ),
        "median_latency_ms": sorted(case["elapsed_ms"] for case in cases)[
            len(cases) // 2
        ] if cases else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arms", nargs="+", choices=list(ARMS), default=list(ARMS),
    )
    parser.add_argument("--base-port", type=int, default=8123)
    parser.add_argument(
        "--output", type=Path, default=REPORT_PATH,
    )
    args = parser.parse_args()

    if not PYTHON.is_file():
        raise SystemExit(f"找不到虚拟环境：{PYTHON}")
    if not DATASET_PATH.is_file() or not MATERIAL_PATH.is_file():
        raise SystemExit("缺少评测数据集或资料文件")

    dataset = json.loads(DATASET_PATH.read_text(encoding="utf-8-sig"))
    material = MATERIAL_PATH.read_text(encoding="utf-8")

    original_course = COURSE_PATH.read_bytes() if COURSE_PATH.exists() else None
    results = []

    try:
        for index, arm_name in enumerate(args.arms):
            assert_port_free(args.base_port + index)

        COURSE_PATH.write_text(material, encoding="utf-8")
        for index, arm_name in enumerate(args.arms):
            result = run_arm(
                arm_name,
                ARMS[arm_name],
                args.base_port + index,
                dataset,
                material,
            )
            results.append(result)
    finally:
        if original_course is None:
            COURSE_PATH.unlink(missing_ok=True)
        else:
            COURSE_PATH.write_bytes(original_course)
        print("\ncourse.txt 已恢复原内容。", flush=True)

    report = {
        "scope": "isolated uvicorn processes, real upload / DeepSeek / MySQL",
        "dataset": str(DATASET_PATH.relative_to(BASE_DIR)),
        "material": str(MATERIAL_PATH.relative_to(BASE_DIR)),
        "cases": len(dataset["cases"]),
        "summaries": [summarize(result) for result in results],
        "details": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== 对比汇总 =====")
    header = (
        f"{'配置':<12}{'证据召回':<12}{'MRR@3':<10}"
        f"{'完整证据':<12}{'正确拒答':<12}{'历史校验':<12}{'中位延迟':<12}"
    )
    print(header)
    for summary in report["summaries"]:
        print(
            f"{summary['arm']:<12}"
            f"{summary['evidence_recall']:<12.1%}"
            f"{summary['mrr_at_3']:<10.3f}"
            f"{str(summary['complete_evidence_count']) + '/' + str(summary['answerable_count']):<12}"
            f"{str(summary['correct_abstentions']) + '/' + str(summary['unanswerable_count']):<12}"
            f"{str(summary['history_verified']) + '/' + str(summary['history_expected']):<12}"
            f"{str(summary['median_latency_ms']) + 'ms':<12}"
        )
    print(f"\n报告已保存：{args.output}")


if __name__ == "__main__":
    main()
