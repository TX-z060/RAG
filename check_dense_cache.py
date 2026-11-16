import json
from pathlib import Path

import numpy as np

from dense_retrieval import build_dense_index


def main():
    dataset_path = (
        Path(__file__).resolve().parent
        / "evals"
        / "retrieval_dataset.json"
    )
    data = json.loads(dataset_path.read_text(encoding="utf-8-sig"))
    chunks = data["chunks"]

    # 第一次：计算资料向量。
    model1, vectors1 = build_dense_index(chunks)

    # 第二次：列表和字典都是新对象，但片段内容完全相同。
    copied_chunks = [dict(chunk) for chunk in chunks]
    model2, vectors2 = build_dense_index(copied_chunks)

    assert vectors1 is vectors2, "相同资料没有复用向量"
    print("相同资料复用向量：通过")

    # 第三次：只修改内存副本，原始数据文件保持原样。
    copied_chunks[0]["text"] += " 每个项目单独管理依赖。"
    model3, vectors3 = build_dense_index(copied_chunks)

    assert vectors3 is not vectors2, "资料更新后没有重建向量"
    assert vectors3.shape == vectors2.shape, "资料数量不变时向量形状不应改变"
    assert not np.allclose(vectors3[0], vectors2[0]), "修改文字后向量没有更新"
    print("资料更新后重建向量：通过")

    assert model1 is model2 is model3, "模型被重复加载"
    print("三次调用复用模型：通过")


if __name__ == "__main__":
    main()
