import os
from pathlib import Path

# 模型缓存放在当前项目的 D 盘目录中。
base_dir = Path(__file__).resolve().parent
cache_dir = base_dir / ".cache" / "huggingface"
cache_dir.mkdir(parents=True, exist_ok=True)

# 在导入模型库之前设置缓存位置。
os.environ["HF_HOME"] = str(cache_dir)

from sentence_transformers import SentenceTransformer


def main():
    print("正在加载模型，首次运行需要下载……", flush=True)

    model = SentenceTransformer(
        "intfloat/multilingual-e5-small",
        device="cpu",
        cache_folder=str(cache_dir),
    )

    question = "怎样避免两个Python项目的依赖版本互相影响？"

    passages = [
        "Python虚拟环境用于隔离不同项目的依赖。",
        "Nginx把浏览器请求转发给内部的后端服务。",
        "MySQL事务可以使用ROLLBACK撤销尚未提交的修改。",
    ]

    texts = ["query: " + question]

    for passage in passages:
        texts.append("passage: " + passage)

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    print("向量矩阵形状：", embeddings.shape)

    question_vector = embeddings[0]
    passage_vectors = embeddings[1:]

    scores = passage_vectors @ question_vector

    ranked_indices = scores.argsort()[::-1]

    print("\n问题：", question)
    print("\n按相似度从高到低排列：")

    for index in ranked_indices:
        print(f"\n片段 {index + 1}，分数：{scores[index]:.4f}")
        print(passages[index])


if __name__ == "__main__":
    main()