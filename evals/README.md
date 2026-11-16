# 检索评测记录

## 实验环境

- 向量模型：intfloat/multilingual-e5-small
- PyTorch：2.15.0+cpu
- Sentence Transformers：6.3.0
- 使用 CPU 运行，模型缓存位于项目的 .cache/huggingface。
- 向量检索使用 query: 和 passage: 前缀，并归一化向量。
- 先取相似度最高的三个片段，再进行分数过滤。

## 数据集

- retrieval_dataset.json：8 道调试题，用于观察结果和选择门槛。
- retrieval_validation.json：12 道验证题，包含单片段、
  多片段和无答案问题。
- 两份数据集使用相同的 12 个资料片段。

## 调试集结果

| 方法 | 门槛 | 召回率 | 精确率 | 无答案空检索率 | MRR@3 |
|---|---|---|---|---|---|
| TF-IDF | 0 | 100% | 33.3% | 0% | 0.9167 |
| E5 | 不过滤 | 100% | 33.3% | 0% | 1.0000 |
| E5 | 0.85 | 100% | 47.2% | 100% | 1.0000 |
| E5 | 0.88 | 100% | 100% | 100% | 1.0000 |
| E5 | 0.91 | 33.3% | 33.3% | 100% | 0.3333 |

## 验证集结果

固定调试集选出的门槛后进行验证：

| 方法 | 门槛 | 召回率 | 精确率 | 无答案空检索率 |
|---|---|---|---|---|
| TF-IDF | 0.2 | 100% | 100% | 50% |
| E5 | 0.88 | 100% | 89.6% | 100% |

E5 在验证集上的有答案问题 MRR@3 为 1.0000。

## 结论与限制

- E5 在调试集上改善了正确证据的排名。
- 在验证集上，E5 减少了无答案问题的误召回，
  但有答案问题的上下文精确率低于 TF-IDF。
- v7 找全了参数校验和状态码证据，但混入 Nginx 片段。
- 暂定 E5 门槛为 0.88，尚未接入网页问答接口。
- 精确率和 MRR 只统计有答案问题。
- 评测数据量较小，尚未验证真实课程长文档和最终回答质量。
- 若继续根据验证题调参，需要另外准备新题验证。

## 复现实验

调试集：

python eval_retrieval.py --retriever dense --no-neighbors --dense-min-score 0.88

验证集：

python eval_retrieval.py --dataset evals/retrieval_validation.json --retriever dense --no-neighbors --dense-min-score 0.88