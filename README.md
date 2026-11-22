\# 课程资料问答助手



一个基于 FastAPI、DeepSeek 和 MySQL 的资料问答项目。

用户上传 TXT 文件后，可以根据资料提问、查看检索片段和问答历史。



\## 主要功能



\- 上传 UTF-8 编码的 TXT 文件，最大 1 MiB。

\- 上传成功后替换当前资料。

\- 提问长度限制为 1000 个字符。

\- 使用 TF-IDF 检索相关片段，并补充相邻片段。

\- 调用 DeepSeek，根据检索资料生成带引用的回答。

\- 展示可以展开查看的原文片段。

\- 将问答及片段快照保存到 MySQL。

\- 分页查看历史记录，查看历史回答和资料片段。

\- 处理空白问题、超长问题、模型超时和数据库异常。



\## 技术栈



\- Python、FastAPI、Uvicorn

\- LangChain、DeepSeek

\- scikit-learn：TF-IDF 与余弦相似度

\- MySQL、PyMySQL

\- HTML、JavaScript

\- pytest



\## 主要文件



\- main.py：接口、上传、检索和模型调用。

\- db.py：数据库连接、保存和查询函数。

\- dense\_retrieval.py：E5 向量检索与资料向量缓存。

\- hybrid\_retrieval.py：RRF 混合检索，融合 E5 与 TF-IDF 排名。

\- evidence\_gate.py：生成前的证据充分性门禁。

\- index.html：网页界面。

\- course.txt：当前上传的资料。

\- requirements.txt：Python 依赖及版本。

\- test\_api.py：问答和检索等接口测试。

\- test\_history.py：历史查询接口测试。

\- database/schema.sql：数据库建表语句。

\- database/mysql\_local.py：本地 MySQL 实例管理脚本。

\- database/README.md：数据库配置说明。



\## 本地配置



当前运行环境为 Windows，项目代码和数据库数据保存在 D 盘。



DeepSeek 密钥保存在项目根目录的 .env 文件中：



```text

DEEPSEEK\_API\_KEY=填写自己的密钥

```



数据库连接信息保存在 .mysql/app.env 中，由 db.py 读取。

本地数据库使用 127.0.0.1:3307，数据库名为 course\_assistant。



.env、.mysql 和 .venv 已加入 Git 忽略规则。

密钥与数据库密码需要在本地配置。



\## 启动项目



以下步骤用于已经完成初始化的本地环境。



先进入项目目录：



```powershell

cd 'D:\\电子实训\\git练习\\课程资料问答助手'

```



启动项目数据库：



```powershell

.\\.venv\\Scripts\\python.exe database\\mysql\_local.py start

```



检查数据库状态：



```powershell

.\\.venv\\Scripts\\python.exe database\\mysql\_local.py status

```



启动后端：



```powershell

.\\.venv\\Scripts\\python.exe -m uvicorn main:app --reload

```



浏览器访问：



\- 问答页面：http://127.0.0.1:8000/chat

\- 接口文档：http://127.0.0.1:8000/docs



项目数据库没有配置开机自动启动，电脑重启后需要重新启动。

首次搭建时，还需要安装 Python 依赖并完成 MySQL 初始化。

\## 安装依赖

基础依赖（接口、TF-IDF 检索、测试）：

```powershell

.\\.venv\\Scripts\\python.exe -m pip install -r requirements.txt

```

E5 稠密检索依赖（可选；运行完整测试套件需安装）：

```powershell

.\\.venv\\Scripts\\python.exe -m pip install -r requirements-dense.txt

```



\## 使用流程



1\. 打开问答页面。

2\. 选择并上传 TXT 资料。

3\. 输入与资料有关的问题。

4\. 查看回答，展开片段核对依据。

5\. 刷新历史列表，选择页码或查看记录详情。



历史详情读取保存时的片段快照。

即使后来上传了新资料，旧记录仍保留当时的片段。



\## 运行测试



```powershell

.\\.venv\\Scripts\\python.exe -m pytest test\_api.py test\_history.py test\_dense\_api.py test\_dense\_upload.py test\_rrf.py -q

```



接口测试使用模型和数据库替身。

检索回归测试使用固定资料，执行真实的检索逻辑。



\## 检索方式与门禁配置



问答检索方式由环境变量 RAG\_RETRIEVER 决定，默认 tfidf，可选 dense 或 rrf。



\- tfidf：字符级 TF-IDF，补充相邻片段。

\- dense：E5 向量检索，门槛由 RAG\_DENSE\_MIN\_SCORE 控制，默认 0.88。

\- rrf：融合 E5 与 TF-IDF 的排名，k 由 RAG\_RRF\_K 控制，默认 60；TF-IDF 最佳分门槛由 RAG\_RRF\_TFIDF\_GATE 控制，默认 0.1。低于门槛时返回空结果，避免无答案误召回。



RAG\_EVIDENCE\_GATE 设为 true 时启用证据门禁：generate 前先核对片段是否覆盖问题要点，不足则返回固定拒答，不调用模型也不写入历史。门禁调用失败或输出无法解析时自动放行。默认关闭，保持原有行为。



\## 生产链路 E2E 验证



e2e\_verify.py 用独立进程启动三组服务，走真实的上传、检索、DeepSeek 和 MySQL 链路，在蝶泳 holdout 集上对比 dense、rrf 和 rrf\_gate：



```powershell

.\\.venv\\Scripts\\python.exe e2e\_verify.py

```



每组使用独立端口，通过环境变量注入配置。脚本会临时替换 course.txt，结束时自动恢复。结果保存到 evals/production\_e2e\_comparison.json，包含证据召回率、MRR@3、正确拒答数和历史校验。三组对照中，rrf 将证据召回率从 87.5% 提升到 100%，rrf\_gate 进一步把无答案题的正确拒答从 1/2 提升到 2/2。门禁拒答不写入历史，因此该组历史校验的可验证条数为 8。



\## 当前局限



当前使用 TF-IDF 检索，复杂语义问题和全文概述仍有改进空间。

补充相邻片段可以减少跨块遗漏，但会增加上下文长度。

引用编号由模型生成，仍需要结合原文检查准确性。

