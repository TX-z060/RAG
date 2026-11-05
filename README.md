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

.\\.venv\\Scripts\\python.exe -m pytest test\_api.py test\_history.py -q

```



接口测试使用模型和数据库替身。

检索回归测试使用固定资料，执行真实的检索逻辑。



\## 当前局限



当前使用 TF-IDF 检索，复杂语义问题和全文概述仍有改进空间。

补充相邻片段可以减少跨块遗漏，但会增加上下文长度。

引用编号由模型生成，仍需要结合原文检查准确性。

