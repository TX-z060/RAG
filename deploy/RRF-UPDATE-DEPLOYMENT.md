# RRF + 证据门禁 + 流式 线上更新部署记录

## 更新原因

线上当前运行 9b86403，配置为 E5 稠密检索 @0.88。本地 E2E 验证发现：该门槛在短调试集上表现完美，但换到真实长文档（20 片段）后证据召回掉到 87.5%，两道无答案题只挡住一道，属于典型的调参集过拟合。

f65b622 引入三项改进，并有真实 E2E 三组对照数据支撑：

- RRF 混合检索（k=60）：证据召回 87.5% → 100%
- 证据门禁：无答案题正确拒答 50% → 100%（代价是中位延迟 +27%）
- SSE 流式输出：回答边生成边显示，引用片段先于回答出现

## 版本对照

| 项    | 当前线上                                       | 目标版本                                       |
| ---- | ------------------------------------------ | ------------------------------------------ |
| 代码版本 | 9b86403                                    | f65b622                                    |
| 发布目录 | /home/zz/releases/course-assistant/9b86403 | /home/zz/releases/course-assistant/f65b622 |
| 检索方式 | dense @0.88                                | rrf（E5 + TF-IDF 排名融合）                      |
| 证据门禁 | 无                                          | 开启（RAG_EVIDENCE_GATE=true）                 |
| 流式输出 | 无                                          | /ask/stream（SSE）                           |
| 测试   | 28 passed                                  | 49 passed                                  |

## 已验证环境（延续旧版）

- 系统：CentOS Stream 9
- Python：3.12
- PyTorch：2.15.0+cpu
- Sentence Transformers：6.3.0
- 模型：intfloat/multilingual-e5-small
- 运行链路：浏览器 → Nginx:80 → FastAPI:127.0.0.1:8000
- Linux MySQL 端口：3306

RRF 依赖 E5 稠密检索，因此新版仍需完整安装 dense 依赖和 E5 模型，不能只装基础依赖。

## 步骤一：准备新发布目录

**先确认仓库可见性。** 若仓库为 Private，HTTPS clone 会要求认证，而 GitHub 已不支持账号密码做 Git 操作，会报：

```
remote: Invalid username or token. Password authentication is not supported for Git operations.
致命错误：'https://github.com/TX-z060/RAG.git' 鉴权失败
```

**推荐解法：把仓库设为 Public**（前提是已确认无密钥和敏感数据）。这样 clone 完全免认证，面试官也能直接看到项目。操作路径：

GitHub 仓库页面 → Settings → 拉到最底部 Danger Zone → Change repository visibility → Make public → 输入仓库名确认。

公开后直接拉取：

```bash
cd /home/zz/releases/course-assistant
git clone https://github.com/TX-z060/RAG.git f65b622
cd f65b622
git checkout f65b622
```

**若必须保持 Private**，用 Personal Access Token 代替密码（不能填账号密码）：

1. GitHub 头像 → Settings → Developer settings → Personal access tokens → Tokens (classic)
2. Generate new token (classic)，勾选 `repo` 权限，设置过期时间
3. 复制生成的 token（只显示一次）
4. 服务器 clone 时 Username 填 GitHub 用户名，Password 处粘贴 token（终端不回显是正常的）

也可配置 SSH key 后改用 `git clone git@github.com:TX-z060/RAG.git f65b622`。

若服务器无法访问 GitHub，可在本地打包后上传（完全绕过认证）：

```bash
# 本地（Windows PowerShell，在项目根目录）
git archive --format=tar f65b622 -o course-assistant.tar
# 上传到服务器后
mkdir -p /home/zz/releases/course-assistant/f65b622
tar -xf course-assistant.tar -C /home/zz/releases/course-assistant/f65b622
```

## 步骤二：虚拟环境与依赖

```bash
cd /home/zz/releases/course-assistant/f65b622
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements-dense.txt
python -m pip check
```

requirements-dense.txt 已包含 requirements.txt，并额外安装 torch CPU 版和 sentence-transformers。

## 步骤三：模型缓存与私密配置

从旧版复制模型缓存，避免重新下载 E5：

```bash
cp -a /home/zz/releases/course-assistant/9b86403/.cache /home/zz/releases/course-assistant/f65b622/.cache
```

复制私密配置（不提交 Git）：

```bash
cp /home/zz/releases/course-assistant/9b86403/.env /home/zz/releases/course-assistant/f65b622/.env
cp /home/zz/releases/course-assistant/9b86403/.mysql/app.env /home/zz/releases/course-assistant/f65b622/.mysql/app.env
```

每个发布目录各自持有 course.txt。把线上正在使用的资料复制过去，保证切换后资料不丢失：

```bash
cp /home/zz/releases/course-assistant/9b86403/course.txt /home/zz/releases/course-assistant/f65b622/course.txt
```

注意：若 9b86403 运行期间通过网页上传过新资料，必须复制它的当前 course.txt，而不是 Git 仓库里的版本。

验证模型可用：

```bash
python check_embeddings.py
python check_dense_cache.py
```

## 步骤四：systemd 覆盖配置

新增覆盖配置 /etc/systemd/system/course-assistant.service.d/rrf-release.conf：

```ini
[Service]
WorkingDirectory=/home/zz/releases/course-assistant/f65b622

Environment=RAG_RETRIEVER=rrf
Environment=RAG_RRF_K=60
Environment=RAG_RRF_TFIDF_GATE=0.1
Environment=RAG_EVIDENCE_GATE=true
Environment=OMP_NUM_THREADS=1
Environment=HF_HOME=/home/zz/releases/course-assistant/f65b622/.cache/huggingface

ExecStart=
ExecStart=/home/zz/releases/course-assistant/f65b622/.venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

关键：新旧两个 override 都设置了 WorkingDirectory 和 ExecStart，不能同时生效。切到新版前必须先禁用旧的：

```bash
sudo mv /etc/systemd/system/course-assistant.service.d/e5-release.conf /etc/systemd/system/course-assistant.service.d/e5-release.conf.disabled
sudo systemctl daemon-reload
```

RAG_EVIDENCE_GATE=true 会为每次问答增加一次 DeepSeek 判定调用，换来无答案题不瞎答。若上线后延迟不可接受，可单独设为 false 只保留 RRF（召回修复仍在），无需回退整个版本。

## 步骤五：SELinux 设置

新目录的 Python 符号链接需要打标签，否则 systemd 可能无法执行：

```bash
sudo semanage fcontext -a -f l -t bin_t '/home/zz/releases/course-assistant/f65b622/\.venv/bin/python(3(\.12)?)?'
sudo restorecon -v /home/zz/releases/course-assistant/f65b622/.venv/bin/python*
```

## 步骤六：8001 端口预览验证

先不改正式 8000 服务，用 8001 独立预览确认新版可用：

```bash
cd /home/zz/releases/course-assistant/f65b622
source .venv/bin/activate
RAG_RETRIEVER=rrf RAG_RRF_K=60 RAG_RRF_TFIDF_GATE=0.1 RAG_EVIDENCE_GATE=true \
HF_HOME=/home/zz/releases/course-assistant/f65b622/.cache/huggingface \
OMP_NUM_THREADS=1 \
nohup .venv/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8001 > preview.log 2>&1 &
```

预览验证项：

```bash
# 接口清单应包含 /ask 和 /ask/stream
curl -s http://127.0.0.1:8001/openapi.json | grep -o '/ask[^\"]*'

# 根接口
curl -s http://127.0.0.1:8001/

# 流式接口：观察 event 顺序 meta → delta → done
curl -N -s -X POST http://127.0.0.1:8001/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"这里换成资料里的一个问题"}'

# 本地全量测试（需装 dense 依赖）
.venv/bin/python -m pytest test_api.py test_history.py test_dense_api.py test_dense_upload.py test_rrf.py test_stream.py -q
```

预期：49 passed；流式响应先返回 meta（引用片段），再逐块返回 delta，最后 done（历史只保存一次）。

验证通过后结束预览进程：

```bash
pkill -f 'uvicorn main:app --host 127.0.0.1 --port 8001'
```

## 步骤七：正式切流到 8000

```bash
sudo systemctl daemon-reload
sudo systemctl restart course-assistant
systemctl status course-assistant --no-pager -l
sudo journalctl -u course-assistant -n 30 --no-pager
```

源码没有打印检索日志，是否切到 RRF 以接口行为为准：调用 /ask 后返回的 retrieved_chunks 是 E5 与 TF-IDF 融合后的结果（通常比单路 dense 的 3 个更多），且旧版 @0.88 的长文档漏检现象消失。

Nginx 无需改动：流式响应的 X-Accel-Buffering: no 请求头已在代码中设置，Nginx 不会缓冲 SSE。仍代理到 127.0.0.1:8000。

## 上线后验收

- 通过 Nginx 访问 <http://192.168.65.132/chat，上传资料并提问>
- 观察引用片段是否先于回答出现（流式生效）
- 问一道资料中不存在的问题，确认返回"资料中未找到相关信息"而非编造回答（门禁生效）
- 用本地 e2e_verify.py 的 10 道 holdout 题（8 有答案 + 2 无答案）复跑，确认长文档召回与拒答表现
- 观察 journalctl 中无异常堆栈

## 回退

出问题需要回到 E5 旧版时：

```bash
sudo mv /etc/systemd/system/course-assistant.service.d/rrf-release.conf /etc/systemd/system/course-assistant.service.d/rrf-release.conf.disabled
sudo mv /etc/systemd/system/course-assistant.service.d/e5-release.conf.disabled /etc/systemd/system/course-assistant.service.d/e5-release.conf
sudo systemctl daemon-reload
sudo systemctl restart course-assistant
systemctl status course-assistant --no-pager -l
```

回退后使用 9b86403 目录和 dense @0.88 配置。两个版本共用 Linux 数据库，但各自持有 course.txt。新版运行期间若通过网页更新过资料，回退前需把新版 course.txt 复制回旧目录：

```bash
cp /home/zz/releases/course-assistant/f65b622/course.txt /home/zz/releases/course-assistant/9b86403/course.txt
```

## 已知注意事项

- 门禁增加一次模型调用，中位延迟约 +27%，属于用延迟换正确性
- 门禁判断依赖模型，本地 10 题全对但未做大样本验证
- RRF 的 TF-IDF 门槛 0.1 是评测数据选定的，换资料后若召回异常可优先调这个值
- 新旧版本目录完全独立，代码、配置、资料都不共享，切换时以 course.txt 和 .env 为重点
- 本地 127.0.0.1:8000/chat 只是开发调试入口，与线上无关

## 尚未验证（延续旧版待办）

- 整台虚拟机重启后的恢复
- 并发请求与性能压力测试
- 新版在 Linux 真实长文档上的检索与回答质量（需上线后复跑 E2E 确认）
