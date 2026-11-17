# E5 向量检索 Linux 部署记录

## 已验证环境

- 系统：CentOS Stream 9
- Python：3.12
- 运行代码版本：9b86403
- 发布目录：/home/zz/releases/course-assistant/9b86403
- PyTorch：2.15.0+cpu
- Sentence Transformers：6.3.0
- 模型：intfloat/multilingual-e5-small
- 模型缓存：发布目录下的 .cache/huggingface
- Linux MySQL 端口：3306

## 运行链路

浏览器 → Nginx:80 → FastAPI:127.0.0.1:8000
→ E5 检索与过滤 → DeepSeek 回答 → MySQL 保存历史

## 环境准备

在发布目录创建独立虚拟环境，安装依赖：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --no-cache-dir -r requirements-dense.txt
python -m pip check
```

从 Linux 旧版复制 .env 和 .mysql/app.env，
保留在服务器本地，不提交 Git。

模型下载与验证：

```bash
python check_embeddings.py
python check_dense_cache.py
```

## 服务配置

基础配置：

/etc/systemd/system/course-assistant.service

新版覆盖配置：

/etc/systemd/system/course-assistant.service.d/e5-release.conf

覆盖配置指定新版目录、虚拟环境、E5 检索方式和 0.88 门槛。
服务使用单个 Uvicorn 进程，OMP_NUM_THREADS=1。

Nginx 仍代理到 127.0.0.1:8000，无需改变代理端口。

## SELinux 设置

曾为新版 Python 符号链接添加以下持久标签规则：

```bash
sudo semanage fcontext -a -f l -t bin_t '/home/zz/releases/course-assistant/9b86403/\.venv/bin/python(3(\.12)?)?'
sudo restorecon -v /home/zz/releases/course-assistant/9b86403/.venv/bin/python*
```

此记录对应 9b86403 路径，规则已存在时无需重复添加。
以后使用其他发布目录时，需要检查对应路径的标签。

## 已完成验证

- Linux 依赖检查通过。
- 模型编码及缓存验证通过。
- 自动化接口测试：28 passed。
- 8001 端口预览完成。
- systemd 成功启动新版，监听 8000。
- 通过 Nginx 的根接口返回 200。
- 正式入口上传、问答和历史查询返回 200。
- systemd 日志中确认 E5 加载和资料向量计算。

## 查看运行状态

```bash
systemctl status course-assistant --no-pager -l
sudo journalctl -u course-assistant -n 30 --no-pager
```

## 回退旧版程序

停用新版覆盖配置，再重新加载并重启：

```bash
sudo mv /etc/systemd/system/course-assistant.service.d/e5-release.conf /etc/systemd/system/course-assistant.service.d/e5-release.conf.disabled
sudo systemctl daemon-reload
sudo systemctl restart course-assistant
```

原服务配置保留，回退后使用：

/home/zz/projects/course-assistant

两个版本共用 Linux 数据库，但各自有 course.txt。
若新版运行期间更新过资料，回退前应备份并处理当前资料，
程序回退不会自动同步这部分文件。

## 尚未验证

- 整台虚拟机重启后的恢复。
- 并发请求与性能压力测试。
- 真实课程长文档上的检索和回答质量。