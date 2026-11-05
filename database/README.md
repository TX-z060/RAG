# 本项目的 D 盘 MySQL

已配置独立的 MySQL 8.0 实例，使用已安装的 MySQL 程序。项目代码、实例配置、数据库数据、日志、临时文件和连接配置均保存在 D 盘。

## 位置与连接

- 项目：`D:\电子实训\git练习\课程资料问答助手`
- 数据目录：项目中的 `.mysql/data`
- 日志：`.mysql/logs`
- 临时目录：`.mysql/tmp`
- 配置：`.mysql/my.ini`
- 地址：`127.0.0.1`
- 端口：`3307`
- 数据库：`course_assistant`
- 表：`qa_history`
- 应用账号：`course_app`，拥有该数据库的查询和插入权限。
- 应用连接配置：`.mysql/app.env`。密码已随机生成，请保留在本地。

现有 `MySQL80` Windows 服务继续使用原来的配置和数据目录。新实例没有注册为开机自动启动的 Windows 服务。

`.mysql/` 已加入项目的 `.gitignore`。不要将其中的数据文件、密码配置和日志提交到 Git。

## 使用命令

先在 PowerShell 进入项目：

```powershell
cd 'D:\电子实训\git练习\课程资料问答助手'
```

启动本项目数据库；已启动时会返回当前状态：

```powershell
.\.venv\Scripts\python.exe database\mysql_local.py start
```

检查实际数据目录、端口、日志位置和问答记录表：

```powershell
.\.venv\Scripts\python.exe database\mysql_local.py status
```

进入本项目的 MySQL 命令行，连接配置由脚本读取：

```powershell
.\.venv\Scripts\python.exe database\mysql_local.py shell
```

进入 `mysql>` 后，可以执行：

```sql
SELECT @@datadir, @@port;
SHOW TABLES;
DESCRIBE qa_history;
SELECT COUNT(*) FROM qa_history;
```

- `SELECT @@datadir, @@port`：查看当前实例实际使用的数据目录和端口。
- `SHOW TABLES`：查看当前数据库中的表。
- `DESCRIBE qa_history`：查看问答记录表有哪些字段。
- `SELECT COUNT(*)`：查看当前记录数量。
- 输入 `exit` 可退出 MySQL 客户端，数据库服务器继续运行。

停止本项目数据库：

```powershell
.\.venv\Scripts\python.exe database\mysql_local.py stop
```

脚本会先核对数据目录及端口，再执行停止操作。

## 问答记录结构

建表语句保存在 `database/schema.sql`，已在新实例中执行。

| 字段 | 内容 |
| --- | --- |
| `id` | 自动增长的记录编号 |
| `question` | 用户问题 |
| `answer` | 回答正文 |
| `source` | 资料来源名称 |
| `retrieved_chunks` | 本次检索片段的 JSON 快照 |
| `created_at` | 创建时间 |

Python连接配置位于项目的 `db.py`，保存函数为 `save_qa_history`。

POST `/ask` 检索到片段并取得模型回复后，会保存问题、答案、来源和片段快照；事务提交成功后，响应中的 `history_id` 返回记录编号。

- 空白问题和缺少字段的请求不写入数据库。
- 没有检索结果时不调用模型、不写入历史，`history_id` 为 `null`。
- MySQL保存失败时返回HTTP 503，并提示检查数据库连接。
- `test_api.py` 中的接口测试会替换保存函数，不写入真实数据库。
- `check_save_history.py` 可手动检查保存函数，每次新增一条标记的测试记录。
- `check_api_history.py` 可手动检查接口、实际检索与MySQL保存。模型使用替身，每次新增一条标记的测试记录。

## 查询问答历史

- `GET /history`：默认返回最新 10 条摘要，按 `id` 从大到小排列。
- `GET /history?limit=10&offset=10`：跳过前 10 条，读取接下来的 10 条。
- `limit` 是本次最多读取的条数，范围 1～100；`offset` 是跳过的条数，最小为 0。
- 列表响应包含 `total`（总记录数）、`limit`、`offset`、`items`（本页摘要）。摘要包含编号、问题、来源和创建时间。
- `GET /history/{history_id}`：返回指定编号的完整记录，包括回答和 `retrieved_chunks` 列表。
- 没有记录时列表返回空 `items`；指定编号不存在返回 404；参数不合法返回 422；数据库查询失败返回 503。
- 在 `http://127.0.0.1:8000/docs` 展开对应 GET 接口，点击 Try it out 后执行。
- `test_history.py` 使用数据库查询替身，不访问真实数据库或模型。

## 中文路径兼容处理

当前安装的 MySQL 在读取 Windows 中文路径时，需要使用本机 Windows 编码的配置文件。控制脚本会按该编码写入 `my.ini`，并从项目目录通过相对路径加载配置；请通过上面的命令管理实例。

`init` 仅供首次初始化空目录。当前实例已初始化完成，之后使用 `start`；不要重复初始化或直接删除数据目录。
