# Linux 运行与备份记录



## 部署环境



- 操作系统：CentOS Stream 9

- Python：3.12

- Nginx：1.20.1

- MySQL：8.0.46

- Linux 用户：zz

- 项目目录：/home/zz/projects/course-assistant

- 页面入口：http://192.168.65.132/chat



## 请求链路



浏览器 → Nginx（80）→ FastAPI（127.0.0.1:8000）



FastAPI 使用 MySQL 保存问答记录及引用片段快照。



## 服务管理



以下命令在 Linux 执行：



```bash

systemctl status nginx course-assistant mysqld --no-pager

sudo journalctl -u course-assistant -n 30 --no-pager

sudo tail -n 20 /var/log/nginx/access.log

sudo tail -n 20 /var/log/nginx/error.log

```



## 备份位置



- 页面备份：

&#x20; /home/zz/backups/course-assistant/v0.1.0/index.html

- 本次数据备份：

&#x20; /home/zz/backups/course-assistant/data-20261110-113318

- Windows 数据副本：

&#x20; D:\电子实训\git练习\项目备份\course-assistant\data-20261110-113318



## 2026-11-10 验证结果



- 页面备份与回滚后的文件 SHA256 一致。

- 数据库导出、临时库导入的退出码均为 0。

- 原库与恢复库的 qa_history 表均有 3 条记录。

- 两张表的 CHECKSUM 均为 4175429218。

- TXT 和 SQL 备份下载到 Windows 后，SHA256 与 Linux 原文件一致。



这些结果对应本次备份快照。

代码版本、上传资料和数据库数据分别保存。

