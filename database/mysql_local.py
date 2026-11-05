"""Manage this project's independent D-drive MySQL instance.

Credentials and all MySQL runtime files are stored in the ignored .mysql folder.
This script does not change Windows services or existing MySQL installations.
"""

import argparse
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
RUNTIME = ROOT / ".mysql"
DATA = RUNTIME / "data"
LOGS = RUNTIME / "logs"
SETTINGS = RUNTIME / "instance.json"
SERVER_CONFIG = RUNTIME / "my.ini"
ADMIN_CONFIG = RUNTIME / "admin-client.ini"
APP_CONFIG = RUNTIME / "app-client.ini"
BOOTSTRAP = RUNTIME / "bootstrap.sql"
PORT = 3307


def native_command(arguments):
    """Quote every argument for MySQL's Windows command-line parser.

    Configuration filenames are passed relative to ROOT to avoid MySQL's
    Windows command-line decoding problems with Chinese paths. No shell is
    involved; double quotes and trailing backslashes are escaped.
    """
    parts = []
    for argument in arguments:
        value = re.sub(r'(\\*)"', lambda match: match.group(1) * 2 + '\\"', str(argument))
        trailing = len(value) - len(value.rstrip("\\"))
        parts.append('"' + value + "\\" * trailing + '"')
    return " ".join(parts)


def read_settings():
    return json.loads(SETTINGS.read_text(encoding="utf-8"))


def client_config(user, password):
    return (
        "[client]\n"
        "host=127.0.0.1\n"
        f"port={PORT}\n"
        "protocol=TCP\n"
        f"user={user}\n"
        f"password={password}\n"
        "default-character-set=utf8mb4\n"
        "connect-timeout=3\n"
    )


def query(sql, *, admin=False):
    settings = read_settings()
    config = ADMIN_CONFIG if admin else APP_CONFIG
    environment = os.environ.copy()
    environment["MYSQL_TEST_LOGIN_FILE"] = ".mysql/unused-login-file"
    return subprocess.run(
        native_command([settings["mysql"], f"--defaults-file={config.relative_to(ROOT).as_posix()}",
                        "--batch", "--raw", "--skip-column-names", "--execute", sql]),
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=12, env=environment,
    )


def verified_status():
    if not SETTINGS.exists():
        return None
    result = query(
        "SELECT HEX(@@datadir), @@port, HEX(@@tmpdir), HEX(@@log_error);", admin=True
    )
    if result.returncode:
        return None
    fields = result.stdout.strip().split("\t")
    if len(fields) != 4 or fields[1] != str(PORT):
        raise RuntimeError("Connected instance does not match this project.")
    # Windows MySQL can expose filesystem paths in the local ANSI encoding.
    # Read raw path bytes as HEX so the SQL connection does not reinterpret them.
    for index in (0, 2, 3):
        raw_path = bytes.fromhex(fields[index])
        try:
            fields[index] = raw_path.decode("utf-8")
        except UnicodeDecodeError:
            fields[index] = raw_path.decode("mbcs")
    if Path(fields[0]).resolve() != DATA.resolve():
        raise RuntimeError("Connected MySQL data directory is not this project's D-drive directory.")
    for directory in (fields[2], fields[3]):
        if not Path(directory).resolve().is_relative_to(RUNTIME.resolve()):
            raise RuntimeError("MySQL runtime path is outside this project.")
    return dict(zip(("datadir", "port", "tmpdir", "error_log"), fields))


def initialize():
    if ROOT.drive.upper() != "D:":
        raise RuntimeError("This project's database must be stored on D:.")
    if SETTINGS.exists():
        raise RuntimeError("Instance configuration already exists; use start or status. Data was not changed.")
    mysql = shutil.which("mysql")
    if not mysql:
        raise RuntimeError("mysql.exe was not found in PATH.")
    mysql = str(Path(mysql).resolve())
    binary_dir = Path(mysql).parent
    mysqld = binary_dir / "mysqld.exe"
    mysqladmin = binary_dir / "mysqladmin.exe"
    if not mysqld.is_file() or not mysqladmin.is_file():
        raise RuntimeError("The installed MySQL server tools were not found.")
    if DATA.exists() and any(DATA.iterdir()):
        raise RuntimeError("Data directory is not empty; initialization was stopped.")
    for directory in (DATA, LOGS, RUNTIME / "tmp", RUNTIME / "files"):
        directory.mkdir(parents=True, exist_ok=True)

    server_ini = (
        "[mysqld]\n"
        f'basedir="{binary_dir.parent.as_posix()}"\n'
        f'datadir="{DATA.as_posix()}"\n'
        f'tmpdir="{(RUNTIME / "tmp").as_posix()}"\n'
        f'log-error="{(LOGS / "mysql-error.log").as_posix()}"\n'
        f'pid-file="{(RUNTIME / "mysql.pid").as_posix()}"\n'
        f'secure-file-priv="{(RUNTIME / "files").as_posix()}"\n'
        f"port={PORT}\n"
        "bind-address=127.0.0.1\n"
        "mysqlx=OFF\n"
        "skip-log-bin\n"
        "character-set-server=utf8mb4\n"
        "collation-server=utf8mb4_0900_ai_ci\n"
        "innodb-buffer-pool-size=64M\n"
        "innodb-redo-log-capacity=32M\n"
        "max-connections=25\n"
    )
    SERVER_CONFIG.write_text(server_ini, encoding="mbcs")
    settings = {"mysql": mysql, "mysqld": str(mysqld),
                "mysqladmin": str(mysqladmin), "port": PORT}
    SETTINGS.write_text(json.dumps(settings, indent=2), encoding="utf-8")

    root_password = secrets.token_urlsafe(32)
    app_password = secrets.token_urlsafe(32)
    ADMIN_CONFIG.write_text(client_config("root", root_password), encoding="utf-8")
    APP_CONFIG.write_text(client_config("course_app", app_password), encoding="utf-8")
    (RUNTIME / "app.env").write_text(
        "MYSQL_HOST=127.0.0.1\n"
        f"MYSQL_PORT={PORT}\n"
        "MYSQL_USER=course_app\n"
        f"MYSQL_PASSWORD={app_password}\n"
        "MYSQL_DATABASE=course_assistant\n",
        encoding="utf-8",
    )
    schema = (ROOT / "database" / "schema.sql").read_text(encoding="utf-8")
    statements = [
        f"ALTER USER 'root'@'localhost' IDENTIFIED BY '{root_password}';",
    ]
    statements.extend(" ".join(part.split()) + ";" for part in schema.split(";") if part.strip())
    for host in ("localhost", "127.0.0.1"):
        statements.extend([
            f"CREATE USER IF NOT EXISTS 'course_app'@'{host}' IDENTIFIED BY '{app_password}';",
            f"GRANT SELECT, INSERT ON course_assistant.* TO 'course_app'@'{host}';",
        ])
    BOOTSTRAP.write_text("\n".join(statements) + "\n", encoding="utf-8")
    print(f"Initializing database files in {DATA}", flush=True)
    with (LOGS / "initialize.log").open("wb") as output:
        result = subprocess.run(
            native_command([str(mysqld), "--defaults-file=.mysql/my.ini", "--initialize", "--console"]),
            cwd=ROOT, stdout=output, stderr=subprocess.STDOUT, timeout=180,
        )
    if result.returncode:
        raise RuntimeError(f"Initialization failed; inspect {LOGS / 'initialize.log'}. No existing database was modified.")
    print("Initialization complete. Starting project database.", flush=True)
    start()


def start():
    settings = read_settings()
    status = verified_status()
    if status:
        print(json.dumps(status, ensure_ascii=False, indent=2))
        return
    if not (DATA / "mysql").is_dir():
        raise RuntimeError("MySQL system tables were not initialized. Existing files were kept for inspection.")
    command = [settings["mysqld"], "--defaults-file=.mysql/my.ini"]
    configuration_lines = [
        line for line in SERVER_CONFIG.read_text(encoding="mbcs").splitlines()
        if not line.startswith("init-file=")
    ]
    if BOOTSTRAP.exists():
        configuration_lines.append(f'init-file="{BOOTSTRAP.as_posix()}"')
    SERVER_CONFIG.write_text("\n".join(configuration_lines) + "\n", encoding="mbcs")
    with (LOGS / "process-stdout.log").open("ab") as stdout, \
            (LOGS / "process-stderr.log").open("ab") as stderr:
        process = subprocess.Popen(
            native_command(command), cwd=ROOT, stdout=stdout, stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP,
            close_fds=True,
        )
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"MySQL stopped during startup; inspect {LOGS / 'mysql-error.log'}.")
        status = verified_status()
        if status:
            if BOOTSTRAP.exists():
                BOOTSTRAP.unlink()
                SERVER_CONFIG.write_text(
                    "\n".join(line for line in configuration_lines if not line.startswith("init-file=")) + "\n",
                    encoding="mbcs",
                )
            print(json.dumps(status, ensure_ascii=False, indent=2))
            return
        time.sleep(0.5)
    raise RuntimeError("Startup did not finish within 60 seconds. Inspect the D-drive logs before retrying.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("init", "start", "status", "stop", "shell"))
    args = parser.parse_args()
    if args.action == "init":
        initialize()
    elif args.action == "start":
        start()
    elif args.action == "status":
        status = verified_status()
        if not status:
            raise RuntimeError("Project MySQL is not running or is not accessible.")
        print(json.dumps(status, ensure_ascii=False, indent=2))
        result = query("SHOW TABLES FROM course_assistant;")
        if result.returncode:
            raise RuntimeError("Database schema verification failed.")
        print("Tables:", result.stdout.strip())
    elif args.action == "stop":
        if not verified_status():
            raise RuntimeError("Project MySQL is not running or is not accessible.")
        environment = os.environ.copy()
        environment["MYSQL_TEST_LOGIN_FILE"] = ".mysql/unused-login-file"
        result = subprocess.run(native_command([
            read_settings()["mysqladmin"], "--defaults-file=.mysql/admin-client.ini", "shutdown"
        ]), cwd=ROOT, capture_output=True, timeout=30, env=environment)
        if result.returncode:
            raise RuntimeError("Project MySQL shutdown failed.")
        print("Project MySQL stopped.")
    else:
        if not verified_status():
            raise RuntimeError("Start the project database first.")
        environment = os.environ.copy()
        environment["MYSQL_TEST_LOGIN_FILE"] = ".mysql/unused-login-file"
        environment["MYSQL_HISTFILE"] = ".mysql/client-history"
        raise SystemExit(subprocess.call(native_command([
            read_settings()["mysql"], "--defaults-file=.mysql/app-client.ini",
            "--database=course_assistant"
        ]), cwd=ROOT, env=environment))


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    try:
        main()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"Database setup: {error}", file=sys.stderr)
        raise SystemExit(1)
