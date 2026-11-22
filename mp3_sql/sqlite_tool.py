"""One read-only query per bounded, isolated Python worker, never Python eval/exec."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError, TokenError

from mp3_sql.common import MAX_SQL_BYTES

WORKER = r'''
import json, math, sqlite3, sys, time
from pathlib import Path
try:
    import resource
except ImportError:
    resource = None
if resource is not None:
    resource.setrlimit(resource.RLIMIT_AS, (256 * 1024**2, 256 * 1024**2))
    resource.setrlimit(resource.RLIMIT_CPU, (2, 2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (4 * 1024**2, 4 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
p = json.loads(sys.stdin.read())
allowed_functions = set(
    "count sum avg total min max abs round coalesce ifnull nullif lower upper length "
    "trim ltrim rtrim substr substring replace instr like glob printf format date "
    "datetime time julianday strftime unixepoch typeof group_concat hex unicode "
    "char concat concat_ws iif sign ceil ceiling floor pow power sqrt log log10 ln "
    "json_extract json_array_length json_valid row_number rank dense_rank lag lead "
    "ntile first_value last_value nth_value cume_dist percent_rank".split())
allowed_tables = {name.casefold() for name in p["tables"]}
def authorize(action, arg1, arg2, database, source):
    if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
        return sqlite3.SQLITE_OK
    if (action == sqlite3.SQLITE_READ and (arg1 or "").casefold() in allowed_tables
            and database in ("main", None)):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() in allowed_functions:
        return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY
deadline = time.monotonic() + 0.6
steps = 0
def progress():
    global steps
    steps += 1000
    return steps > 5_000_000 or time.monotonic() > deadline
try:
    with sqlite3.connect(Path(p["database"]).as_uri() + "?mode=ro&immutable=1", uri=True) as c:
        c.enable_load_extension(False)
        c.execute("PRAGMA query_only=ON")
        c.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 100_000)
        c.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 16_000)
        c.setlimit(sqlite3.SQLITE_LIMIT_COLUMN, 100)
        c.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 100)
        c.setlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT, 20)
        c.setlimit(sqlite3.SQLITE_LIMIT_VDBE_OP, 100_000)
        c.set_authorizer(authorize)
        c.set_progress_handler(progress, 1000)
        cursor = c.execute(p["sql"])
        rows = cursor.fetchmany(5001)
        if len(rows) > 5000:
            raise ValueError("Result exceeds 5000 rows")
        output = []
        for row in rows:
            cells = []
            for value in row:
                if isinstance(value, bytes):
                    value = "blob:" + value.hex()
                if isinstance(value, str) and len(value.encode()) > 4000:
                    raise ValueError("Result cell exceeds 4000 bytes")
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Non-finite SQL result")
                cells.append(value)
            output.append(cells)
        result = {"ok": True, "columns": [item[0] for item in cursor.description],
                  "rows": output, "error": None}
except (sqlite3.Error, ValueError, MemoryError) as error:
    result = {"ok": False, "columns": [], "rows": [],
              "error": {"type": type(error).__name__, "message": str(error)[:300]}}
encoded = json.dumps(result, ensure_ascii=True, allow_nan=False)
if len(encoded) > 2_000_000:
    encoded = json.dumps({"ok": False, "columns": [], "rows": [],
                          "error": {"type": "OutputLimit", "message": "Result exceeds 2 MB"}})
print(encoded)
'''


def validate_sql(sql):
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a nonempty string")
    if len(sql.encode()) > MAX_SQL_BYTES:
        raise ValueError("SQL exceeds the 16000-byte limit")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except (ParseError, TokenError, RecursionError) as error:
        raise ValueError("SQL parse error: " + str(error)[:200]) from error
    if len(statements) != 1 or not isinstance(statements[0], (exp.Select, exp.SetOperation)):
        raise ValueError("Exactly one SELECT/WITH query is permitted")
    if any(isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop,
                             exp.Command, exp.Into)) for node in statements[0].walk()):
        raise ValueError("Only read-only SQL is permitted")
    return statements[0]


def execute_sql(database, sql, tables, isolated=False):
    try:
        validate_sql(sql)
    except ValueError as error:
        return {"ok": False, "columns": [], "rows": [],
                "error": {"type": "InvalidSQL", "message": str(error)}}
    database = Path(database).absolute()
    if not database.is_file():
        raise FileNotFoundError(f"Database missing: {database}")
    payload = {"database": str(database), "sql": sql, "tables": list(tables)}
    interpreter = str(Path(sys.executable).resolve())
    command = [interpreter, "-I", "-S", "-c", WORKER]
    if isolated:
        if os.name != "posix" or not shutil.which("bwrap"):
            raise RuntimeError("Linux bubblewrap required for isolated instructor verification")
        payload["database"] = "/database.sqlite"
        base = str(Path(sys.base_prefix).resolve())
        command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
                   "--ro-bind", "/usr", "/usr", "--ro-bind", base, base,
                   "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
                   "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
                   "--ro-bind", str(database), "/database.sqlite", "--chdir", "/tmp",
                   "--clearenv", *command]
    environment = {key: value for key, value in os.environ.items()
                   if key in ("SYSTEMROOT", "WINDIR")}
    with tempfile.TemporaryFile() as output:
        try:
            process = subprocess.run(command, input=json.dumps(payload).encode(), stdout=output,
                                     stderr=subprocess.PIPE, env=environment, timeout=3)
        except subprocess.TimeoutExpired:
            return {"ok": False, "columns": [], "rows": [],
                    "error": {"type": "Timeout", "message": "SQL worker exceeded 3 seconds"}}
        if b"bwrap:" in process.stderr:
            raise RuntimeError("SQL sandbox startup failed: " + process.stderr.decode()[:400])
        if process.returncode != 0:
            if process.returncode < 0:
                return {"ok": False, "columns": [], "rows": [],
                        "error": {"type": "ResourceLimit", "message": "SQL worker was terminated"}}
            raise RuntimeError("SQL worker failed: " + process.stderr.decode(errors="replace")[:500])
        output.seek(0)
        text = output.read(2_100_000)
    result = json.loads(text)
    if (set(result) != {"ok", "columns", "rows", "error"}
            or type(result["ok"]) is not bool):
        raise RuntimeError("Malformed SQL worker output")
    return result
