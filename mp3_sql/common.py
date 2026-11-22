"""Shared formats and identities; no grading references are imported here."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROTOCOL = "dase7506-sql-repair-v1"
BUDGET = 8
MAX_SQL_BYTES = 16000
SPLITS = ("train", "validation", "test", "audit")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n"
                            for row in rows), encoding="utf-8")


def result_fingerprint(result, ordered=False):
    if not result["ok"]:
        raise ValueError("A failed execution has no result fingerprint")
    rows = []
    for row in result["rows"]:
        cells = []
        for cell in row:
            if cell is None:
                cells.append(["null"])
            elif isinstance(cell, (int, float)):
                number = str(cell) if isinstance(cell, int) else format(round(cell, 6), ".6f")
                if "." in number:
                    number = number.rstrip("0").rstrip(".")
                cells.append(["number", "0" if number == "-0" else number])
            else:
                cells.append(["text", cell])
        rows.append(json.dumps(cells, sort_keys=True, ensure_ascii=True))
    if not ordered:
        rows.sort()
    value = {"columns": len(result["columns"]), "rows": rows}
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_bank(split, data_dir=None):
    if split not in SPLITS:
        raise ValueError("Unknown split")
    folder = Path(data_dir) if data_dir is not None else ROOT / "data"
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest["protocol"] != PROTOCOL:
        raise ValueError("Bank protocol mismatch")
    rows = read_jsonl(folder / f"{split}.jsonl")
    metadata = manifest["splits"][split]
    if len(rows) != metadata["n"] or digest(folder / f"{split}.jsonl") != metadata["sha256"]:
        raise ValueError("Bank count/hash mismatch")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task IDs")
    fields = {"id", "db_id", "question", "issue_sql", "ordered", "example"}
    for row in rows:
        if (set(row) != fields or type(row["ordered"]) is not bool
                or any(not isinstance(row[key], str) for key in
                       ("id", "db_id", "question", "issue_sql"))
                or row["db_id"] not in manifest["databases"]):
            raise ValueError("Invalid task layout")
    return rows, manifest, folder
