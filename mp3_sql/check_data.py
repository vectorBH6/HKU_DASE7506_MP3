"""Verify the released public bank, database hashes, supervision and split isolation."""
import json

from mp3_sql.common import ROOT, digest, load_bank, read_jsonl


def main():
    folder = ROOT / "data"
    manifest = json.loads((folder / "manifest.json").read_text())
    observed, identifiers = {}, {}
    for split in ("train", "validation", "test"):
        rows, _, _ = load_bank(split, folder)
        observed[split] = {row["db_id"] for row in rows}
        identifiers[split] = {row["id"] for row in rows}
        if split in ("train", "validation", "test"):
            references = json.loads((folder / f"{split}_references.json").read_text())
            if references["bank_sha256"] != manifest["splits"][split]["sha256"]:
                raise ValueError("Supervision bank hash mismatch")
            if references["ids"] != [row["id"] for row in rows]:
                raise ValueError("Supervision order mismatch")
            for cases in references["tasks"].values():
                for case in cases:
                    if digest(folder / case["path"]) != case["database_sha256"]:
                        raise ValueError("Training/validation case database mismatch")
        print(f"{split}: {len(rows)} tasks, {len(observed[split])} schemas, hashes verified")
    for left, right in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if observed[left] & observed[right] or identifiers[left] & identifiers[right]:
            raise ValueError("Schema or task leakage between splits")
    for metadata in manifest["databases"].values():
        if digest(folder / metadata["public_path"]) != metadata["public_sha256"]:
            raise ValueError("Public database hash mismatch")
    for name, expected in manifest["support_files"].items():
        if digest(folder / name) != expected:
            raise ValueError("Public tool support hash mismatch")
    memory = read_jsonl(folder / "train_memory.jsonl")
    if {item["id"] for item in memory} != identifiers["train"]:
        raise ValueError("Retrieval memory must contain exactly the training tasks")
    results = json.loads((folder / "public_results.json").read_text())
    if set(results) != set.union(*identifiers.values()):
        raise ValueError("Public example alignment mismatch")
    print("Public data OK; all three evaluation cases per test task are available.")


if __name__ == "__main__":
    main()
