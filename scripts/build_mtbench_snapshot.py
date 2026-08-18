"""Local-only: build the committed MT-Bench agreement snapshot from HuggingFace.

Fetches the ``human`` and ``gpt4_pair`` splits of ``lmsys/mt_bench_human_judgments``
(CC-BY-4.0) through the HF datasets-server, keeps only the label/key columns (the
embedded conversations are intentionally dropped), and writes deterministic JSONL
snapshots plus a ``PROVENANCE.json``.

CI never runs this script — it only reads the committed output. Re-run locally to
refresh the snapshot:  ``uv run python scripts/build_mtbench_snapshot.py``
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DATASET = "lmsys/mt_bench_human_judgments"
CONFIG = "default"
SPLITS = ("human", "gpt4_pair")
FIELDS = ("question_id", "model_a", "model_b", "winner", "judge", "turn")
BASE = "https://datasets-server.huggingface.co"
PAGE = 100
RETRIEVED = "2026-08-18"
OUT = Path(__file__).resolve().parent.parent / "data" / "snapshots" / "mtbench"


def _get(url: str, *, retries: int = 6) -> dict[str, Any]:
    delay = 2.0
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=60) as response:
                data: dict[str, Any] = json.load(response)
            return data
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt < retries - 1:
                time.sleep(delay)
                delay *= 2
                continue
            raise
    raise RuntimeError("unreachable")


def _fetch_split(split: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    while True:
        url = (
            f"{BASE}/rows?dataset={DATASET}&config={CONFIG}"
            f"&split={split}&offset={offset}&length={PAGE}"
        )
        payload = _get(url)
        batch = payload["rows"]
        if not batch:
            break
        rows.extend({field: item["row"][field] for field in FIELDS} for item in batch)
        offset += len(batch)
        total = payload.get("num_rows_total")
        if total is not None and offset >= total:
            break
        time.sleep(0.5)  # politeness between pages to avoid rate limiting
    rows.sort(key=lambda r: (r["question_id"], r["turn"], r["model_a"], r["model_b"], r["judge"]))
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> str:
    text = "".join(json.dumps(r, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    meta = _get(f"https://huggingface.co/api/datasets/{DATASET}")
    provenance: dict[str, Any] = {
        "dataset": DATASET,
        "config": CONFIG,
        "revision": meta.get("sha", "unknown"),
        "source": f"https://huggingface.co/datasets/{DATASET}",
        "license": "CC-BY-4.0",
        "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "citation": "Zheng et al. 2023, Judging LLM-as-a-Judge, arXiv:2306.05685",
        "retrieved": RETRIEVED,
        "changes": "kept label/key columns only; embedded conversations dropped; rows sorted",
        "files": {},
    }
    for split in SPLITS:
        rows = _fetch_split(split)
        digest = _write_jsonl(OUT / f"{split}.jsonl", rows)
        provenance["files"][f"{split}.jsonl"] = {"rows": len(rows), "sha256": digest}
        print(f"{split}: {len(rows)} rows  sha256={digest[:12]}...")
    (OUT / "PROVENANCE.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("wrote PROVENANCE.json")


if __name__ == "__main__":
    main()
