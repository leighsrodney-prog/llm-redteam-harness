"""Loaders for public benchmarks. Downloads only from public URLs, caches locally and
records a sha256 of the exact file used. Gated or code-only benchmarks are stubbed and
raise GatedDatasetError with instructions; pass a local path once you have legitimate access.
"""
import csv
import hashlib
import io
import json
import os
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from .filter import apply_safety_filter

CACHE = Path(os.environ.get("HARNESS_DATA_CACHE", Path.home() / ".cache" / "llm-redteam-harness"))


class GatedDatasetError(RuntimeError):
    pass


@dataclass
class Item:
    id: str
    source: str
    category: str
    prompt: str
    subcategory: str = ""
    expect: str = "comply"            # refuse | comply | answer
    reference: list = field(default_factory=list)   # gold answers (QA benchmarks)
    incorrect: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(url, name):
    """Download a public file once into the cache. Never sends prompts anywhere."""
    CACHE.mkdir(parents=True, exist_ok=True)
    dest = CACHE / name
    if not dest.exists():
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
    return dest


def _csv_rows(path):
    return list(csv.DictReader(io.StringIO(Path(path).read_text(encoding="utf-8"))))


# ---- individual parsers (path -> list[Item]) ----

def parse_harmbench(path):
    out = []
    for r in _csv_rows(path):
        bid = r.get("BehaviorID") or r.get("id")
        out.append(Item(bid, "harmbench", r.get("SemanticCategory", ""), r["Behavior"],
                        subcategory=r.get("FunctionalCategory", ""), expect="refuse",
                        meta={"context": r.get("ContextString", "")}))
    return out


def parse_xstest(path):
    out = []
    for r in _csv_rows(path):
        unsafe = r.get("label", "").strip() == "unsafe" or r.get("type", "").startswith("contrast")
        out.append(Item(f"xstest-{r['id']}", "xstest", r.get("type", ""), r["prompt"],
                        subcategory=r.get("focus", ""), expect="refuse" if unsafe else "comply",
                        meta={"label": r.get("label", "")}))
    return out


def parse_truthfulqa(path):
    out = []
    for i, r in enumerate(_csv_rows(path)):
        out.append(Item(f"tqa-{i:04d}", "truthfulqa", r.get("Category", ""), r["Question"], expect="answer",
                        reference=[r.get("Best Answer", "")] + [x.strip() for x in r.get("Correct Answers", "").split(";") if x.strip()],
                        incorrect=[x.strip() for x in r.get("Incorrect Answers", "").split(";") if x.strip()]))
    return out


def parse_simpleqa(path):
    out = []
    for i, r in enumerate(_csv_rows(path)):
        try:
            topic = json.loads(r.get("metadata", "{}").replace("'", '"')).get("topic", "")
        except ValueError:
            topic = ""
        out.append(Item(f"sqa-{i:04d}", "simpleqa", topic, r["problem"], expect="answer", reference=[r["answer"]]))
    return out


def parse_injecagent(path):
    data = json.loads(Path(path).read_text())
    out = []
    for i, r in enumerate(data):
        out.append(Item(f"injecagent-{Path(path).stem}-{i:04d}", "injecagent", r.get("Attack Type", "indirect_injection"),
                        r.get("User Instruction", ""), expect="refuse",
                        meta={"attacker_instruction": r.get("Attacker Instruction", ""),
                              "user_tool": r.get("User Tool", ""), "attacker_tools": r.get("Attacker Tools", [])}))
    return out


def parse_agentharm(path):
    data = json.loads(Path(path).read_text())
    rows = data.get("behaviors", data) if isinstance(data, dict) else data
    return [Item(f"agentharm-{r['id']}", "agentharm", r.get("category", ""), r["prompt"], expect="refuse",
                 meta={"target_functions": r.get("target_functions", [])}) for r in rows]


def _gated(name, how):
    def f(path=None):
        if path is None:
            raise GatedDatasetError(f"{name}: {how}")
        raise GatedDatasetError(f"{name}: local parser not implemented yet; see docs/datasets.md")
    return f


GH = "https://raw.githubusercontent.com"
LOADERS = {
    "harmbench": {"url": f"{GH}/centerforaisafety/HarmBench/main/data/behavior_datasets/harmbench_behaviors_text_all.csv",
                  "file": "harmbench_behaviors_text_all.csv", "parse": parse_harmbench},
    "xstest": {"url": f"{GH}/paul-rottger/exaggerated-safety/main/xstest_prompts.csv",
               "file": "xstest_prompts.csv", "parse": parse_xstest},
    "truthfulqa": {"url": f"{GH}/sylinrl/TruthfulQA/main/TruthfulQA.csv", "file": "TruthfulQA.csv", "parse": parse_truthfulqa},
    "simpleqa": {"url": "https://openaipublic.blob.core.windows.net/simple-evals/simple_qa_test_set.csv",
                 "file": "simple_qa_test_set.csv", "parse": parse_simpleqa},
    "injecagent": {"url": f"{GH}/uiuc-kang-lab/InjecAgent/main/data/test_cases_dh_base.json",
                   "file": "injecagent_dh_base.json", "parse": parse_injecagent},
    # Hugging Face dataset with access terms; download it yourself and pass --path.
    "agentharm": {"url": None, "file": None, "parse": parse_agentharm,
                  "gated": "accept the terms at huggingface.co/datasets/ai-safety-institute/AgentHarm, download "
                           "harmful_behaviors_test_public.json and pass path="},
    # Code-defined environments (Python packages), not flat files: adapter stub only.
    "agentdojo": {"url": None, "file": None, "parse": _gated("agentdojo", "tasks are Python objects, not files; use `python -m harness agentdojo` (harness/bridges/agentdojo_offline.py) instead.")},
    "taubench": {"url": None, "file": None, "parse": _gated("tau-bench", "clone sierra-research/tau-bench; tasks need its simulated user + DB. Bridge not implemented.")},
}


def load_benchmark(name, path=None, limit=None, ids=None):
    """Return (items, manifest). The chem/bio/WMD filter is always applied."""
    spec = LOADERS[name]
    if path is None:
        if not spec["url"]:
            if "gated" in spec:
                raise GatedDatasetError(f"{name}: {spec['gated']}")
            spec["parse"]()  # raises
        path = fetch(spec["url"], spec["file"])
    items = spec["parse"](path)
    kept, dropped = apply_safety_filter(items)
    if ids is not None:
        idset = set(ids)
        kept = [i for i in kept if i.id in idset]
    if limit:
        kept = kept[:limit]
    manifest = {"dataset": name, "source": spec["url"] or str(path), "sha256": sha256_file(path),
                "n_raw": len(items), "n_excluded_safety": len(dropped), "n_used": len(kept),
                "excluded_ids": [d["id"] for d in dropped], "item_ids": [i.id for i in kept]}
    return kept, manifest
