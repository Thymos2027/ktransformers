"""Freeze diagnostic prompts and histories using the running TP1 tokenizer."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request


HERE = Path(__file__).resolve().parent
BASE = "http://192.168.137.11:18080"
PRIOR = (HERE.parent / "2026-11-11-taishan-tp-speed" / "runs" /
         "speed-20261111-graph-check" / "remote" / "requests.jsonl")
PROMPTS = [
    ("chinese", "请用一句话解释为何太阳会升起。"),
    ("english", "Explain in one sentence why a checksum is useful."),
    ("arithmetic", "计算 17 + 25。只输出数字答案。"),
    ("json", 'Return only JSON with keys "a" and "b" for the values 1 and 2.'),
    ("code", "Write a short Python function that squares an integer."),
    ("mixed", "用 English 和中文各写一句关于机器学习的说明。"),
]


def sha(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False,
                                     sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tokenize(payload):
    request = urllib.request.Request(
        BASE + "/tokenize", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        value = json.load(response)
    if value["count"] != len(value["tokens"]):
        raise RuntimeError("Tokenizer returned inconsistent count")
    return value["tokens"]


def main():
    speed_cases = json.loads((HERE.parent / "2026-11-11-taishan-tp-speed" /
                              "requests.json").read_text(encoding="utf-8"))
    old = [json.loads(line) for line in PRIOR.read_text(encoding="utf-8").splitlines()]
    cases = []
    for name, prompt in PROMPTS:
        messages = [{"role": "user", "content": prompt}]
        ids = tokenize({"messages": messages})
        history = tokenize({"prompt": "The answer is 42. I can explain.",
                            "add_special_tokens": False})
        cases.append({"name": name, "messages": messages, "prompt_ids": ids,
                      "forced_ids": history[:8], "history_source": "fixed_text"})
    for source_tp, speed_case in ((1, speed_cases[0]), (2, speed_cases[1])):
        prior = next(x for x in old if x["tp"] == source_tp and x["case"] == speed_case["name"]
                     and x["kind"] == "formal")
        ids = tokenize({"messages": speed_case["messages"]})
        if ids != speed_case["token_ids"]:
            raise RuntimeError("Prior speed prompt tokenization changed")
        history = tokenize({"prompt": prior["content"], "add_special_tokens": False})
        cases.append({"name": f"prior_tp{source_tp}", "messages": speed_case["messages"],
                      "prompt_ids": ids, "forced_ids": history[:8],
                      "history_source": {"phase": prior["phase"], "case": prior["case"],
                                         "content_sha256": prior["content_sha256"]}})
    if len(cases) != 8 or any(len(x["forced_ids"]) != 8 for x in cases):
        raise RuntimeError("Expected eight nonempty eight-token histories")
    for case in cases:
        case["prompt_sha256"] = sha(case["prompt_ids"])
        case["history_sha256"] = sha(case["forced_ids"])
    target = HERE / "cases.json"
    if target.exists():
        raise RuntimeError("Frozen cases already exist; do not overwrite")
    target.write_text(json.dumps(cases, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"{target}: {sha(cases)}")


if __name__ == "__main__":
    main()
