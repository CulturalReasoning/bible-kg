"""Best-worst scaling: tuples for expert/LLM annotation

"""
from __future__ import annotations
from datetime import datetime, timezone

import os
import sys

import requests
import collections
import csv
import json
import random
from pathlib import Path
import time
import numpy as np

from .corpus import Corpus
from .gold import GoldSet
from .paths import DATA, PER_QUERY_OOF, ROOT, annotation_file
from .sparse import BM25Retriever
from .text import jaccard, load_dacy_tokens
import re
TOP_K = 3            #: candidates taken from each retriever
REPS = 4             #: tuples each pair appears in
TUPLE_SIZE = 4
MAX_PER_ANNOTATION = 2   #: pairs sharing one Blixen passage inside a tuple
MAX_PER_BIN = 2          #: pairs from one bin inside a tuple
DEFAULT_SEED = 20260906

BINS = ("1_both", "2_bm25_only", "3_dfmft_only_jac", "4_dfmft_only_zero")

API_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
RETRIES = 3
_LABEL_PAIR = re.compile(r"\b(best|worst)\b\s*[:=]?\s*(?:\w+\s+){0,2}[\"']?([1-4])\b",
                          re.IGNORECASE)
LABELS = "1234"
CHOICES = set(LABELS)

SYSTEM_PROMPT = (
    "You are an expert on intertextuality between Danish literature and the "
    "Danish Bible. You judge how convincing a proposed "
    "intertextual reference is: a passage from Karen Blixen's short stories "
    "paired with a candidate Bible verse. An intertextual reference "
    "can be a quotation that reproduces distinctive wording from a biblical "
    "source passage, allowing for minor linguistic variation, a paraphrase "
    "that reformulates the content of a localized source passage without preserving "
    "its wording, or an allusion evokes a particular biblical passage, event, figure, "
    "or motif more indirectly."
)

def load_finetuned_rankings() -> dict[str, list[int]]:
    rows = json.loads(annotation_file(PER_QUERY_OOF).read_text(encoding="utf-8"))
    return {row["id"]: row["ft_oof_top20"] for row in rows}


def build_pool(corpus: Corpus, gold: GoldSet) -> list[dict]:
    tokens = load_dacy_tokens()
    tokens.check_alignment(len(corpus))
    corpus_tokens = [set(t) for t in tokens.corpus_stem]
    bm25 = BM25Retriever(tokens.corpus_stem)
    finetuned = load_finetuned_rankings()

    rows: list[dict] = []
    seen: set[tuple[str, int]] = set()
    for reference in gold:
        query_tokens = sorted(set(tokens.query_stem[reference.id]))
        bm25_ranks = {doc: rank for rank, doc
                      in enumerate(bm25.search(query_tokens, 200), start=1)}
        ft_ranks = {int(doc): rank for rank, doc
                    in enumerate(finetuned[reference.id], start=1)}

        excluded = (corpus.duplicates_of(reference.gold)
                    | _adjacent_verses(corpus, reference))
        bm25_top = {d for d in bm25.search(query_tokens, TOP_K) if d not in excluded}
        ft_top = {int(d) for d in finetuned[reference.id][:TOP_K] if int(d) not in excluded}

        for candidate in sorted(bm25_top | ft_top):
            if (reference.id, candidate) in seen:
                continue
            seen.add((reference.id, candidate))
            overlap = jaccard(query_tokens, corpus_tokens[candidate])
            verse = corpus[candidate]
            rows.append({
                "pair_id": f"{reference.id}__{verse.book}{verse.chapter}.{verse.verse}",
                "annotation_id": reference.id,
                "reference": corpus.reference(candidate),
                "category": reference.band,
                "bin": _bin_for(candidate, bm25_top, ft_top, overlap),
                "jaccard": f"{overlap:.3f}",
                "blixen_text": reference.text,
                "bible_text": verse.text,
                "bm25_lemstem_rank": bm25_ranks.get(candidate, ""),
                "dfm_ft_rank": ft_ranks.get(candidate, ""),
            })

    rows.sort(key=lambda r: (r["bin"], r["annotation_id"], r["reference"]))
    return rows


def _adjacent_verses(corpus: Corpus, reference) -> set[int]:
    ordered = sorted(reference.gold)
    first, last = corpus[ordered[0]], corpus[ordered[-1]]
    neighbours = (corpus.index_of(first.book, first.chapter, first.verse - 1),
                  corpus.index_of(last.book, last.chapter, last.verse + 1))
    return {i for i in neighbours if i is not None}


def _bin_for(candidate: int, bm25_top: set[int], ft_top: set[int], overlap: float) -> str:
    if candidate in bm25_top and candidate in ft_top:
        return "1_both"
    if candidate in bm25_top:
        return "2_bm25_only"
    return "3_dfmft_only_jac" if overlap > 0 else "4_dfmft_only_zero"


def build_tuples(pool: list[dict], reps: int = REPS, size: int = TUPLE_SIZE,
                 seed: int = DEFAULT_SEED) -> tuple[list[list[str]], int]:
    """Assign pool rows into 4-tuples. Returns (tuples, dropped slots).

    greedy, fill each tuple from the largest bin that has not hit its cap, skipping
    pairs already in the tuple or over the per-annotation cap
    """
    rng = random.Random(seed)
    by_bin: dict[str, list[str]] = collections.defaultdict(list)
    for row in pool:
        by_bin[row["bin"]].extend([row["pair_id"]] * reps)
    for bin_name in by_bin:
        rng.shuffle(by_bin[bin_name])

    info = {row["pair_id"]: row for row in pool}
    tuples: list[list[str]] = []

    while sum(len(v) for v in by_bin.values()) >= size:
        chosen: list[str] = []
        per_annotation: collections.Counter = collections.Counter()
        per_bin: collections.Counter = collections.Counter()

        while len(chosen) < size:
            available = [b for b in by_bin if by_bin[b] and per_bin[b] < MAX_PER_BIN]
            placed = False
            for bin_name in sorted(available, key=lambda b: -len(by_bin[b])):
                for position, pair_id in enumerate(by_bin[bin_name]):
                    annotation = info[pair_id]["annotation_id"]
                    if pair_id in chosen or per_annotation[annotation] >= MAX_PER_ANNOTATION:
                        continue
                    chosen.append(by_bin[bin_name].pop(position))
                    per_annotation[annotation] += 1
                    per_bin[bin_name] += 1
                    placed = True
                    break
                if placed:
                    break
            if not placed:
                break

        if len(chosen) == size:
            rng.shuffle(chosen)
            tuples.append(chosen)
        else:
            for pair_id in chosen:                # return unusable slots and stop
                by_bin[info[pair_id]["bin"]].append(pair_id)
            break

    return tuples, sum(len(v) for v in by_bin.values())


def write_study(pool: list[dict], tuples: list[list[str]], out_dir: Path) -> tuple[Path, Path]:
    info = {row["pair_id"]: row for row in pool}
    annotator_rows, key_rows = [], []
    for number, tuple_pairs in enumerate(tuples, start=1):
        tuple_id = f"T{number:04d}"
        for position, pair_id in enumerate(tuple_pairs, start=1):
            row = info[pair_id]
            annotator_rows.append({
                "tuple_id": tuple_id, "position": position, "pair_id": pair_id,
                "reference": row["reference"], "blixen_text": row["blixen_text"],
                "bible_text": row["bible_text"], "BEST": "", "WORST": "",
            })
            key_rows.append({
                "tuple_id": tuple_id, "position": position, "pair_id": pair_id,
                "annotation_id": row["annotation_id"], "reference": row["reference"],
                "category": row["category"], "bin": row["bin"], "jaccard": row["jaccard"],
                "bm25_lemstem_rank": row["bm25_lemstem_rank"],
                "dfm_ft_rank": row["dfm_ft_rank"],
            })

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, rows in (("bws_tuples_annotator.csv", annotator_rows),
                       ("bws_tuples_key.csv", key_rows)):
        path = out_dir / name
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths[0], paths[1]


def write_pool(pool: list[dict], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "bws_annotation_pool.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(pool[0].keys()))
        writer.writeheader()
        writer.writerows(pool)
    return path


def load_pool(input_path: Path) -> list[dict]:
    """load the quaples to be annotated. check sanity"""
    with input_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["tuple_id"], []).append(row)
    tuples: list[dict] = []
    for tuple_id, members in grouped.items():
        members.sort(key=lambda r: int(r["position"]))

        if len(members) != 4:
            raise ValueError(f"something wrong. {tuple_id} has {len(members)} rows ")
        tuples.append({"tuple_id": tuple_id, "members": members})
    return tuples

def build_messages(tup: dict) -> list[dict]:
    """The prompt for one tuple that contains four pairs of texts"""
    candidates = "\n\n".join(
        f"{label}. \nPassage: {member['blixen_text']}\n"
        f"Verse {member['reference']}: {member['bible_text']}"
        for label, member in zip(LABELS, tup["members"]))

    user = (
        "Below are four candidate (passage, verse) pairs. Decide which pair is "
        "the most convincing intertextual reference (BEST) and which is the "
        "least convincing (WORST).\n\n"
        f"{candidates}\n\n"
        'Answer with JSON only: {"best": "<1|2|3|4>", "worst": "<1|2|3|4>", '
        '"best_reason": "one short sentence", "worst_reason": "one short sentence"}'
    )
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user}]


def query(messages: list[dict], model: str, api_key: str, timeout: int) -> dict:
    """POST the chat request to OpenRouter; retries on network errors and 429s."""
    headers = {"Authorization": f"Bearer {api_key}"}
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload,
                                     timeout=timeout)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2 ** attempt)
            continue
        if response.status_code in (400, 422) and "response_format" in payload:
            payload.pop("response_format", None)   # retry without JSON mode
            continue
        if response.status_code == 429:
            retry_after = response.headers.get("Retry-After")
            time.sleep(float(retry_after) if retry_after else 2 ** attempt)
            continue
        if response.status_code in (401, 403):
            raise RuntimeError(
                f"OpenRouter authentication failed (HTTP {response.status_code}); "
                "check OPENROUTER_API_KEY")
        if response.status_code != 200:
            last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:200]}")
            time.sleep(2 ** attempt)
            continue
        return response.json()
    raise RuntimeError(f"OpenRouter request failed after {RETRIES} attempts: {last_error}")

def parse_answer(content: str) -> dict:
    """Extract BEST/WORST from the model's answer, robust to prose and fences."""
    text = content.strip()
    cleaned = re.sub(r"```(?:json)?\s*\n?(.*?)\n?\s*```", r"\1", text,
                     flags=re.DOTALL)
    data: dict = {}
    try:
        candidate = json.loads(cleaned)
        if isinstance(candidate, dict):
            data = candidate
    except json.JSONDecodeError:
        pass

    picks = {key.lower(): str(value).strip().upper()
             for key, value in data.items()
             if key.lower() in ("best", "worst") and str(value).strip().upper() in CHOICES}
    for match in _LABEL_PAIR.finditer(text):
        picks.setdefault(match.group(1).lower(), match.group(2).upper())

    best, worst = picks.get("best", ""), picks.get("worst", "")
    status = "ok" if best in CHOICES and worst in CHOICES and best != worst else "unparseable"
    reasons = {key.lower(): value for key, value in data.items()
               if isinstance(value, str) and key.lower().endswith("_reason")}
    return {"status": status, "best": best, "worst": worst,
            "best_reason": reasons.get("best_reason", ""),
            "worst_reason": reasons.get("worst_reason", "")}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_raw_answer(raw_dir: Path, tuple_id: str, model: str,
                     messages: list[dict], response: dict) -> Path:
    """Save the prompt sent and the complete API response for one tuple."""
    path = raw_dir / f"{tuple_id}.json"
    path.write_text(json.dumps({
        "tuple_id": tuple_id,
        "model": model,
        "timestamp": _utc_now(),
        "messages": messages,
        "response": response,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def write_parsed_answer(parsed_dir: Path, tup: dict, model: str,
                        response: dict) -> tuple[Path, dict]:
    """Extract the annotation from the API response and save it.

    Returns the written path and the parsed answer, so callers can report
    the BEST/WORST picks without parsing twice.
    """
    message = response["choices"][0].get("message", {})
    content = message.get("content") or ""
    if isinstance(content, list):
        content = " ".join(str(part) for part in content)
    answer = parse_answer(content)

    labels = {label: member for label, member in zip(LABELS, tup["members"])}
    parsed = {
        "tuple_id": tup["tuple_id"],
        "model": model,
        "timestamp": _utc_now(),
        "status": answer["status"],
        "best": answer["best"],
        "worst": answer["worst"],
        "best_pair_id": labels.get(answer["best"], {}).get("pair_id", ""),
        "worst_pair_id": labels.get(answer["worst"], {}).get("pair_id", ""),
        "best_reference": labels.get(answer["best"], {}).get("reference", ""),
        "worst_reference": labels.get(answer["worst"], {}).get("reference", ""),
        "best_reason": answer["best_reason"],
        "worst_reason": answer["worst_reason"],
        "candidates": {
            label: {"pair_id": member["pair_id"],
                    "reference": member["reference"],
                    "blixen_text": member["blixen_text"],
                    "bible_text": member["bible_text"]}
            for label, member in labels.items()},
        "usage": response.get("usage", {}),
    }
    path = parsed_dir / f"{tup['tuple_id']}.json"
    path.write_text(json.dumps(parsed, ensure_ascii=False, indent=2), encoding="utf-8")
    return path, answer


def write_aggregate(parsed_dir: Path, out_dir: Path) -> Path:
    """Rebuild the combined CSV from every parsed file."""
    rows = [json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(parsed_dir.glob("*.json"))]
    fieldnames = ["tuple_id", "status", "best", "worst", "best_pair_id",
                  "worst_pair_id", "best_reference", "worst_reference",
                  "best_reason", "worst_reason", "model", "timestamp"]
    path = out_dir / "bws_tuples_annotator.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return path


def run_toy(tuples: list[dict], out_dir: Path, model: str, count: int) -> None:
    """Save and print the first `count` prompts without querying the API."""
    prompt_dir = out_dir / "prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    for tup in tuples[:count]:
        messages = build_messages(tup)
        path = prompt_dir / f"{tup['tuple_id']}.json"
        path.write_text(json.dumps({
            "tuple_id": tup["tuple_id"],
            "model": model,
            "messages": messages,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n{'-' * 78}\n{tup['tuple_id']} -> {path}\n{'-' * 78}")
        for message in messages:
            print(f"[{message['role']}]\n{message['content']}\n")
    print(f"saved {len(tuples[:count])} prompts to {prompt_dir} - no API calls were made")


def run_llm_bws(input_path: Path = ROOT / "data" / "bws_tuples_annotator.csv",
                out_dir: Path = DATA / "LLM_annotations",
                model: str = DEFAULT_MODEL,
                limit: int = 0, toy: int | None = None,
                overwrite: bool = False, timeout: int = 120) -> None:
    """Annotate the BWS tuples with an LLM through the OpenRouter API.

    Reads `input_path` (four rows per tuple, as written by build_bws.py) and
    asks the LLM, once per tuple, which candidate pair is the BEST match for
    its Blixen passage and which is the WORST. The API key is read from the
    OPENROUTER_API_KEY environment variable.

    Every answer is saved immediately under `out_dir/raw/` and
    `out_dir/parsed/`; tuples that already have a saved response are skipped
    on rerun unless `overwrite` is set, so a crashed run can simply be
    restarted. A combined CSV is rebuilt from the parsed files at the end.
    With `toy=N` the first N prompts are instead written to
    `out_dir/prompts/` and printed, without any API call.
    """
    tuples = load_pool(input_path)

    if toy:
        run_toy(tuples, out_dir, model, toy)
        return

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        sys.exit("OPENROUTER_API_KEY is not set - export it before running, "
                 "e.g. $env:OPENROUTER_API_KEY = \"sk-or-v1-...\"")

    raw_dir = out_dir / "raw"
    parsed_dir = out_dir / "parsed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    parsed_dir.mkdir(parents=True, exist_ok=True)
    pending = [tup for tup in tuples
               if overwrite or not (raw_dir / f"{tup['tuple_id']}.json").exists()]
    if limit:
        pending = pending[:limit]
    print(f"{len(pending)} of {len(tuples)} tuples to annotate "
          f"(model {model}, out {out_dir})")
    if not pending:
        aggregate = write_aggregate(parsed_dir, out_dir)
        print(f"nothing to annotate; wrote {aggregate}")
        return

    done = failed = 0
    for number, tup in enumerate(pending, start=1):
        try:
            messages = build_messages(tup)
            response = query(messages, model, api_key, timeout)
            # save both raw and parsed responses of the LLM
            write_raw_answer(raw_dir, tup["tuple_id"], model, messages, response)
            _, answer = write_parsed_answer(parsed_dir, tup, model, response)

            done += 1
            print(f"[{number}/{len(pending)}] {tup['tuple_id']}: "
                  f"best={answer['best']} worst={answer['worst']} ({answer['status']})")
        except (requests.RequestException, RuntimeError, KeyError, IndexError,
                json.JSONDecodeError, OSError) as exc:
            failed += 1
            print(f"[{number}/{len(pending)}] {tup['tuple_id']}: ERROR {exc}",
                  file=sys.stderr)

    aggregate = write_aggregate(parsed_dir, out_dir)
    print(f"\ndone: {done} annotated, {failed} failed, "
          f"{len(tuples) - len(pending)} skipped (already saved)")
    print(f"wrote {aggregate}")

