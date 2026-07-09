"""
b1_create_embeddings_upsert.py
Option B v1: create embeddings from Part A chunk JSONL and upsert into Supabase.

Run after:
    1. python evaluate_20_openai.py
    2. running sql/01_create_vector_store.sql in Supabase SQL Editor

Run:
    python b1_create_embeddings_upsert.py
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from supabase import create_client


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
SUPABASE_URL = os.getenv("SUPABASE_URL", "").strip()
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "").strip()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
EMBEDDING_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "1536"))

CHUNK_TABLE = os.getenv("CHUNK_TABLE", "trace_eval_chunks")
SUMMARY_TABLE = os.getenv("SUMMARY_TABLE", "trace_eval_summaries")

A_CHUNKS_JSONL = BASE_DIR / os.getenv(
    "A_CHUNKS_JSONL",
    "outputs_A_v6/openai_eval_20_chunks_for_embeddings.jsonl",
)
A_FINAL_JSONL = BASE_DIR / os.getenv(
    "A_FINAL_JSONL",
    "outputs_A_v6/openai_eval_20_final_labels.jsonl",
)

UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "20"))
SLEEP_BETWEEN_EMBEDDING_CALLS_SECONDS = float(os.getenv("SLEEP_BETWEEN_EMBEDDING_CALLS_SECONDS", "0.10"))


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    sys.exit(1)


def require_env() -> None:
    if not OPENAI_API_KEY or OPENAI_API_KEY in {"paste_your_openai_key_here", "paste_your_key_here"}:
        fail("Set OPENAI_API_KEY in .env")
    if not SUPABASE_URL or "your-project-ref" in SUPABASE_URL:
        fail("Set SUPABASE_URL in .env")
    if not SUPABASE_ANON_KEY or SUPABASE_ANON_KEY == "paste_your_supabase_anon_key_here":
        fail("Set SUPABASE_ANON_KEY in .env")
    if not A_CHUNKS_JSONL.exists():
        fail(
            "Part A chunk JSONL not found. Run `python evaluate_20_openai.py` first. "
            f"Expected: {A_CHUNKS_JSONL}"
        )


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSONL in {path} at line {line_number}: {exc}")
    return rows


def embed_text(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def safe_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def make_summary_records(final_rows: list[dict[str, Any]], chunk_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Upsert transcript-level summaries.

    Preferred source is Part A final labels. If that file is missing, create a minimal summary
    from chunk metadata so the chunk table still has a parent run_id.
    """
    summaries: dict[str, dict[str, Any]] = {}

    for row in final_rows:
        run_id = str(row.get("run_id") or "").strip()
        if not run_id:
            continue
        summaries[run_id] = {
            "run_id": run_id,
            "model": row.get("model") or "",
            "system_prompt_version": row.get("system_prompt_version") or "",
            "user_input": row.get("user_input") or "",
            "final_response": row.get("final_response") or "",
            "animal_harm_issue": safe_bool(row.get("animal_harm_issue")),
            "bias_issue": safe_bool(row.get("bias_issue")),
            "model_not_responded_or_refused": safe_bool(row.get("model_not_responded_or_refused")),
            "safety_evaluation": row.get("safety_evaluation") or None,
            "reason": row.get("reason") or "",
            "eval_error": row.get("eval_error") or "",
            "metadata": {"source": "A_v6_final_labels"},
        }

    # Minimal fallback for every run_id that appears in chunks.
    for chunk in chunk_rows:
        run_id = str(chunk.get("run_id") or "").strip()
        if not run_id or run_id in summaries:
            continue
        summaries[run_id] = {
            "run_id": run_id,
            "model": chunk.get("source_model") or "",
            "system_prompt_version": chunk.get("system_prompt_version") or "",
            "user_input": "",
            "final_response": "",
            "animal_harm_issue": None,
            "bias_issue": None,
            "model_not_responded_or_refused": None,
            "safety_evaluation": None,
            "reason": "Final row-level summary was not available when embeddings were upserted.",
            "eval_error": "missing_final_labels_jsonl",
            "metadata": {"source": "A_v6_chunk_metadata_fallback"},
        }

    return list(summaries.values())


def make_chunk_record(openai_client: OpenAI, chunk: dict[str, Any], index: int, total: int) -> dict[str, Any]:
    """Create one Supabase row from one Part A chunk JSONL record."""
    chunk_text = str(chunk.get("chunk_text") or "")
    if not chunk_text.strip():
        fail(f"Empty chunk_text for chunk_id={chunk.get('chunk_id')}")

    embedding = embed_text(openai_client, chunk_text)

    chunk_evaluation = chunk.get("chunk_evaluation") or {}
    metadata = {
        "record_type": chunk.get("record_type"),
        "a_version": chunk.get("a_version"),
        "evaluator_model": chunk.get("evaluator_model"),
        "row_number_global": chunk.get("row_number_global"),
        "thread_id": chunk.get("thread_id"),
        "date_time": chunk.get("date_time"),
        "source_model": chunk.get("source_model"),
        "system_prompt_version": chunk.get("system_prompt_version"),
        "chunk_character_start": chunk.get("chunk_character_start"),
        "chunk_character_end": chunk.get("chunk_character_end"),
        "animal_harm_intent_signal": chunk_evaluation.get("animal_harm_intent_signal"),
        "bias_intent_signal": chunk_evaluation.get("bias_intent_signal"),
        "refusal_or_safe_redirect_signal": chunk_evaluation.get("refusal_or_safe_redirect_signal"),
        "non_response_signal": chunk_evaluation.get("non_response_signal"),
        "harmful_or_biased_compliance_signal": chunk_evaluation.get("harmful_or_biased_compliance_signal"),
        "benign_substantive_answer_signal": chunk_evaluation.get("benign_substantive_answer_signal"),
    }

    return {
        "chunk_id": str(chunk.get("chunk_id")),
        "run_id": str(chunk.get("run_id")),
        "chunk_index": int(chunk.get("chunk_index") or 0),
        "chunk_count": int(chunk.get("chunk_count") or 0),
        "previous_chunk_id": chunk.get("previous_chunk_id"),
        "next_chunk_id": chunk.get("next_chunk_id"),
        "chunk_text": chunk_text,
        "chunk_evaluation": chunk_evaluation,
        "metadata": metadata,
        "embedding": embedding,
    }


def upsert_in_batches(supabase, table_name: str, rows: list[dict[str, Any]], batch_size: int) -> None:
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        supabase.table(table_name).upsert(batch).execute()
        print(f"  upserted {min(start + batch_size, len(rows))}/{len(rows)} rows into {table_name}")


def main() -> None:
    require_env()

    print("Option B v1: embedding Part A chunk records and upserting to Supabase")
    print(f"Input chunks: {A_CHUNKS_JSONL}")
    print(f"Final labels: {A_FINAL_JSONL if A_FINAL_JSONL.exists() else 'not found, using minimal summaries'}")
    print(f"Embedding model: {EMBEDDING_MODEL} ({EMBEDDING_DIMENSIONS} dimensions)")
    print(f"Supabase chunk table: {CHUNK_TABLE}\n")

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    chunk_rows = read_jsonl(A_CHUNKS_JSONL)
    final_rows = read_jsonl(A_FINAL_JSONL) if A_FINAL_JSONL.exists() else []

    if not chunk_rows:
        fail("No chunk records found in Part A JSONL.")

    summary_records = make_summary_records(final_rows, chunk_rows)
    print(f"Upserting {len(summary_records)} transcript summaries...")
    upsert_in_batches(supabase, SUMMARY_TABLE, summary_records, UPSERT_BATCH_SIZE)

    print(f"\nEmbedding and preparing {len(chunk_rows)} chunk rows...")
    db_chunk_records: list[dict[str, Any]] = []
    for i, chunk in enumerate(chunk_rows, start=1):
        print(f"  embedding chunk {i}/{len(chunk_rows)}: {chunk.get('chunk_id')}")
        db_chunk_records.append(make_chunk_record(openai_client, chunk, i, len(chunk_rows)))
        time.sleep(SLEEP_BETWEEN_EMBEDDING_CALLS_SECONDS)

    print(f"\nUpserting {len(db_chunk_records)} vector chunks...")
    upsert_in_batches(supabase, CHUNK_TABLE, db_chunk_records, UPSERT_BATCH_SIZE)

    print("\nDone.")
    print("You can now run: python b1_retrieve_bias_example.py")


if __name__ == "__main__":
    main()
