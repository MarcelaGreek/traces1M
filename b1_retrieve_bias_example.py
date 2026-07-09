"""
b1_retrieve_bias_example.py
Option B v1: super-basic retrieval example for a bias question.

Run after:
    python b1_create_embeddings_upsert.py

Run:
    python b1_retrieve_bias_example.py
"""

from __future__ import annotations

import json
import os
import sys
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
MATCH_FUNCTION = os.getenv("MATCH_FUNCTION", "match_trace_eval_chunks")

QUESTION = os.getenv(
    "RETRIEVAL_QUESTION",
    "Find examples of bias or discrimination in the evaluated traces. Explain whether the model response was safe or not_safe.",
)
MATCH_THRESHOLD = float(os.getenv("MATCH_THRESHOLD", "0.05"))
MATCH_COUNT = int(os.getenv("MATCH_COUNT", "5"))
INCLUDE_NEIGHBOR_CHUNKS = os.getenv("INCLUDE_NEIGHBOR_CHUNKS", "true").strip().lower() == "true"
ASK_OPENAI_AFTER_RETRIEVAL = os.getenv("ASK_OPENAI_AFTER_RETRIEVAL", "true").strip().lower() == "true"
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-5.5")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", "1200"))

OUTPUT_RETRIEVAL_JSON = BASE_DIR / "b1_retrieval_result.json"


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


def embed_query(client: OpenAI, question: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=question,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def search_matching_chunks(supabase, query_embedding: list[float]) -> list[dict[str, Any]]:
    result = supabase.rpc(
        MATCH_FUNCTION,
        {
            "query_embedding": query_embedding,
            "match_threshold": MATCH_THRESHOLD,
            "match_count": MATCH_COUNT,
            "metadata_filter": {},
        },
    ).execute()
    return result.data or []


def unique_values(rows: list[dict[str, Any]], key: str) -> list[str]:
    seen = set()
    out = []
    for row in rows:
        value = row.get(key)
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def fetch_summaries(supabase, run_ids: list[str]) -> list[dict[str, Any]]:
    if not run_ids:
        return []
    result = supabase.table(SUMMARY_TABLE).select("*").in_("run_id", run_ids).execute()
    return result.data or []


def fetch_neighbor_chunks(supabase, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fetch matched chunks plus immediate previous/next chunks by chunk_id."""
    wanted = set()
    for match in matches:
        for key in ("chunk_id", "previous_chunk_id", "next_chunk_id"):
            value = match.get(key)
            if value:
                wanted.add(value)

    if not wanted:
        return []

    wanted_list = sorted(wanted)
    result = (
        supabase.table(CHUNK_TABLE)
        .select("chunk_id, run_id, chunk_index, chunk_count, previous_chunk_id, next_chunk_id, chunk_text, chunk_evaluation, metadata")
        .in_("chunk_id", wanted_list)
        .order("run_id")
        .order("chunk_index")
        .execute()
    )
    return result.data or []


def print_matches(matches: list[dict[str, Any]]) -> None:
    print("\n=== VECTOR SEARCH MATCHES ===")
    if not matches:
        print("No matches returned. Try lowering MATCH_THRESHOLD in .env.")
        return

    for i, row in enumerate(matches, start=1):
        text = (row.get("chunk_text") or "").replace("\n", " ")
        snippet = text[:500] + ("..." if len(text) > 500 else "")
        print(f"\nMatch {i}")
        print(f"  similarity: {row.get('similarity')}")
        print(f"  run_id: {row.get('run_id')}")
        print(f"  chunk: {row.get('chunk_index')}/{row.get('chunk_count')}")
        print(f"  chunk_id: {row.get('chunk_id')}")
        print(f"  labels: {row.get('chunk_evaluation')}")
        print(f"  snippet: {snippet}")


def format_context(
    matches: list[dict[str, Any]],
    neighbors: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
) -> str:
    lines: list[str] = []

    lines.append("MATCHED CHUNKS:")
    for row in matches:
        lines.append(
            f"\n[MATCH run_id={row.get('run_id')} chunk={row.get('chunk_index')}/{row.get('chunk_count')} "
            f"similarity={row.get('similarity')}]\n"
            f"chunk_id={row.get('chunk_id')}\n"
            f"chunk_evaluation={json.dumps(row.get('chunk_evaluation'), ensure_ascii=False)}\n"
            f"text={row.get('chunk_text')}"
        )

    if neighbors:
        lines.append("\n\nNEIGHBOR CHUNKS:")
        for row in neighbors:
            lines.append(
                f"\n[NEIGHBOR run_id={row.get('run_id')} chunk={row.get('chunk_index')}/{row.get('chunk_count')}]\n"
                f"chunk_id={row.get('chunk_id')}\n"
                f"chunk_evaluation={json.dumps(row.get('chunk_evaluation'), ensure_ascii=False)}\n"
                f"text={row.get('chunk_text')}"
            )

    if summaries:
        lines.append("\n\nFINAL ROW SUMMARIES FROM PART A:")
        for row in summaries:
            lines.append(
                f"\n[SUMMARY run_id={row.get('run_id')}]\n"
                f"model={row.get('model')}\n"
                f"system_prompt_version={row.get('system_prompt_version')}\n"
                f"animal_harm_issue={row.get('animal_harm_issue')}\n"
                f"bias_issue={row.get('bias_issue')}\n"
                f"model_not_responded_or_refused={row.get('model_not_responded_or_refused')}\n"
                f"safety_evaluation={row.get('safety_evaluation')}\n"
                f"reason={row.get('reason')}"
            )

    return "\n".join(lines)


def answer_with_openai(client: OpenAI, question: str, context: str) -> str:
    response = client.responses.create(
        model=ANSWER_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You answer questions about previously evaluated model traces. "
                    "Use only the retrieved context. If context is insufficient, say so. "
                    "Do not claim you searched the full database beyond the retrieved chunks."
                ),
            },
            {
                "role": "user",
                "content": f"Question:\n{question}\n\nRetrieved context:\n{context}",
            },
        ],
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )
    return response.output_text


def main() -> None:
    require_env()

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    supabase = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)

    print("Option B v1 retrieval example")
    print(f"Question: {QUESTION}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Match threshold: {MATCH_THRESHOLD}")
    print(f"Match count: {MATCH_COUNT}")

    query_embedding = embed_query(openai_client, QUESTION)
    matches = search_matching_chunks(supabase, query_embedding)
    print_matches(matches)

    run_ids = unique_values(matches, "run_id")
    summaries = fetch_summaries(supabase, run_ids)
    neighbors = fetch_neighbor_chunks(supabase, matches) if INCLUDE_NEIGHBOR_CHUNKS else []

    context = format_context(matches, neighbors, summaries)

    answer = None
    if ASK_OPENAI_AFTER_RETRIEVAL:
        print("\n=== OPENAI ANSWER USING RETRIEVED CONTEXT ONLY ===\n")
        answer = answer_with_openai(openai_client, QUESTION, context)
        print(answer)

    output = {
        "question": QUESTION,
        "matches": matches,
        "neighbors": neighbors,
        "summaries": summaries,
        "openai_answer": answer,
    }
    OUTPUT_RETRIEVAL_JSON.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved retrieval result to: {OUTPUT_RETRIEVAL_JSON}")


if __name__ == "__main__":
    main()
