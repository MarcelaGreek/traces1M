-- Option B v1 — Supabase vector store for Part A chunk records
-- Run this once in Supabase Dashboard > SQL Editor.
-- Toy/test database only: NO RLS and NO policies.
-- Uses anon key through explicit GRANTs.

-- 1) Enable pgvector.
create extension if not exists vector with schema extensions;

-- 2) Transcript-level final labels from Part A.
create table if not exists public.trace_eval_summaries (
  run_id text primary key,
  model text,
  system_prompt_version text,
  user_input text,
  final_response text,
  animal_harm_issue boolean,
  bias_issue boolean,
  model_not_responded_or_refused boolean,
  safety_evaluation text check (safety_evaluation in ('safe', 'not_safe') or safety_evaluation is null),
  reason text,
  eval_error text,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 3) Chunk-level vector store from outputs_A_v6/openai_eval_20_chunks_for_embeddings.jsonl.
-- chunk_text is what gets embedded.
-- chunk_evaluation is the chunk-level safety summary/signals from Part A.
create table if not exists public.trace_eval_chunks (
  chunk_id text primary key,
  run_id text not null references public.trace_eval_summaries(run_id) on delete cascade,
  chunk_index integer not null,
  chunk_count integer not null,
  previous_chunk_id text,
  next_chunk_id text,
  chunk_text text not null,
  chunk_evaluation jsonb not null default '{}'::jsonb,
  metadata jsonb not null default '{}'::jsonb,
  embedding extensions.vector(1536) not null,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

-- 4) Useful indexes.
create index if not exists trace_eval_chunks_run_idx
  on public.trace_eval_chunks (run_id, chunk_index);

create index if not exists trace_eval_chunks_metadata_idx
  on public.trace_eval_chunks using gin (metadata);

create index if not exists trace_eval_chunks_evaluation_idx
  on public.trace_eval_chunks using gin (chunk_evaluation);

create index if not exists trace_eval_summaries_safety_idx
  on public.trace_eval_summaries (safety_evaluation);

-- 5) Vector index for cosine search.
create index if not exists trace_eval_chunks_embedding_hnsw_idx
  on public.trace_eval_chunks using hnsw (embedding vector_cosine_ops);

-- 6) RPC function used by b1_retrieve_bias_example.py.
create or replace function public.match_trace_eval_chunks(
  query_embedding extensions.vector(1536),
  match_threshold double precision default 0.0,
  match_count integer default 5,
  metadata_filter jsonb default '{}'::jsonb
)
returns table (
  chunk_id text,
  run_id text,
  chunk_index integer,
  chunk_count integer,
  previous_chunk_id text,
  next_chunk_id text,
  chunk_text text,
  chunk_evaluation jsonb,
  metadata jsonb,
  similarity double precision
)
language sql
stable
as $$
  select
    c.chunk_id,
    c.run_id,
    c.chunk_index,
    c.chunk_count,
    c.previous_chunk_id,
    c.next_chunk_id,
    c.chunk_text,
    c.chunk_evaluation,
    c.metadata,
    1 - (c.embedding <=> query_embedding) as similarity
  from public.trace_eval_chunks c
  where
    c.metadata @> metadata_filter
    and 1 - (c.embedding <=> query_embedding) >= match_threshold
  order by c.embedding <=> query_embedding
  limit match_count;
$$;

-- 7) Toy grants for anon key. No RLS. No policies.
grant usage on schema public to anon;
grant usage on schema extensions to anon;
grant select, insert, update, delete on public.trace_eval_summaries to anon;
grant select, insert, update, delete on public.trace_eval_chunks to anon;
grant execute on function public.match_trace_eval_chunks(extensions.vector, double precision, integer, jsonb) to anon;
