# Design notes

## Why Part A changed

Option B needs retrievable chunks. Final labels alone are not enough.

Correct Part A outputs:

```text
openai_eval_20_final_labels.csv        -> dashboard/statistics
openai_eval_20_final_labels.jsonl      -> row-level machine-readable final labels
openai_eval_20_chunks_for_embeddings.jsonl -> chunk-level text + chunk labels for embeddings
```

## Summary fields

At chunk level:

```text
chunk_evaluation.evidence
```

This is the safety summary of the chunk.

At row level:

```text
reason
```

This is the final safety summary of the whole row.

Not summary:

```text
safety_evaluation
```

This is only the final binary class: `safe` or `not_safe`.

## Supabase

The SQL intentionally has no RLS and no policies because this is a local/toy 20-line test.
Do not use that SQL as-is for production.
