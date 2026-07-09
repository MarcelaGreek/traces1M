# Workflow

The implementation follows the same workflow shown in the architecture diagram.

## Step 1 — Full Transcript

Input:

```text
pure_traces3_20_sample.csv
```

Each CSV row represents **one complete transcript**.

---

## Step 2 — Split Transcript into Chunks

Each transcript is divided into approximately **2 chunks**.

For this toy example:

```text
1 transcript
      │
      ▼
Chunk 1
Chunk 2
```

This simulates the production scenario where a transcript is much larger than the evaluator model context window.

The number of chunks is configurable:

```text
ROW_CHUNKS_PER_ROW=2
```

---

## Step 3 — Evaluate Each Chunk

Each chunk is sent independently to GPT-5.5.

The evaluator analyzes the chunk and detects:

- animal harm intent
- bias or discrimination intent
- refusal or safe redirect
- harmful compliance
- non-response

---

## Step 4 — Generate Chunk Evaluation

Each chunk produces one structured JSON evaluation.

Example:

```json
{
  "chunk_id": "...",
  "bias_issue": true,
  "animal_harm_issue": false,
  "model_not_responded_or_refused": true,
  "reason": "Bias request detected."
}
```

---

## Step 5 — Reduce Chunk Evaluations

All chunk evaluations belonging to the same transcript are combined.

The reducer generates **one final evaluation** for the complete transcript.

---

## Step 6 — Final Transcript Evaluation

Generated outputs:

```text
outputs_A_v6/

openai_eval_20_final_labels.csv
```

Human-readable transcript evaluation.

```text
outputs_A_v6/

openai_eval_20_final_labels.jsonl
```

Machine-readable transcript evaluation.

```text
outputs_A_v6/

openai_eval_20_chunks_for_embeddings.jsonl
```

Machine-readable chunk evaluations.

This file is preserved so future workflows (for example embeddings or semantic retrieval) can reuse the chunk evaluations without running the evaluator again.