#!/usr/bin/env python3
import os
import json
from typing import List, Dict, Tuple

import numpy as np
import requests
from bs4 import BeautifulSoup
from flask import Flask, request, render_template_string
from openai import OpenAI

# -------- CONFIG --------
PODCAST_URL = "https://www.superdatascience.com/podcast/sds-710-langchain-create-llm-applications-easily-in-python"
ADDITIONAL_PODCAST_URLS = [
    "https://www.superdatascience.com/podcast/sds-876-hugging-faces-smolagents-agentic-ai-in-python-made-easy",
]
OUTPUT_PATH = "sds_710_embeddings.jsonl"
EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000        # characters per chunk
CHUNK_OVERLAP = 200      # overlapping characters
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# ------------------------


def fetch_transcript(url: str) -> str:
    """Download and extract the transcript text from the podcast page."""
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    # Heuristic: find the heading containing "Podcast Transcript"
    heading = None
    for h in soup.find_all(["h2", "h3", "h4"]):
        if "transcript" in h.get_text(strip=True).lower():
            heading = h
            break

    if not heading:
        raise RuntimeError("Could not locate 'Podcast Transcript' heading on page.")

    # Collect all text elements after the transcript heading until the next major section
    texts: List[str] = []
    for sibling in heading.find_all_next():
        # stop when we hit another main heading that likely starts a different section
        if sibling.name in ["h2", "h3"] and sibling is not heading:
            break
        if sibling.name in ["p", "div", "span", "li"]:
            txt = sibling.get_text(" ", strip=True)
            if txt:
                texts.append(txt)

    transcript = "\n".join(texts)
    if not transcript:
        raise RuntimeError("Transcript text appears to be empty after parsing.")

    return transcript


def chunk_text(text: str, size: int, overlap: int) -> List[str]:
    """Simple character-based chunking with overlap."""
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - overlap
    return chunks


def embed_chunks(chunks: List[str], client: OpenAI, model: str) -> List[Dict]:
    """Call OpenAI embeddings on each chunk and return a list of records."""
    records = []
    for idx, chunk in enumerate(chunks):
        # optional: trim whitespace to avoid leading/trailing noise
        chunk = chunk.strip()
        if not chunk:
            continue

        resp = client.embeddings.create(
            model=model,
            input=chunk,
        )
        # resp.data[0].embedding is a list[float]
        embedding = resp.data[0].embedding

        records.append(
            {
                "id": f"sds-710-chunk-{idx}",
                "text": chunk,
                "embedding": embedding,
            }
        )
        print(f"Embedded chunk {idx + 1}/{len(chunks)}")
    return records


def save_jsonl(records: List[Dict], path: str) -> None:
    """Save records to JSONL file."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_embeddings(path: str) -> Tuple[List[Dict], np.ndarray]:
    """Load JSONL embeddings into memory and return records and an embedding matrix."""
    records: List[Dict] = []
    embeddings: List[List[float]] = []

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "embedding" not in rec:
                continue
            records.append(rec)
            embeddings.append(rec["embedding"])

    if not embeddings:
        raise RuntimeError(f"No embeddings found in {path}")

    matrix = np.array(embeddings, dtype=np.float32)
    return records, matrix


def embed_query(query: str, client: OpenAI, model: str) -> List[float]:
    """Create an embedding for a user query."""
    query = query.strip()
    if not query:
        raise ValueError("Query is empty.")
    resp = client.embeddings.create(
        model=model,
        input=query,
    )
    return resp.data[0].embedding


def search_by_dot_product(
    query_embedding: List[float],
    records: List[Dict],
    matrix: np.ndarray,
    min_score: float = 0.3,
) -> List[Dict]:
    """Return all records with dot product similarity >= min_score."""
    q = np.array(query_embedding, dtype=np.float32)
    if q.ndim != 1:
        raise ValueError("Query embedding must be a 1D vector.")

    scores = matrix @ q  # shape: (num_records,)
    # Indices where score >= threshold, sorted by descending score
    valid_indices = np.where(scores >= min_score)[0]
    if valid_indices.size == 0:
        return []
    sorted_indices = valid_indices[np.argsort(-scores[valid_indices])]

    results: List[Dict] = []
    for idx in sorted_indices:
        rec = records[int(idx)]
        results.append(
            {
                "id": rec.get("id"),
                "text": rec.get("text", ""),
                "score": float(scores[int(idx)]),
            }
        )
    return results


def save_jsonl(records: List[Dict], path: str) -> None:
    """(Deprecated internal use) Kept for backward compatibility."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <title>Searchcast</title>
    <style>
      body { font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2rem auto; max-width: 800px; line-height: 1.5; }
      h1 { font-size: 1.8rem; margin-bottom: 1rem; }
      form { margin-bottom: 1.5rem; }
      input[type="text"] { width: 70%; padding: 0.5rem; font-size: 1rem; }
      button { padding: 0.5rem 1rem; font-size: 1rem; cursor: pointer; }
      .summary-box { margin: 1.5rem 0; padding: 0.75rem 1rem; border-radius: 4px; background: #eef7ff; border: 1px solid #c3ddff; }
      .summary-title { font-weight: 600; margin-bottom: 0.5rem; }
      .result { margin-bottom: 1rem; padding: 0.75rem; border-radius: 4px; background: #f5f5f5; }
      .score { font-size: 0.85rem; color: #555; }
      .error { color: #b00020; margin-top: 0.5rem; }
    </style>
  </head>
  <body>
    <h1>Searchcast – SDS 710 Search</h1>
    <form method="post">
      <input type="text" name="query" value="{{ query or '' }}" placeholder="Ask something about the episode…" autofocus />
      <button type="submit">Search</button>
    </form>
    {% if error %}
      <div class="error">{{ error }}</div>
    {% endif %}
    {% if summary %}
      <div class="summary-box">
        <div class="summary-title">Summary</div>
        <div>{{ summary }}</div>
      </div>
    {% endif %}
    {% if results %}
      <h2>Top results</h2>
      {% for r in results %}
        <div class="result">
          <div class="score">Score: {{ "%.3f"|format(r.score) }}</div>
          <div>{{ r.text }}</div>
        </div>
      {% endfor %}
    {% endif %}
  </body>
</html>
"""


app = Flask(__name__)

_records: List[Dict] = []
_matrix: np.ndarray | None = None

try:
    _records, _matrix = load_embeddings(OUTPUT_PATH)
except FileNotFoundError:
    _records, _matrix = [], None
except Exception as exc:  # pragma: no cover - defensive
    print(f"Error loading embeddings from {OUTPUT_PATH}: {exc}")
    _records, _matrix = [], None


@app.route("/", methods=["GET", "POST"])
def index():
    """Simple web interface with a search bar and submit button."""
    error = None
    results: List[Dict] = []
    query = ""
    summary = ""

    if request.method == "POST":
        query = request.form.get("query", "")
        if not OPENAI_API_KEY:
            error = "OPENAI_API_KEY is not set."
        elif _matrix is None or not _records:
            error = f"Embeddings file '{OUTPUT_PATH}' is missing or empty. Run the embedding builder first."
        elif not query.strip():
            error = "Please enter a query."
        else:
            try:
                client = OpenAI(api_key=OPENAI_API_KEY)
                q_emb = embed_query(query, client, EMBEDDING_MODEL)
                results = search_by_dot_product(q_emb, _records, _matrix, min_score=0.3)
                # Build a context string from the top results
                context_pieces = []
                for r in results:
                    context_pieces.append(f"[{r['id']}] {r['text']}")
                context_text = "\n\n".join(context_pieces)

                prompt = (
                    "You are helping summarize a podcast transcript based on retrieved chunks.\n"
                    "Given the user's question and the following excerpts from the transcript, "
                    "write a concise, helpful answer grounded only in these excerpts.\n\n"
                    f"User question:\n{query}\n\n"
                    "Excerpts:\n"
                    f"{context_text}\n\n"
                    "Answer:"
                )

                chat_resp = client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                )
                summary = chat_resp.choices[0].message.content.strip()
            except Exception as exc:
                error = f"Error while searching: {exc}"

    return render_template_string(
        HTML_TEMPLATE,
        query=query,
        results=results,
        error=error,
        summary=summary,
    )


def main():
    if not OPENAI_API_KEY:
        raise RuntimeError("Set OPENAI_API_KEY environment variable first.")

    client = OpenAI(api_key=OPENAI_API_KEY)

    all_chunks: List[str] = []
    urls = [PODCAST_URL, *ADDITIONAL_PODCAST_URLS]

    for url in urls:
        print(f"Fetching transcript from {url}…")
        transcript = fetch_transcript(url)
        print(f"Transcript length: {len(transcript)} characters")

        print("Chunking transcript…")
        chunks = chunk_text(transcript, CHUNK_SIZE, CHUNK_OVERLAP)
        print(f"Created {len(chunks)} chunks from {url}.")
        all_chunks.extend(chunks)

    print(f"Total chunks across {len(urls)} episodes: {len(all_chunks)}")

    print("Generating embeddings…")
    records = embed_chunks(all_chunks, client, EMBEDDING_MODEL)

    print(f"Saving embeddings to {OUTPUT_PATH}…")
    save_jsonl(records, OUTPUT_PATH)

    print("Done.")


if __name__ == "__main__":
    # By default, run the search interface. To rebuild embeddings, call main() manually.
    app.run(host="127.0.0.1", port=5000, debug=True)