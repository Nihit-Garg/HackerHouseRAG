"""
benchmark.py — Latency analytics for the RAG pipeline.

Runs N queries through retrieval-only and full pipeline stages.
Computes P50 / P75 / P90 / P95 / P99 / P100 percentiles.
Outputs a clean markdown-formatted table — no charts needed.

Usage:
  python scripts/benchmark.py                    # 50 queries, retrieval only
  python scripts/benchmark.py --full             # includes LLM generation
  python scripts/benchmark.py --n 100            # custom query count
  python scripts/benchmark.py --output bench.md  # save table to file
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("benchmark")

# ── Benchmark queries — diverse mix: in-corpus, off-corpus, ambiguous ─────────
BENCHMARK_QUERIES = [
    # In-corpus — likely grounded
    "What was the impact of the Manhattan Project?",
    "How does FAISS similarity search work?",
    "What is the role of nitrogen in plant growth?",
    "When did World War II end?",
    "What causes hurricanes?",
    "How is steel manufactured?",
    "What is the speed of light?",
    "Who invented the telephone?",
    "What is photosynthesis?",
    "How does a vaccine work?",
    "What is the largest ocean on Earth?",
    "How do black holes form?",
    "What is the capital of France?",
    "What are the symptoms of diabetes?",
    "How is glass made?",
    "What is quantum mechanics?",
    "What causes earthquakes?",
    "How does the immune system work?",
    "What is DNA replication?",
    "What is machine learning?",
    # More specific
    "What was the Dust Bowl of the 1930s?",
    "What is the Hubble Space Telescope?",
    "How do solar panels generate electricity?",
    "What is the structure of the atom?",
    "How does natural selection work?",
    "What is the greenhouse effect?",
    "How are mountains formed?",
    "What is the ozone layer?",
    "What is a supernova?",
    "How does Wi-Fi work?",
    # Ambiguous / partial
    "impact of atomic bomb",
    "climate change effects",
    "deep sea creatures",
    "ancient civilizations",
    "renewable energy sources",
    "gene editing technology",
    "artificial intelligence applications",
    "space exploration history",
    "ocean currents patterns",
    "volcanic eruptions causes",
    # Off-corpus — should trigger low retrieval / off-topic guard
    "best pizza recipe in Naples",
    "how to win at chess",
    "stock market predictions 2025",
    "latest fashion trends",
    "movie recommendations",
    "how to bake sourdough bread",
    "cryptocurrency investment advice",
    "top 10 tourist destinations",
    "social media marketing tips",
    "video game release dates",
]


def percentile(data: list[float], p: float) -> float:
    """Compute p-th percentile (0-100) of data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = (p / 100) * (len(sorted_data) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sorted_data) - 1)
    return sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo])


def format_table(
    retrieval_times: list[float],
    full_times: list[float] | None = None,
) -> str:
    """Format latency results as a markdown table."""
    ps = [50, 75, 90, 95, 99, 100]

    lines = []
    lines.append("## Latency Benchmark Results\n")
    lines.append(f"**Queries run:** {len(retrieval_times)}")
    lines.append(f"**Embedding model:** BAAI/bge-small-en-v1.5")
    lines.append(f"**Retrieval:** FAISS (dense) + BM25, RRF fusion\n")

    # Retrieval table
    lines.append("### Retrieval-Only Latency (ms)")
    lines.append("| Percentile | Latency (ms) |")
    lines.append("|-----------|-------------|")
    for p in ps:
        lines.append(f"| P{p:<8} | {percentile(retrieval_times, p):>10.1f} |")

    lines.append(f"\n**Mean:** {mean(retrieval_times):.1f}ms")
    if len(retrieval_times) > 1:
        lines.append(f"**Std Dev:** {stdev(retrieval_times):.1f}ms")
    lines.append(f"**Min:** {min(retrieval_times):.1f}ms  |  **Max:** {max(retrieval_times):.1f}ms")

    # Full pipeline table (optional)
    if full_times:
        lines.append("\n### Full Pipeline Latency (ms) [STT bypassed, text input]")
        lines.append("| Percentile | Latency (ms) |")
        lines.append("|-----------|-------------|")
        for p in ps:
            lines.append(f"| P{p:<8} | {percentile(full_times, p):>10.1f} |")

        lines.append(f"\n**Mean:** {mean(full_times):.1f}ms")
        if len(full_times) > 1:
            lines.append(f"**Std Dev:** {stdev(full_times):.1f}ms")
        lines.append(f"**Min:** {min(full_times):.1f}ms  |  **Max:** {max(full_times):.1f}ms")

    return "\n".join(lines)


def run_retrieval_benchmark(retriever, queries: list[str]) -> list[float]:
    """Run retrieval-only benchmark. Returns list of elapsed_ms per query."""
    times = []
    for i, q in enumerate(queries, 1):
        t0 = time.perf_counter()
        try:
            retriever.retrieve(q)
        except Exception as exc:
            logger.warning("Query %d failed: %s", i, exc)
            continue
        elapsed = (time.perf_counter() - t0) * 1000
        times.append(elapsed)
        if i % 10 == 0:
            print(f"  [{i}/{len(queries)}] last={elapsed:.1f}ms  running_mean={mean(times):.1f}ms",
                  file=sys.stderr)
    return times


def run_full_benchmark(pipeline, queries: list[str]) -> list[float]:
    """Run full pipeline benchmark (no LLM — mocked via retrieval threshold guard).
    Uses queries that should pass retrieval threshold. Returns total_ms per query."""
    times = []
    for i, q in enumerate(queries[:20], 1):  # limit full runs — LLM is slow
        result = pipeline.answer(text_query=q)
        elapsed = result["stage_timings"]["total_ms"]
        times.append(elapsed)
        guard = result.get("guardrail_triggered") or "none"
        print(f"  [{i}/20] {elapsed:.0f}ms | guard={guard} | {q[:40]}…", file=sys.stderr)
    return times


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Latency benchmark for the RAG pipeline.")
    parser.add_argument("--n", type=int, default=50, help="Number of retrieval queries (default: 50)")
    parser.add_argument("--full", action="store_true", help="Also run full pipeline (LLM) benchmark on 20 queries")
    parser.add_argument("--output", type=str, default=None, help="Save results to file (default: print to stdout)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("⏳ Loading indexes…", file=sys.stderr)
    from embeddings import get_embedding_model
    from retrieval import HybridRetriever
    from vectorstore import load_bm25_index, load_faiss_index

    faiss_index, chunks = load_faiss_index()
    bm25_index = load_bm25_index()
    emb_model = get_embedding_model()
    retriever = HybridRetriever(faiss_index, bm25_index, chunks, emb_model)
    print("✓ Ready. Starting benchmark…\n", file=sys.stderr)

    queries = BENCHMARK_QUERIES[:args.n]

    # ── Retrieval benchmark ───────────────────────────────────────────────────
    print(f"Running retrieval benchmark ({len(queries)} queries)…", file=sys.stderr)
    retrieval_times = run_retrieval_benchmark(retriever, queries)

    # ── Full pipeline benchmark (optional) ───────────────────────────────────
    full_times = None
    if args.full:
        print("\nRunning full pipeline benchmark (20 queries, includes guardrails + LLM)…", file=sys.stderr)
        from graph import RAGPipeline
        pipeline = RAGPipeline(retriever, emb_model)
        full_times = run_full_benchmark(pipeline, queries)

    # ── Format and output results ─────────────────────────────────────────────
    table = format_table(retrieval_times, full_times)

    if args.output:
        Path(args.output).write_text(table, encoding="utf-8")
        print(f"\n✓ Results saved to: {args.output}", file=sys.stderr)
    else:
        print("\n" + "=" * 60)
        print(table)
        print("=" * 60)

    # Also save JSON for programmatic use
    results_json = {
        "n_queries": len(retrieval_times),
        "retrieval_ms": {
            "p50": percentile(retrieval_times, 50),
            "p75": percentile(retrieval_times, 75),
            "p90": percentile(retrieval_times, 90),
            "p95": percentile(retrieval_times, 95),
            "p99": percentile(retrieval_times, 99),
            "p100": percentile(retrieval_times, 100),
            "mean": mean(retrieval_times),
        },
    }
    print("\n📊 JSON summary:", file=sys.stderr)
    print(json.dumps(results_json, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
