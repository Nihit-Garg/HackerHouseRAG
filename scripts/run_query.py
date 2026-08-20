"""
run_query.py — CLI query runner for the RAG pipeline.

Accepts either a text query or an audio file, runs the full pipeline,
and prints the structured result as formatted JSON.

Usage:
  python scripts/run_query.py --text "What is machine learning?"
  python scripts/run_query.py --audio path/to/query.wav
  python scripts/run_query.py --text "..." --top-k 10
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

logging.basicConfig(
    level=logging.WARNING,  # suppress INFO noise during CLI use
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger("run_query")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a query through the voice RAG pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_query.py --text "What is deep learning?"
  python scripts/run_query.py --audio query.wav
  python scripts/run_query.py --text "FAISS library" --top-k 10 --verbose
        """,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", "-t", type=str, help="Text query (bypasses STT).")
    group.add_argument("--audio", "-a", type=str, help="Path to audio file for STT.")

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Number of chunks to retrieve (default: from config).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging (INFO level).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        default=True,
        help="Pretty-print JSON output (default: True).",
    )
    return parser.parse_args()


def _load_pipeline(top_k: int | None):
    """Load indexes and build the RAGPipeline."""
    from config import TOP_K
    from embeddings import get_embedding_model
    from graph import RAGPipeline
    from retrieval import HybridRetriever
    from vectorstore import load_bm25_index, load_faiss_index

    k = top_k if top_k is not None else TOP_K

    try:
        faiss_index, chunks = load_faiss_index()
        bm25_index = load_bm25_index()
    except FileNotFoundError as exc:
        print(f"\n❌ Error: {exc}", file=sys.stderr)
        print("Run: python scripts/build_index.py", file=sys.stderr)
        sys.exit(1)

    emb_model = get_embedding_model()
    retriever = HybridRetriever(faiss_index, bm25_index, chunks, emb_model)
    return RAGPipeline(retriever, emb_model, top_k=k)


def _print_result(result: dict, pretty: bool) -> None:
    indent = 2 if pretty else None
    print(json.dumps(result, indent=indent, ensure_ascii=False))


def _print_summary(result: dict) -> None:
    """Print a human-readable summary before the JSON blob."""
    print("\n" + "=" * 60)
    print(f"📝 Query:   {result.get('transcribed_text', result.get('query', ''))}")
    guard = result.get("guardrail_triggered")
    if guard:
        print(f"🛡  Guardrail: {guard.upper()} — {result.get('guardrail_detail', {}).get('reason', '')}")
    print(f"💬 Answer:  {result.get('answer', '')}")
    timings = result.get("stage_timings", {})
    print(
        f"⏱  Timings: STT={timings.get('stt_ms', 0):.0f}ms  "
        f"Retrieval={timings.get('retrieval_ms', 0):.0f}ms  "
        f"LLM={timings.get('generation_ms', 0):.0f}ms  "
        f"Total={timings.get('total_ms', 0):.0f}ms"
    )
    errors = result.get("errors", [])
    if errors:
        print(f"⚠️  Errors: {errors}")
    print("=" * 60)
    print("\n📄 Full JSON output:\n")


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.INFO)

    print("⏳ Loading pipeline…", file=sys.stderr)
    pipeline = _load_pipeline(args.top_k)
    print("✓ Pipeline ready.\n", file=sys.stderr)

    if args.text:
        result = pipeline.answer(text_query=args.text)
    else:
        audio_path = str(Path(args.audio).resolve())
        result = pipeline.answer(audio_path=audio_path)

    _print_summary(result)
    _print_result(result, pretty=args.pretty)

    # Exit with non-zero code if pipeline had errors
    if result.get("errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
