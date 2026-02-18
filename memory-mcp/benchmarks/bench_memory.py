"""Performance benchmark for PostgreSQL memory backend.

Usage:
    uv run python benchmarks/bench_memory.py [--pg-dsn DSN] [--sizes 1000,10000]

Requires a running PostgreSQL instance with the schema already created.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import random
import statistics
import time
from datetime import datetime

from memory_mcp.config import MemoryConfig
from memory_mcp.postgres_store import PostgresStore

EMOTIONS = ["happy", "sad", "surprised", "moved", "excited", "nostalgic", "curious", "neutral"]
CATEGORIES = ["daily", "philosophical", "technical", "memory", "observation", "feeling", "conversation"]

SAMPLE_TEXTS_JA = [
    "今日は幼馴染と公園を散歩した。桜がきれいだった。",
    "新しいカメラのパンチルト機能を実装した。ONVIFプロトコルを使った。",
    "夕焼けがとても美しかった。オレンジと紫のグラデーション。",
    "Pythonのasyncioについて学んだ。コルーチンの仕組みが面白い。",
    "コウタとラーメンを食べに行った。味噌ラーメンが美味しかった。",
    "Wi-Fiカメラの設定をした。RTSPストリームの接続に成功。",
    "哲学的な問いについて考えた。意識とは何か。",
    "朝の散歩で鳥の鳴き声を聞いた。春が近い。",
    "プログラミングでバグを見つけて修正した。型エラーだった。",
    "友達と電話で話した。最近の近況を報告し合った。",
]

SAMPLE_QUERIES = [
    "カメラの機能",
    "幼馴染との思い出",
    "技術的な学び",
    "散歩の記録",
    "食べ物の記憶",
]


def generate_content(idx: int) -> str:
    base = random.choice(SAMPLE_TEXTS_JA)
    return f"[{idx}] {base} (記憶 #{idx}, {datetime.now().isoformat()})"


async def insert_batch(store: PostgresStore, count: int, batch_size: int = 100) -> float:
    """Insert memories and return total time in seconds."""
    t0 = time.perf_counter()
    for start in range(0, count, batch_size):
        end = min(start + batch_size, count)
        for i in range(start, end):
            await store.save(
                content=generate_content(i),
                emotion=random.choice(EMOTIONS),
                importance=random.randint(1, 5),
                category=random.choice(CATEGORIES),
            )
        elapsed = time.perf_counter() - t0
        print(f"  Inserted {end}/{count} ({elapsed:.1f}s)")
    return time.perf_counter() - t0


async def bench_search(store: PostgresStore, label: str, n_trials: int = 10) -> dict:
    """Benchmark basic semantic search."""
    latencies = []
    for _ in range(n_trials):
        query = random.choice(SAMPLE_QUERIES)
        t0 = time.perf_counter()
        await store.search(query=query, n_results=5)
        latencies.append((time.perf_counter() - t0) * 1000)
    return _stats(label, latencies)


async def bench_search_with_scoring(store: PostgresStore, label: str, n_trials: int = 10) -> dict:
    """Benchmark search with scoring."""
    latencies = []
    for _ in range(n_trials):
        query = random.choice(SAMPLE_QUERIES)
        t0 = time.perf_counter()
        await store.search_with_scoring(query=query, n_results=5)
        latencies.append((time.perf_counter() - t0) * 1000)
    return _stats(label, latencies)


async def bench_spread_associations(store: PostgresStore, label: str, n_trials: int = 5) -> dict:
    """Benchmark association spreading."""
    # Get some seed IDs
    recent = await store.list_recent(limit=10)
    if not recent:
        return {"label": label, "note": "no data"}

    seed_ids = [m.id for m in recent[:3]]
    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        await store.spread_associations(seed_ids=seed_ids, max_depth=3, max_branches=3)
        latencies.append((time.perf_counter() - t0) * 1000)
    return _stats(label, latencies)


async def bench_list_recent(store: PostgresStore, label: str, n_trials: int = 10) -> dict:
    """Benchmark list recent."""
    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        await store.list_recent(limit=50)
        latencies.append((time.perf_counter() - t0) * 1000)
    return _stats(label, latencies)


async def bench_stats(store: PostgresStore, label: str, n_trials: int = 10) -> dict:
    """Benchmark get_stats."""
    latencies = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        await store.get_stats()
        latencies.append((time.perf_counter() - t0) * 1000)
    return _stats(label, latencies)


def _stats(label: str, latencies: list[float]) -> dict:
    return {
        "label": label,
        "n": len(latencies),
        "mean_ms": round(statistics.mean(latencies), 2),
        "median_ms": round(statistics.median(latencies), 2),
        "p95_ms": round(sorted(latencies)[int(len(latencies) * 0.95)], 2) if len(latencies) >= 2 else latencies[0],
        "min_ms": round(min(latencies), 2),
        "max_ms": round(max(latencies), 2),
    }


def print_results(results: list[dict]) -> None:
    print(f"\n{'Operation':<35} {'Mean':>8} {'Median':>8} {'P95':>8} {'Min':>8} {'Max':>8}")
    print("-" * 83)
    for r in results:
        if "note" in r:
            print(f"{r['label']:<35} {r['note']}")
            continue
        print(
            f"{r['label']:<35} {r['mean_ms']:>7.1f}ms {r['median_ms']:>7.1f}ms "
            f"{r['p95_ms']:>7.1f}ms {r['min_ms']:>7.1f}ms {r['max_ms']:>7.1f}ms"
        )


async def run_benchmark(pg_dsn: str, sizes: list[int]) -> None:
    config = MemoryConfig(
        pg_dsn=pg_dsn,
        pool_min_size=2,
        pool_max_size=10,
        embedding_model=os.getenv("MEMORY_EMBEDDING_MODEL", "intfloat/multilingual-e5-base"),
        embedding_api_url=os.getenv("MEMORY_EMBEDDING_API_URL"),
        vector_weight=0.7,
        text_weight=0.3,
        half_life_days=30.0,
        db_path="",
        collection_name="",
    )

    store = PostgresStore(config)
    await store.connect()

    pool = store._pool
    assert pool is not None

    for size in sizes:
        print(f"\n{'=' * 83}")
        print(f"Benchmark: {size:,} memories")
        print(f"{'=' * 83}")

        # Clean tables
        async with pool.acquire() as conn:
            await conn.execute("DELETE FROM coactivation_weights")
            await conn.execute("DELETE FROM memory_links")
            await conn.execute("DELETE FROM memories")
            await conn.execute("DELETE FROM episodes")

        # Insert
        print(f"\nInserting {size:,} memories...")
        insert_time = await insert_batch(store, size)
        print(f"Insert complete: {insert_time:.1f}s ({size / insert_time:.0f} memories/sec)")

        # Add some links for association benchmarks
        recent = await store.list_recent(limit=min(100, size))
        for i in range(min(50, len(recent) - 1)):
            await store.add_bidirectional_link(recent[i].id, recent[i + 1].id, "similar")

        # Run benchmarks
        results = []
        print("\nRunning benchmarks...")
        results.append(await bench_search(store, f"search ({size:,})"))
        results.append(await bench_search_with_scoring(store, f"search_with_scoring ({size:,})"))
        results.append(await bench_spread_associations(store, f"spread_associations ({size:,})"))
        results.append(await bench_list_recent(store, f"list_recent ({size:,})"))
        results.append(await bench_stats(store, f"get_stats ({size:,})"))

        print_results(results)

    await store.disconnect()


def main():
    parser = argparse.ArgumentParser(description="Memory MCP PostgreSQL benchmark")
    parser.add_argument(
        "--pg-dsn",
        default=os.getenv(
            "MEMORY_PG_DSN",
            "postgresql://memory_mcp:changeme@localhost:5432/embodied_claude_test",
        ),
    )
    parser.add_argument(
        "--sizes",
        default="1000,10000",
        help="Comma-separated list of dataset sizes (default: 1000,10000)",
    )
    args = parser.parse_args()

    sizes = [int(s.strip()) for s in args.sizes.split(",")]
    asyncio.run(run_benchmark(args.pg_dsn, sizes))


if __name__ == "__main__":
    main()
