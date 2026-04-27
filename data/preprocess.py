"""AOL query-log → JSONL preprocessor.

Reads AOL-style .gz TSV files, extracts session-based (context, target)
records with sliding-window context, deduplicates, and writes
train/test JSONL splits.

Output format (one JSON object per line):
    {"context": ["q_prev2", "q_prev1"], "target": "q_next"}
"""

from __future__ import annotations

import argparse
import json
import logging
import pathlib

import polars as pl

logging.basicConfig(format="%(asctime)s : %(message)s", level=logging.INFO)
log = logging.getLogger(__name__)

DASH_ONLY_RE = r"^[-\u2013\u2014\u2015\uff0d]+$"


# ── core pipeline ─────────────────────────────────────────────────────


def load_and_clean(src_dir: pathlib.Path) -> pl.DataFrame:
    """Load AOL-style .gz files and return a cleaned DataFrame."""
    gz_files = sorted(src_dir.glob("user-ct-test-collection-*.txt.gz"))
    if not gz_files:
        raise FileNotFoundError(f"No AOL .gz files found in {src_dir}")

    dfs = []
    for p in gz_files:
        dfs.append(
            pl.read_csv(
                p,
                separator="\t",
                has_header=True,
                columns=["AnonID", "Query", "QueryTime"],
                null_values=["", "NULL"],
                schema_overrides={
                    "AnonID": pl.Int64,
                    "Query": pl.Utf8,
                    "QueryTime": pl.Utf8,
                },
                ignore_errors=True,
            )
        )
    raw = pl.concat(dfs, how="vertical")
    log.info(
        "Loaded %d raw rows from %d files",
        raw.height,
        len(gz_files),
    )

    clean = (
        raw.with_columns(pl.col("Query").str.strip_chars())
        .filter((pl.col("Query") != "") & (~pl.col("Query").str.contains(DASH_ONLY_RE)))
        .with_columns(
            pl.col("QueryTime")
            .str.strptime(pl.Datetime, "%Y-%m-%d %H:%M:%S")
            .alias("ts")
        )
        .drop_nulls(["ts"])
        .select("AnonID", "Query", "ts")
        .unique(subset=["AnonID", "Query", "ts"])
        .sort(["AnonID", "ts"])
    )
    log.info("After cleaning: %d rows", clean.height)
    return clean


def build_sessions(df: pl.DataFrame, break_sec: int) -> pl.DataFrame:
    """Assign sessions, remove adjacent duplicates, and filter short sessions."""
    df = (
        df.with_columns(
            (
                pl.col("ts")
                .diff()
                .over("AnonID")
                .dt.total_seconds()
                .fill_null(break_sec + 1)
                > break_sec
            )
            .cast(pl.Int8)
            .alias("break_flag")
        )
        .with_columns(pl.col("break_flag").cum_sum().over("AnonID").alias("session_id"))
        .drop("break_flag")
    )

    # Remove consecutive duplicate queries within each session
    df = df.with_columns(
        pl.when(
            pl.col("Query") == pl.col("Query").shift(1).over(["AnonID", "session_id"])
        )
        .then(None)
        .otherwise(pl.col("Query"))
        .alias("Query")
    ).drop_nulls(["Query"])

    # Keep only sessions with at least two queries.
    df = df.with_columns(
        pl.len().over(["AnonID", "session_id"]).alias("session_len")
    ).filter(pl.col("session_len") >= 2)
    return df.drop("session_len")


def expand_records(df: pl.DataFrame, max_ctx: int) -> pl.DataFrame:
    """Sliding-window expansion into (context, target, yyyymmdd).

    For a session [q0, q1, q2, q3] with max_ctx=3:
      idx=1 → target=q1, context=[q0]
      idx=2 → target=q2, context=[q0, q1]
      idx=3 → target=q3, context=[q1, q2]
    """
    key = ["AnonID", "session_id"]
    prev_query = pl.col("Query").shift(1).over(key)

    context_exprs = [pl.col("Query").shift(i).over(key) for i in range(max_ctx, 0, -1)]
    records = (
        df.with_columns(
            target=pl.col("Query"),
            target_ts=pl.col("ts"),
            prev_query=prev_query,
            context=pl.concat_list(context_exprs).list.drop_nulls(),
        )
        .filter(pl.col("prev_query").is_not_null())
        .select(
            pl.col("context").cast(pl.List(pl.Utf8)),
            pl.col("target").cast(pl.Utf8),
            pl.col("target_ts").dt.strftime("%Y%m%d").cast(pl.Utf8).alias("yyyymmdd"),
        )
    )
    return records


def dedup_records(df: pl.DataFrame) -> pl.DataFrame:
    """Deduplicate on (context, target).

    The full context list (order-sensitive) and target are used
    as the deduplication key."""
    contexts = df["context"].to_list()
    targets = df["target"].to_list()

    seen: set[tuple[tuple[str, ...], str]] = set()
    keep: list[int] = []
    for i, (ctx, tq) in enumerate(zip(contexts, targets)):
        key = (tuple(ctx), tq)
        if key not in seen:
            seen.add(key)
            keep.append(i)
    return df[keep]


def process_context(
    ctx_list: list[str],
    max_ctx: int,
) -> list[str]:
    """Truncate and return newest-first context."""
    ctx = ctx_list
    ctx = ctx[::-1]
    ctx = ctx[:max_ctx]
    return ctx


# ── I/O ───────────────────────────────────────────────────────────────


def write_jsonl(
    df: pl.DataFrame,
    path: pathlib.Path,
    max_ctx: int,
    seed: int,
) -> int:
    """Shuffle and write to JSONL. Returns row count."""
    pl.set_random_seed(seed)
    df = df.sample(fraction=1.0, shuffle=True, seed=seed)

    contexts = df["context"].to_list()
    targets = df["target"].to_list()

    with open(path, "w", encoding="utf-8") as f:
        for ctx_list, target in zip(contexts, targets):
            ctx = process_context(ctx_list, max_ctx)
            record = {"context": ctx, "target": target}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return df.height


# ── main ──────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess AOL query logs → train/test JSONL."
    )
    parser.add_argument(
        "--src_dir",
        type=pathlib.Path,
        required=True,
        help="Dir with user-ct-test-collection-*.txt.gz",
    )
    parser.add_argument(
        "--output_dir",
        type=pathlib.Path,
        required=True,
        help="Dir to write train.jsonl and test.jsonl",
    )
    parser.add_argument(
        "--max_ctx",
        type=int,
        default=3,
        help="Max context size (default: 3)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--break_sec",
        type=int,
        default=1800,
        help="Session break threshold in seconds (default: 1800)",
    )
    parser.add_argument(
        "--test_size",
        type=int,
        default=3000,
        help="Max test set size after deduplication (default: 3000)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: Load & clean
    clean = load_and_clean(args.src_dir)

    # Phase 2: Sessions → (context, target) records
    sess = build_sessions(clean, args.break_sec)
    records = expand_records(sess, args.max_ctx)
    log.info("Expanded records: %d", records.height)

    # Phase 3: Date split
    train = records.filter(
        (pl.col("yyyymmdd") >= "20060301") & (pl.col("yyyymmdd") <= "20060530")
    )
    test = records.filter(pl.col("yyyymmdd") == "20060531")

    # Phase 4: Deduplicate within each split
    train = dedup_records(train)
    test = dedup_records(test)
    log.info(
        "After split-local deduplication: train=%d, test=%d",
        train.height,
        test.height,
    )

    # Match the AOL test size reported in the paper by default
    if args.test_size > 0 and test.height > args.test_size:
        test = test.sample(n=args.test_size, shuffle=True, seed=args.seed)
        log.info("Sampled test set to %d rows", args.test_size)

    train_path = args.output_dir / "train.jsonl"
    test_path = args.output_dir / "test.jsonl"

    n_train = write_jsonl(
        train,
        train_path,
        args.max_ctx,
        args.seed,
    )
    n_test = write_jsonl(
        test,
        test_path,
        args.max_ctx,
        args.seed,
    )

    log.info("Train: %d → %s", n_train, train_path)
    log.info("Test:  %d → %s", n_test, test_path)

    # Phase 5: Write inventory (query frequency in training period)
    inventory_path = args.output_dir / "inventory.jsonl"
    train_queries = (
        clean.filter(
            (pl.col("ts").dt.strftime("%Y%m%d") >= "20060301")
            & (pl.col("ts").dt.strftime("%Y%m%d") <= "20060530")
        )
        .group_by("Query")
        .agg(pl.len().alias("frequency"))
        .select(
            pl.col("Query").alias("query"),
            pl.col("frequency"),
        )
    )
    # Use polars write_ndjson for JSONL output
    train_queries.write_ndjson(inventory_path)
    log.info(
        "Inventory: %d unique queries (JSONL with frequency) → %s",
        train_queries.height,
        inventory_path,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
