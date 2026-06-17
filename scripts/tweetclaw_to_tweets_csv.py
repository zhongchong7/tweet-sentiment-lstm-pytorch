#!/usr/bin/env python3
"""Convert TweetClaw exports into the notebook's Tweets.csv schema."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any


TEXT_KEYS = (
    "text",
    "tweetText",
    "tweet_text",
    "replyText",
    "reply_text",
    "full_text",
    "fullText",
    "tweet",
    "Tweet",
    "content",
    "body",
    "message",
)
SENTIMENT_KEYS = (
    "sentiment",
    "Sentiment",
    "label",
    "sentiment_label",
    "prediction",
    "true_label",
)
WRAPPER_KEYS = ("data", "tweets", "results", "items", "records", "rows")
NESTED_TEXT_PATHS = (
    ("legacy", "full_text"),
    ("tweet", "text"),
    ("tweet", "full_text"),
    ("tweet", "fullText"),
)
LABELS = {
    "positive": "positive",
    "pos": "positive",
    "label_2": "positive",
    "2": "positive",
    "negative": "negative",
    "neg": "negative",
    "label_0": "negative",
    "0": "negative",
    "neutral": "neutral",
    "neu": "neutral",
    "label_1": "neutral",
    "1": "neutral",
}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert TweetClaw JSON, JSONL, or CSV exports to Tweets.csv."
    )
    parser.add_argument("input", type=Path, help="TweetClaw export path")
    parser.add_argument(
        "output",
        type=Path,
        nargs="?",
        default=Path("Tweets.csv"),
        help="Output CSV path. Defaults to Tweets.csv.",
    )
    parser.add_argument(
        "--default-sentiment",
        choices=("negative", "neutral", "positive"),
        default="neutral",
        help="Sentiment value for rows without an export sentiment field.",
    )
    args = parser.parse_args()

    rows = [
        {"text": text, "sentiment": sentiment}
        for record in read_records(args.input)
        if (text := extract_text(record))
        for sentiment in [extract_sentiment(record, args.default_sentiment)]
    ]

    if not rows:
        print("No tweet text found in the input export.", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("text", "sentiment"))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")
    return 0


def read_records(path: Path) -> Iterator[Mapping[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            yield from csv.DictReader(handle)
        return

    raw = path.read_text(encoding="utf-8-sig")
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        yield from read_jsonl(raw)
        return

    yield from unwrap_records(payload)


def read_jsonl(raw: str) -> Iterator[Mapping[str, Any]]:
    for line_number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            yield from unwrap_records(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL on line {line_number}: {exc}") from exc


def unwrap_records(value: Any) -> Iterator[Mapping[str, Any]]:
    if isinstance(value, list):
        for item in value:
            yield from unwrap_records(item)
        return

    if not isinstance(value, Mapping):
        return

    for key in WRAPPER_KEYS:
        child = value.get(key)
        if isinstance(child, (list, Mapping)):
            yield from unwrap_records(child)
            return

    yield value


def extract_text(record: Mapping[str, Any]) -> str:
    for key in TEXT_KEYS:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()

    for path in NESTED_TEXT_PATHS:
        value = nested_value(record, path)
        if isinstance(value, str) and value.strip():
            return value.strip()

    return ""


def extract_sentiment(record: Mapping[str, Any], default_sentiment: str) -> str:
    for key in SENTIMENT_KEYS:
        value = record.get(key)
        if value is None:
            continue
        label = LABELS.get(str(value).strip().lower())
        if label:
            return label
    return default_sentiment


def nested_value(record: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = record
    for key in path:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


if __name__ == "__main__":
    raise SystemExit(main())
