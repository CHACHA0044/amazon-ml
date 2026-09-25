"""Data loading utilities for streaming and chunked TSV processing."""

import csv
import os
from typing import Iterator, Dict, List, Optional
import polars as pl


def stream_tsv_rows(filepath: str) -> Iterator[Dict[str, str]]:
    """Stream TSV rows one by one with minimal memory footprint."""
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            yield row


def load_table_polars(filepath: str, columns: Optional[List[str]] = None) -> pl.DataFrame:
    """Load TSV using Polars for high-speed columnar processing."""
    return pl.read_csv(
        filepath,
        separator="\t",
        columns=columns,
        quote_char=None,
        truncate_ragged_lines=True,
    )
