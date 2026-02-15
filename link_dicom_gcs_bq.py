#!/usr/bin/env python3
"""Link DICOM files in a GCS bucket to a BigQuery metadata table via SOPInstanceUID.

Iterates over DICOM files in a GCS path, extracts SOPInstanceUID from each,
then joins against a BigQuery table to confirm which ones are present.
Outputs a table of (SOPInstanceUID, gcs_path).
"""

import argparse
import logging
import os
import signal
import sys
import threading
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO

import pandas as pd
import pydicom
from google.cloud import bigquery, storage
from tqdm import tqdm

# Suppress noisy warnings from google-cloud-bigquery about optional dependencies
warnings.filterwarnings("ignore", message=".*pandas-gbq.*")
warnings.filterwarnings("ignore", message=".*BigQuery Storage module not found.*")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

class PerfTracker:
    """Lightweight wall-clock profiler that accumulates time per named phase."""

    def __init__(self):
        self.totals: dict[str, float] = {}
        self.counts: dict[str, int] = {}
        self.metadata: dict[str, object] = {}
        self._current_phase: str | None = None
        self._phase_start: float = 0.0
        self.start_time: float = time.monotonic()
        self._lock = threading.Lock()

    def begin(self, phase: str) -> None:
        self._end_current()
        self._current_phase = phase
        self._phase_start = time.monotonic()

    def record(self, phase: str, elapsed: float) -> None:
        """Thread-safe: record elapsed time for a phase (used from workers)."""
        with self._lock:
            self.totals[phase] = self.totals.get(phase, 0.0) + elapsed
            self.counts[phase] = self.counts.get(phase, 0) + 1

    def _end_current(self) -> None:
        if self._current_phase is not None:
            elapsed = time.monotonic() - self._phase_start
            self.totals[self._current_phase] = (
                self.totals.get(self._current_phase, 0.0) + elapsed
            )
            self.counts[self._current_phase] = (
                self.counts.get(self._current_phase, 0) + 1
            )
            self._current_phase = None

    def summary(self) -> str:
        self._end_current()
        wall = time.monotonic() - self.start_time
        header = f"\nProfiling ({wall:.1f}s wall clock"
        if "workers" in self.metadata:
            header += f", {self.metadata['workers']} threads"
        header += "):"
        lines = [header]
        for phase, total in sorted(self.totals.items(), key=lambda x: -x[1]):
            count = self.counts[phase]
            pct = total / wall * 100 if wall > 0 else 0
            avg = total / count if count > 1 else total
            line = f"  {phase:30s}  {total:8.1f}s  {pct:5.1f}%"
            if count > 1:
                line += f"  (n={count:,}, avg={avg:.3f}s)"
            lines.append(line)
        return "\n".join(lines)


perf = PerfTracker()


def _sigint_handler(signum, frame):
    """On Ctrl+C, print profiling summary and exit."""
    print("\n\nInterrupted. Partial profiling results:", file=sys.stderr)
    print(perf.summary(), file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# GCS helpers
# ---------------------------------------------------------------------------

def parse_gcs_url(gcs_url: str) -> tuple[str, str]:
    """Parse 'gs://bucket/prefix/' into (bucket_name, prefix)."""
    if not gcs_url.startswith("gs://"):
        raise ValueError(f"Invalid GCS URL (must start with gs://): {gcs_url}")
    path = gcs_url[len("gs://"):]
    parts = path.split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    # Ensure prefix ends with '/' so we don't match sibling folders
    # e.g., "series_chunk_003" would also match "series_chunk_0031"
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    return bucket_name, prefix


def list_dicom_blobs(
    gcs_client: storage.Client,
    bucket_name: str,
    prefix: str,
    include_all: bool = False,
) -> list[storage.Blob]:
    """List blobs under the GCS prefix, filtered to likely DICOM files."""
    blobs = []
    with tqdm(desc="Listing GCS files", unit=" files") as pbar:
        for blob in gcs_client.list_blobs(bucket_name, prefix=prefix):
            if blob.name.endswith("/"):
                continue  # skip directory markers
            if include_all or _is_likely_dicom(blob.name):
                blobs.append(blob)
                pbar.update(1)
    return blobs


_SKIP_FILENAMES = {
    "license", "readme", "readme.md", "readme.txt", "changes", "changelog",
    "changelog.md", "notice", "manifest", "manifest.txt", "metadata.json",
    ".ds_store", "thumbs.db", "desktop.ini",
}


def _listing_table_id(output_table: str) -> str:
    """Derive the listing table name from the output table name."""
    return f"{output_table}__listing"


def save_listing_to_bq(
    gcs_client: storage.Client,
    bucket_name: str,
    prefix: str,
    listing_table: str,
    bq_client: bigquery.Client,
    include_all: bool = False,
    chunk_size: int = 10000,
) -> list[str]:
    """List blobs in GCS, stream paths to a BQ listing table in chunks.

    Truncates the listing table on the first chunk, then appends subsequent
    chunks. Returns the full list of gcs_path strings.
    """
    all_paths: list[str] = []
    chunk: list[str] = []
    first_chunk = True
    chunks_flushed = 0

    with tqdm(desc="Listing GCS files", unit=" files") as pbar:
        for blob in gcs_client.list_blobs(bucket_name, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            if include_all or _is_likely_dicom(blob.name):
                gcs_path = f"gs://{bucket_name}/{blob.name}"
                all_paths.append(gcs_path)
                chunk.append(gcs_path)
                pbar.update(1)

                if len(chunk) >= chunk_size:
                    chunks_flushed += 1
                    pbar.set_postfix(bq_chunks=chunks_flushed)
                    _flush_listing_chunk(
                        chunk, listing_table, bq_client, truncate=first_chunk,
                    )
                    first_chunk = False
                    chunk = []

    # Flush remaining
    if chunk:
        _flush_listing_chunk(
            chunk, listing_table, bq_client, truncate=first_chunk,
        )

    return all_paths


def _flush_listing_chunk(
    paths: list[str],
    listing_table: str,
    bq_client: bigquery.Client,
    truncate: bool = False,
) -> None:
    """Write a chunk of gcs_path values to the listing table."""
    t0 = time.monotonic()
    df = pd.DataFrame({"gcs_path": paths})
    disposition = (
        bigquery.WriteDisposition.WRITE_TRUNCATE if truncate
        else bigquery.WriteDisposition.WRITE_APPEND
    )
    job_config = bigquery.LoadJobConfig(write_disposition=disposition)
    load_job = bq_client.load_table_from_dataframe(
        df, listing_table, job_config=job_config,
    )
    load_job.result()
    perf.record("bq_flush_listing", time.monotonic() - t0)


def load_listing_from_bq(
    listing_table: str,
    bq_client: bigquery.Client,
) -> list[str] | None:
    """Load the cached GCS file listing from BQ. Returns None if table doesn't exist."""
    try:
        bq_client.get_table(listing_table)
    except Exception:
        return None

    sql = f"SELECT gcs_path FROM `{listing_table}`"
    df = bq_client.query(sql).to_dataframe()
    return list(df["gcs_path"])


def _is_likely_dicom(name: str) -> bool:
    """Return True if the file is likely DICOM (.dcm or no extension)."""
    basename = os.path.basename(name)
    if basename.lower() in _SKIP_FILENAMES:
        return False
    if basename.lower().endswith(".dcm"):
        return True
    # Files with no extension are common in DICOM
    if "." not in basename:
        return True
    return False


# ---------------------------------------------------------------------------
# DICOM metadata extraction
# ---------------------------------------------------------------------------

_HEADER_SMALL = 1024       # 1KB — SOPInstanceUID (0008,0018) is typically
                           # within the first ~700 bytes; this hits most files.
_HEADER_LARGE = 16 * 1024  # 16KB — fallback for unusual files with extra
                           # elements before SOPInstanceUID.


def _try_parse_sop_uid(data: bytes) -> str | None:
    """Try to extract SOPInstanceUID from a byte buffer."""
    ds = pydicom.dcmread(
        BytesIO(data),
        stop_before_pixels=True,
        specific_tags=["SOPInstanceUID"],
        force=True,
    )
    sop_uid = getattr(ds, "SOPInstanceUID", None)
    return str(sop_uid) if sop_uid is not None else None


def extract_sop_uid_from_blob(blob: storage.Blob) -> tuple[str | None, str]:
    """Extract SOPInstanceUID using adaptive partial downloads.

    First tries a 1KB read (sufficient for most DICOM files). If
    SOPInstanceUID isn't found, retries with 16KB. This minimizes
    network I/O for the common case while still handling unusual files.

    Returns (sop_uid_or_none, gcs_path).
    """
    gcs_path = f"gs://{blob.bucket.name}/{blob.name}"
    try:
        # First attempt: small read
        t0 = time.monotonic()
        header_bytes = blob.download_as_bytes(start=0, end=_HEADER_SMALL - 1)
        t1 = time.monotonic()
        perf.record("gcs_download", t1 - t0)

        t_parse = time.monotonic()
        sop_uid = _try_parse_sop_uid(header_bytes)
        perf.record("dicom_parse", time.monotonic() - t_parse)

        if sop_uid is not None:
            return sop_uid, gcs_path

        # Fallback: larger read
        t0 = time.monotonic()
        header_bytes = blob.download_as_bytes(start=0, end=_HEADER_LARGE - 1)
        t1 = time.monotonic()
        perf.record("gcs_download", t1 - t0)
        perf.record("gcs_download_retry", t1 - t0)

        t_parse = time.monotonic()
        sop_uid = _try_parse_sop_uid(header_bytes)
        perf.record("dicom_parse", time.monotonic() - t_parse)

        if sop_uid is not None:
            return sop_uid, gcs_path

        logger.debug("No SOPInstanceUID found in %s", gcs_path)
        return None, gcs_path
    except Exception as exc:
        logger.debug("Failed to read %s: %s", gcs_path, exc)
        return None, gcs_path


# ---------------------------------------------------------------------------
# BigQuery helpers
# ---------------------------------------------------------------------------

def flush_chunk_to_bq(
    chunk: list[dict],
    bq_table_id: str,
    sop_column: str,
    output_table: str,
    bq_client: bigquery.Client,
) -> int:
    """Write all processed paths to the output table, join to count matches.

    All paths (including failures with NULL SOPInstanceUID) are written so that
    the incremental skip check can exclude them on restarts.

    Returns the number of matched rows.
    """
    t0 = time.monotonic()
    gcs_df = pd.DataFrame(chunk)

    # Write all processed paths (successes and failures) for skip tracking
    upload_to_bq(gcs_df, output_table, bq_client, append=True)

    # Join only the rows that have a SOPInstanceUID to count matches
    has_uid = gcs_df[gcs_df["SOPInstanceUID"].notna()]
    matched = 0
    if not has_uid.empty:
        matched_df = join_with_bq_table(has_uid, bq_table_id, sop_column, bq_client)
        matched = len(matched_df)

    perf.record("bq_flush_chunk", time.monotonic() - t0)
    return matched


def join_with_bq_table(
    gcs_df: pd.DataFrame,
    bq_table_id: str,
    sop_column: str,
    bq_client: bigquery.Client,
) -> pd.DataFrame:
    """Upload gcs_df as a temp table, JOIN with the target table, return results."""
    parts = bq_table_id.split(".")
    if len(parts) != 3:
        raise ValueError(
            f"bq_table must be fully qualified as project.dataset.table, got: {bq_table_id}"
        )
    project, dataset, _ = parts
    temp_table_id = f"{project}.{dataset}._temp_sop_lookup_{uuid.uuid4().hex[:8]}"

    try:
        # Upload temp table
        job_config = bigquery.LoadJobConfig(
            write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
        )
        load_job = bq_client.load_table_from_dataframe(
            gcs_df, temp_table_id, job_config=job_config,
        )
        load_job.result()  # wait for completion

        # Run JOIN query
        sql = f"""
            SELECT t.{sop_column} AS SOPInstanceUID, g.gcs_path
            FROM `{bq_table_id}` AS t
            INNER JOIN `{temp_table_id}` AS g
            ON t.{sop_column} = g.SOPInstanceUID
        """
        result_df = bq_client.query(sql).to_dataframe()
        return result_df

    except Exception as exc:
        # If temp-table creation fails (e.g. permissions), fall back to local join
        if "Access Denied" in str(exc) or "403" in str(exc):
            logger.warning(
                "Cannot create temp table (permission denied). "
                "Falling back to local pandas join."
            )
            return _local_join_fallback(gcs_df, bq_table_id, sop_column, bq_client)
        raise
    finally:
        # Clean up temp table
        bq_client.delete_table(temp_table_id, not_found_ok=True)


def _local_join_fallback(
    gcs_df: pd.DataFrame,
    bq_table_id: str,
    sop_column: str,
    bq_client: bigquery.Client,
) -> pd.DataFrame:
    """Fallback: download SOPInstanceUID column from BQ and join locally."""
    print(f"Downloading {sop_column} column from {bq_table_id}...", file=sys.stderr)
    sql = f"SELECT {sop_column} FROM `{bq_table_id}`"
    bq_df = bq_client.query(sql).to_dataframe()
    bq_df = bq_df.rename(columns={sop_column: "SOPInstanceUID"})
    return pd.merge(gcs_df, bq_df, on="SOPInstanceUID", how="inner")


def get_already_processed_paths(
    output_table: str,
    bq_client: bigquery.Client,
) -> set[str]:
    """Query the output BQ table for gcs_path values already processed.

    Returns an empty set if the table does not exist.
    """
    try:
        bq_client.get_table(output_table)
    except Exception:
        return set()

    sql = f"SELECT gcs_path FROM `{output_table}`"
    df = bq_client.query(sql).to_dataframe()
    return set(df["gcs_path"])


def upload_to_bq(
    df: pd.DataFrame,
    destination_table: str,
    bq_client: bigquery.Client,
    append: bool = False,
) -> None:
    """Upload a DataFrame to a BigQuery table."""
    disposition = (
        bigquery.WriteDisposition.WRITE_APPEND if append
        else bigquery.WriteDisposition.WRITE_TRUNCATE
    )
    job_config = bigquery.LoadJobConfig(write_disposition=disposition)
    load_job = bq_client.load_table_from_dataframe(
        df, destination_table, job_config=job_config,
    )
    load_job.result()


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_results(
    df: pd.DataFrame,
    output_mode: str,
    output_path: str | None,
    bq_client: bigquery.Client | None = None,
    append: bool = False,
) -> None:
    """Write the result DataFrame to the chosen destination."""
    if output_mode == "csv":
        df.to_csv(output_path, index=False)
        print(f"  Output: {output_path}", file=sys.stderr)
    elif output_mode == "bq":
        upload_to_bq(df, output_path, bq_client, append=append)
        action = "Appended to" if append else "Wrote to"
        print(f"  Output: {action} BigQuery table {output_path}", file=sys.stderr)
    else:  # stdout
        print(df.to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Match DICOM SOPInstanceUIDs from a GCS path against a BigQuery table.",
    )
    parser.add_argument(
        "gcs_url",
        help="GCS path containing DICOM files (e.g. gs://bucket/path/to/dicoms/)",
    )
    parser.add_argument(
        "bq_table",
        help="Fully-qualified BigQuery table ID (e.g. project.dataset.table)",
    )
    parser.add_argument(
        "--output-mode",
        choices=["stdout", "csv", "bq"],
        default="stdout",
        help="How to output results (default: stdout)",
    )
    parser.add_argument(
        "--output-path",
        help="File path for CSV output, or BQ table ID for BQ output. "
             "Required when --output-mode is csv or bq.",
    )
    parser.add_argument(
        "--sop-column",
        default="SOPInstanceUID",
        help="Name of the SOPInstanceUID column in the BigQuery table (default: SOPInstanceUID)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel threads for reading DICOM files (default: 8)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="When --output-mode=bq, flush results to BigQuery every N files "
             "(default: 5000). Smaller values = more fault tolerance but more "
             "BQ round-trips (~3-5s overhead each).",
    )
    parser.add_argument(
        "--include-all-files",
        action="store_true",
        help="Process all files, not just .dcm and extensionless",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-listing files from GCS even if a cached listing exists "
             "in BigQuery. Use when new files have been added to the bucket.",
    )
    parser.add_argument(
        "--listing-chunk-size",
        type=int,
        default=10000,
        help="Number of GCS paths to flush to the listing table at a time "
             "(default: 10000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Validate arguments
    if args.output_mode in ("csv", "bq") and not args.output_path:
        parser.error(f"--output-path is required when --output-mode is {args.output_mode}")

    # Set up logging
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
        stream=sys.stderr,
    )

    signal.signal(signal.SIGINT, _sigint_handler)
    perf.metadata["workers"] = args.workers

    try:
        # --- Step 1: Parse GCS URL ---
        bucket_name, prefix = parse_gcs_url(args.gcs_url)

        # --- Step 2: Get file listing (cached BQ or fresh GCS) ---
        gcs_client = storage.Client()
        bq_client = bigquery.Client()
        use_bq = args.output_mode == "bq" and args.output_path
        listing_table = _listing_table_id(args.output_path) if use_bq else None

        gcs_paths: list[str] | None = None

        if use_bq and not args.refresh:
            perf.begin("bq_load_listing")
            cached = load_listing_from_bq(listing_table, bq_client)
            if cached is not None:
                gcs_paths = cached
                print(
                    f"Loaded {len(gcs_paths):,} paths from cached listing "
                    f"({listing_table})",
                    file=sys.stderr,
                )

        if gcs_paths is None:
            # List from GCS (first run or --refresh)
            perf.begin("gcs_list_blobs")
            if use_bq:
                gcs_paths = save_listing_to_bq(
                    gcs_client, bucket_name, prefix, listing_table,
                    bq_client, args.include_all_files, args.listing_chunk_size,
                )
                print(
                    f"Saved {len(gcs_paths):,} paths to listing table "
                    f"({listing_table})",
                    file=sys.stderr,
                )
            else:
                blobs = list_dicom_blobs(
                    gcs_client, bucket_name, prefix, args.include_all_files,
                )
                gcs_paths = [f"gs://{b.bucket.name}/{b.name}" for b in blobs]

        total_files = len(gcs_paths)
        if total_files == 0:
            print("No DICOM files found at the specified GCS path.", file=sys.stderr)
            sys.exit(0)

        # --- Step 2b: Skip already-processed files (incremental mode) ---
        perf.begin("bq_skip_check")
        if use_bq:
            already_done = get_already_processed_paths(args.output_path, bq_client)
            if already_done:
                before = len(gcs_paths)
                gcs_paths = [p for p in gcs_paths if p not in already_done]
                skipped_existing = before - len(gcs_paths)
                print(
                    f"Skipping {skipped_existing:,} already-processed files "
                    f"(found in {args.output_path})",
                    file=sys.stderr,
                )

                if not gcs_paths:
                    print("All files already processed. Nothing to do.", file=sys.stderr)
                    sys.exit(0)

        # Convert gcs_paths to Blob objects for DICOM processing
        bucket = gcs_client.bucket(bucket_name)
        blobs = [bucket.blob(p.split(f"gs://{bucket_name}/", 1)[1]) for p in gcs_paths]

        # --- Step 3: Extract SOPInstanceUIDs (with chunked BQ writes) ---
        perf.begin("dicom_processing")
        use_chunked_bq = args.output_mode == "bq" and args.output_path
        chunk_buffer = []
        total_extracted = 0
        total_matched = 0
        total_errors = 0
        chunks_flushed = 0

        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(extract_sop_uid_from_blob, blob): blob
                for blob in blobs
            }
            with tqdm(total=len(futures), desc="Processing DICOM files", unit=" files") as pbar:
                for future in as_completed(futures):
                    sop_uid, gcs_path = future.result()
                    chunk_buffer.append({"SOPInstanceUID": sop_uid, "gcs_path": gcs_path})
                    if sop_uid is not None:
                        total_extracted += 1
                    else:
                        total_errors += 1
                    pbar.update(1)

                    # Flush chunk to BQ when buffer is full
                    if use_chunked_bq and len(chunk_buffer) >= args.chunk_size:
                        chunks_flushed += 1
                        pbar.set_postfix(chunk=chunks_flushed, matched=total_matched)
                        matched = flush_chunk_to_bq(
                            chunk_buffer, args.bq_table, args.sop_column,
                            args.output_path, bq_client,
                        )
                        total_matched += matched
                        chunk_buffer = []

        # Flush remaining buffer
        if chunk_buffer:
            if use_chunked_bq:
                matched = flush_chunk_to_bq(
                    chunk_buffer, args.bq_table, args.sop_column,
                    args.output_path, bq_client,
                )
                total_matched += matched
            # For non-BQ modes, do the join and output in one shot
            elif total_extracted > 0:
                gcs_df = pd.DataFrame(chunk_buffer)
                has_uid = gcs_df[gcs_df["SOPInstanceUID"].notna()]
                matched_df = join_with_bq_table(
                    has_uid, args.bq_table, args.sop_column, bq_client,
                )
                total_matched = len(matched_df)
                write_results(matched_df, args.output_mode, args.output_path, bq_client)

        if total_extracted == 0:
            print("No SOPInstanceUIDs could be extracted from any files.", file=sys.stderr)
            sys.exit(0)

        # --- Summary ---
        skipped_msg = f" ({total_errors} skipped)" if total_errors else ""
        print(f"\nDone.", file=sys.stderr)
        print(f"  GCS files scanned:     {len(blobs):,}", file=sys.stderr)
        print(f"  SOPInstanceUIDs found: {total_extracted:,}{skipped_msg}", file=sys.stderr)
        print(f"  Matched in BigQuery:   {total_matched:,}", file=sys.stderr)
        if use_chunked_bq:
            print(f"  Output: BigQuery table {args.output_path}", file=sys.stderr)
        print(perf.summary(), file=sys.stderr)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Partial profiling results:", file=sys.stderr)
        print(perf.summary(), file=sys.stderr)
        sys.exit(1)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        if "DefaultCredentialsError" in type(exc).__name__ or "default credentials" in str(exc).lower():
            print(
                "Error: Google Cloud credentials not found.\n"
                "Run: gcloud auth application-default login",
                file=sys.stderr,
            )
        else:
            logger.debug("Unhandled exception", exc_info=True)
            print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
