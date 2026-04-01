"""
delete_files_by_prefix.py

Delete files whose names start with a given prefix from two specified folders.

By default the script performs a dry run and only lists matching files.
Pass --confirm to actually delete them. This makes it safe for use on
non-interactive environments such as HPC clusters.

Usage:
    # Dry run (lists files, deletes nothing):
    python delete_files_by_prefix.py <folder1> <folder2> --prefix <prefix>

    # Live deletion:
    python delete_files_by_prefix.py <folder1> <folder2> --prefix <prefix> --confirm

Example:
    python delete_files_by_prefix.py TRAIN_DATA VALIDATION_DATA --prefix m291 --confirm
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Delete files starting with a given prefix from two folders."
    )
    parser.add_argument(
        "folder1",
        type=Path,
        help="First folder to scan.",
    )
    parser.add_argument(
        "folder2",
        type=Path,
        help="Second folder to scan.",
    )
    parser.add_argument(
        "--prefix",
        required=True,
        help="Filename prefix. Only files whose names start with this string are deleted.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        default=False,
        help="Actually delete the files. Without this flag the script is a dry run.",
    )
    return parser.parse_args()


def validate_folder(folder: Path) -> None:
    if not folder.exists():
        log.error("Folder does not exist: %s", folder)
        sys.exit(1)
    if not folder.is_dir():
        log.error("Path is not a directory: %s", folder)
        sys.exit(1)


def collect_matches(folder: Path, prefix: str) -> list:
    return sorted(p for p in folder.iterdir() if p.is_file() and p.name.startswith(prefix))


def main():
    args = parse_args()

    validate_folder(args.folder1)
    validate_folder(args.folder2)

    matches1 = collect_matches(args.folder1, args.prefix)
    matches2 = collect_matches(args.folder2, args.prefix)
    all_matches = matches1 + matches2

    if not all_matches:
        log.info(
            "No files found starting with prefix '%s' in either folder. Nothing to delete.",
            args.prefix,
        )
        sys.exit(0)

    label = "DRY RUN" if not args.confirm else "LIVE"
    log.info("[%s] Files matching prefix '%s':", label, args.prefix)
    for p in all_matches:
        log.info("  %s", p)
    log.info("Total: %d file(s).", len(all_matches))

    if not args.confirm:
        log.info("Dry run complete. Re-run with --confirm to delete these files.")
        sys.exit(0)

    deleted = 0
    failed = 0
    for p in all_matches:
        try:
            p.unlink()
            log.info("Deleted: %s", p)
            deleted += 1
        except OSError as exc:
            log.error("Failed to delete %s: %s", p, exc)
            failed += 1

    log.info("Done. Deleted: %d, Failed: %d.", deleted, failed)


if __name__ == "__main__":
    main()
