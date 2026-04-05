import os
import shutil
import logging
import threading
import concurrent.futures
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Logging setup -- FileHandler only, no StreamHandler, ASCII format
# ---------------------------------------------------------------------------

class _ThreadSafeFileHandler(logging.FileHandler):
    """FileHandler with an internal lock to prevent interleaved writes."""

    def __init__(self, filename, mode='a', encoding=None, delay=False):
        super().__init__(filename, mode=mode, encoding=encoding, delay=delay)
        self._write_lock = threading.Lock()

    def emit(self, record):
        with self._write_lock:
            super().emit(record)


def _configure_logger():
    logger = logging.getLogger('merge_training_data')
    logger.setLevel(logging.DEBUG)
    if logger.handlers:
        return logger
    DEST_ROOT.mkdir(parents=True, exist_ok=True)
    handler = _ThreadSafeFileHandler(str(DEST_ROOT / 'merge_training_data.log'), encoding='utf-8')
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter(
        fmt='%(asctime)s  %(levelname)-8s  %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    return logger


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_ROOTS = [
    '/scratch/22206468/TRAIN_DATA',
    '/scratch/22206468/TRAIN_DATA_2',
    '/scratch/22206468/TRAIN_DATA_3',
    '/scratch/22206468/TRAIN_DATA_4',
    '/scratch/22206468/TRAIN_DATA_5',
]

DEST_ROOT = Path('/scratch/22206468/TRAIN_DATA_MERGED')
DEST_SEIZURE = DEST_ROOT / 'seizure'
DEST_NON_SEIZURE = DEST_ROOT / 'non_seizure'
LABEL_ARRAY_PATH = DEST_ROOT / 'train_y.npy'
LABEL_TEXT_PATH = DEST_ROOT / 'train_y_filenames.txt'

SUBDIR_MAP = {
    'seizure': DEST_SEIZURE,
    'non_seizure': DEST_NON_SEIZURE,
}

LOG_INTERVAL = 500


# ---------------------------------------------------------------------------
# File-collection helpers
# ---------------------------------------------------------------------------

def _collect_copy_tasks(source_roots, logger):
    """Return a list of (src_path, dest_path) pairs for every .npy file found."""
    tasks = []
    for root_name in source_roots:
        root = Path(root_name)
        if not root.is_dir():
            logger.warning('Source root not found, skipping: %s', root_name)
            continue
        for subdir_name, dest_dir in SUBDIR_MAP.items():
            src_subdir = root / subdir_name
            if not src_subdir.is_dir():
                logger.warning(
                    'Subdirectory missing in %s, skipping: %s',
                    root_name, subdir_name
                )
                continue
            for src_file in src_subdir.glob('*.npy'):
                dest_name = src_file.name
                dest_file = dest_dir / dest_name
                if dest_file.exists():
                    dest_name = '%s_%s' % (root_name, src_file.name)
                    dest_file = dest_dir / dest_name
                tasks.append((src_file, dest_file))
    return tasks


# ---------------------------------------------------------------------------
# Copy worker
# ---------------------------------------------------------------------------

def _copy_file(src, dest, counter, counter_lock, total, logger):
    """Copy one file; return (src, dest, success, error_msg)."""
    try:
        shutil.copy2(src, dest)
    except Exception as exc:
        msg = 'FAILED to copy %s -> %s : %s' % (src, dest, exc)
        logger.warning(msg)
        return (src, dest, False, str(exc))

    with counter_lock:
        counter[0] += 1
        n = counter[0]
        if n % LOG_INTERVAL == 0:
            logger.info('Copied %d of %d files', n, total)

    return (src, dest, True, None)


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------

def _build_labels(logger):
    """Build train_y.npy and train_y_filenames.txt from merged directory."""
    seizure_files = sorted(
        f.name for f in DEST_SEIZURE.glob('*.npy')
    )
    non_seizure_files = sorted(
        f.name for f in DEST_NON_SEIZURE.glob('*.npy')
    )

    ictal_labels = np.ones(len(seizure_files), dtype=np.int8)
    interictal_labels = np.zeros(len(non_seizure_files), dtype=np.int8)
    train_y = np.concatenate([ictal_labels, interictal_labels])

    np.save(LABEL_ARRAY_PATH, train_y)
    logger.info('Saved label array to %s  shape=%s', LABEL_ARRAY_PATH, train_y.shape)

    lines = []
    for fname in seizure_files:
        lines.append('%s,1' % fname)
    for fname in non_seizure_files:
        lines.append('%s,0' % fname)
    LABEL_TEXT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    logger.info('Saved label text file to %s', LABEL_TEXT_PATH)

    return len(seizure_files), len(non_seizure_files)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    logger = _configure_logger()
    logger.info('=' * 72)
    logger.info('merge_training_data.py -- starting')

    # Create destination directories
    DEST_SEIZURE.mkdir(parents=True, exist_ok=True)
    DEST_NON_SEIZURE.mkdir(parents=True, exist_ok=True)
    logger.info('Destination directories ready: %s', DEST_ROOT)

    # Collect all copy tasks
    logger.info('Scanning source roots for .npy files ...')
    tasks = _collect_copy_tasks(SOURCE_ROOTS, logger)
    total = len(tasks)
    logger.info('Total files to copy: %d', total)

    if total == 0:
        logger.warning('No files found. Check SOURCE_ROOTS paths. Exiting.')
        raise SystemExit(0)

    # Parallel copy
    counter = [0]
    counter_lock = threading.Lock()
    failed_files = []

    max_workers = os.cpu_count() or 4
    logger.info('Launching ThreadPoolExecutor with max_workers=%d', max_workers)

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_task = {
            executor.submit(
                _copy_file, src, dest, counter, counter_lock, total, logger
            ): (src, dest)
            for src, dest in tasks
        }

        for future in concurrent.futures.as_completed(future_to_task):
            src, dest, success, err = future.result()
            if success:
                logger.debug('Copied: %s -> %s', src, dest)
            else:
                failed_files.append((str(src), str(dest), err))

    logger.info('Parallel copy complete. Copied %d of %d files.', counter[0], total)

    # Summarise failures
    if failed_files:
        logger.warning('Failed copies (%d total):', len(failed_files))
        for src, dest, err in failed_files:
            logger.warning('  FAILED  src=%s  dest=%s  err=%s', src, dest, err)
    else:
        logger.info('No copy failures.')

    # Build label files
    logger.info('Building label files ...')
    n_ictal, n_interictal = _build_labels(logger)

    # Class balance report
    n_total = n_ictal + n_interictal
    if n_interictal > 0:
        ratio = n_interictal / n_ictal if n_ictal > 0 else float('inf')
        logger.info(
            'Class balance -- ictal: %d  interictal: %d  total: %d  '
            'interictal:ictal ratio: %.2f:1',
            n_ictal, n_interictal, n_total, ratio
        )
    else:
        logger.info(
            'Class balance -- ictal: %d  interictal: %d  total: %d',
            n_ictal, n_interictal, n_total
        )

    logger.info('merge_training_data.py -- done')
    logger.info('=' * 72)
