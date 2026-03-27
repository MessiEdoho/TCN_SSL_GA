# -- Section 2: Library imports ------------------------------------------------

import logging                                      # structured logging to file and console
import mne                                          # read EDF headers and load data on demand
import numpy as np                                  # numerical array operations
import pandas as pd                                 # read Excel files and build summary tables
import matplotlib                                   # plotting backend configuration
matplotlib.use('Agg')                               # non-interactive backend for cluster use
import matplotlib.pyplot as plt                     # plotting API for bar charts
from pathlib import Path                            # cross-platform file path handling
from scipy.signal import iirnotch, filtfilt         # notch filter design and zero-phase filtering
from scipy.stats import median_abs_deviation        # robust scale estimator for MAD


# -- Section 3: Configuration --------------------------------------------------

# -- Paths (EDF and annotation files are in SEPARATE directories) --------------
input_dir      = Path("/home/people/22206468/scratch/EEG_TRAINING_PLANB")                     # directory containing .edf files only
annotation_dir = Path("/home/people/22206468/scratch/seizure_times_updated'")    # directory containing .xlsx files only
output_dir     = Path("/home/people/22206468/scratch/TRAIN_DATA_PLANB")                   # root output directory for saved segments

# -- Signal parameters ---------------------------------------------------------
fs       = 500    # sampling rate in Hz -- must match the EDF file header
win_len  = 2500   # window length in samples: 5 s * 500 Hz
step     = 1250   # step size in samples: 2.5 s * 500 Hz (50% overlap)

# -- Chunking ------------------------------------------------------------------
segments_per_chunk = 40  # number of segments per RAM-loaded block

# -- Preprocessing -------------------------------------------------------------
notch_freq = 50.0  # powerline notch filter centre frequency in Hz
notch_Q    = 30    # notch filter Q-factor (higher = narrower notch)

# -- Labelling windows ---------------------------------------------------------
preictal_window_sec  = 1800  # pre-ictal context window in seconds (30 min)
postictal_window_sec = 1800  # post-ictal context window in seconds (30 min)

# -- Interictal reduction ------------------------------------------------------
ictal_to_interictal_ratio = 2   # max interictal segments per ictal segment
random_seed               = 42  # seed for reproducible random under-sampling

# -- Logging setup -------------------------------------------------------------
output_dir.mkdir(parents=True, exist_ok=True)  # create output directory

logging.basicConfig(
    level=logging.INFO,                                            # minimum log level
    format="%(asctime)s | %(levelname)s | %(message)s",            # log format
    datefmt="%Y-%m-%d %H:%M:%S",                                  # timestamp format
    handlers=[
        logging.StreamHandler(),                                   # log to console
        logging.FileHandler(output_dir / "plan_b_traineeg.log",           # log to file
                            mode="w", encoding="utf-8")
    ],
    force=True                                                     # override any prior config
)
logger = logging.getLogger(__name__)  # create module-level logger

logger.info("Configuration loaded.")
logger.info(f"  EDF dir:        {input_dir}")
logger.info(f"  Annotation dir: {annotation_dir}")
logger.info(f"  Output dir:     {output_dir}")


# -- load_annotations ----------------------------------------------------------

def load_annotations(xlsx_path, recording_start_dt):
    """
    Read a seizure annotation Excel file and convert timestamps to seconds
    relative to the EDF recording start.

    Parameters
    ----------
    xlsx_path : pathlib.Path
        Path to the .xlsx annotation file.
    recording_start_dt : datetime.datetime
        Timezone-naive recording start datetime from the EDF header.

    Returns
    -------
    intervals : list of tuple(float, float)
        Each tuple is (start_sec, end_sec) relative to recording start.
        Returns empty list if the file cannot be read.
    """
    try:
        df = pd.read_excel(xlsx_path)                          # load the Excel file into a DataFrame
    except Exception as e:
        logger.error(f"Failed to read annotation file {xlsx_path}: {e}")
        return []                                              # return empty on failure

    df['start_time'] = pd.to_datetime(                         # parse onset timestamps
        df['start_time'], dayfirst=True                        # DD/MM/YYYY format
    )
    df['end_time'] = pd.to_datetime(                           # parse offset timestamps
        df['end_time'], dayfirst=True                          # DD/MM/YYYY format
    )

    intervals = []                                             # accumulate (start_sec, end_sec) tuples
    for _, row in df.iterrows():                               # iterate over each annotation row
        s_dt = row['start_time'].to_pydatetime().replace(tzinfo=None)  # strip timezone
        e_dt = row['end_time'].to_pydatetime().replace(tzinfo=None)    # strip timezone
        s_sec = (s_dt - recording_start_dt).total_seconds()    # seconds from recording start
        e_sec = (e_dt - recording_start_dt).total_seconds()    # seconds from recording start
        intervals.append((s_sec, e_sec))                       # store as tuple

    return intervals                                           # return all seizure intervals


# -- compute_segment_grid ------------------------------------------------------

def compute_segment_grid(n_samples, win_len, step):
    """
    Pre-compute every valid segment start index for the entire recording.

    Parameters
    ----------
    n_samples : int
        Total samples in the recording.
    win_len : int
        Segment length in samples.
    step : int
        Step size between consecutive segment starts.

    Returns
    -------
    starts : np.ndarray, dtype int64
        1-D array of valid segment start indices.
    """
    last_valid = n_samples - win_len                           # largest start where a full window fits
    starts = np.arange(0, last_valid + 1, step, dtype=np.int64)  # evenly spaced start indices
    return starts                                              # return the complete segment grid


# -- Preprocessing functions ---------------------------------------------------

def apply_notch_filter(signal, fs, freq, Q):
    """
    Remove powerline interference with a zero-phase IIR notch filter.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D raw EEG signal.
    fs : float
        Sampling rate in Hz.
    freq : float
        Centre frequency of the notch in Hz.
    Q : float
        Quality factor.

    Returns
    -------
    filtered : np.ndarray, shape (n,)
        Notch-filtered signal.
    """
    w0 = freq / (fs / 2.0)                                    # normalise to Nyquist
    b, a = iirnotch(w0, Q)                                    # design notch filter coefficients
    filtered = filtfilt(b, a, signal)                          # zero-phase forward-backward filtering
    return filtered                                            # return filtered signal


def remove_dc(signal):
    """
    Remove DC offset by subtracting the signal mean.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D EEG signal.

    Returns
    -------
    centred : np.ndarray, shape (n,)
        Signal with zero mean.
    """
    dc = np.mean(signal)                                       # compute mean (DC component)
    centred = signal - dc                                      # subtract from every sample
    return centred                                             # return DC-corrected signal


def robust_zscore(signal):
    """
    Robust z-score normalisation using median and MAD.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D EEG signal.

    Returns
    -------
    normed : np.ndarray, shape (n,)
        Normalised signal. Returned unchanged if MAD = 0.
    """
    med = np.median(signal)                                    # robust location estimate
    mad = median_abs_deviation(signal, scale=1)                # MAD without built-in scaling
    if mad == 0.0:                                             # constant signal -- cannot normalise
        logger.warning("MAD = 0 encountered; returning chunk unchanged.")
        return signal                                          # return unchanged
    normed = (signal - med) / (1.4826 * mad)                   # robust z-score formula
    return normed                                              # return normalised signal


def preprocess_chunk(chunk, fs, notch_freq, notch_Q):
    """
    Apply the full preprocessing pipeline to one chunk block.

    Parameters
    ----------
    chunk : np.ndarray, shape (n,)
        Raw 1-D signal for one chunk.
    fs : float
        Sampling rate in Hz.
    notch_freq : float
        Notch centre frequency.
    notch_Q : float
        Notch Q-factor.

    Returns
    -------
    processed : np.ndarray, shape (n,)
        Preprocessed chunk.
    """
    step1 = apply_notch_filter(chunk, fs, notch_freq, notch_Q)  # step 1: remove 50 Hz
    step2 = remove_dc(step1)                                     # step 2: subtract DC offset
    processed = robust_zscore(step2)                              # step 3: robust z-score
    return processed                                             # return preprocessed chunk


# -- Labelling functions -------------------------------------------------------

def assign_label(seg_start_sec, seg_end_sec, seizure_intervals,
                 preictal_window_sec, postictal_window_sec):
    """
    Assign one of four labels to a segment based on temporal position.

    Priority order: ictal > pre-ictal > post-ictal > interictal.

    Parameters
    ----------
    seg_start_sec : float
        Segment start time in seconds relative to recording start.
    seg_end_sec : float
        Segment end time in seconds relative to recording start.
    seizure_intervals : list of tuple(float, float)
        Each tuple is (seizure_start_sec, seizure_end_sec).
    preictal_window_sec : float
        Duration of pre-ictal window in seconds.
    postictal_window_sec : float
        Duration of post-ictal window in seconds.

    Returns
    -------
    label : str
        One of "ictal", "preictal", "postictal", "interictal".
    """
    # Priority 1: ictal -- any overlap with any seizure
    for sz_s, sz_e in seizure_intervals:                       # check each seizure
        if seg_start_sec < sz_e and seg_end_sec > sz_s:        # overlap condition
            return "ictal"                                     # highest priority

    # Priority 2: pre-ictal -- segment end within pre-ictal window before any seizure
    for sz_s, sz_e in seizure_intervals:                       # check each seizure
        pre_start = sz_s - preictal_window_sec                 # pre-ictal window start
        if pre_start <= seg_end_sec <= sz_s:                   # segment end in window
            return "preictal"                                  # second priority

    # Priority 3: post-ictal -- segment start within post-ictal window after any seizure
    for sz_s, sz_e in seizure_intervals:                       # check each seizure
        post_end = sz_e + postictal_window_sec                 # post-ictal window end
        if sz_e <= seg_start_sec <= post_end:                  # segment start in window
            return "postictal"                                 # third priority

    return "interictal"                                        # default: none of the above


def undersample_interictal(interictal_indices, n_ictal, ratio, rng):
    """
    Under-sample interictal indices to limit the ictal-to-interictal ratio.

    Parameters
    ----------
    interictal_indices : list of int
        Sample start indices for interictal segments.
    n_ictal : int
        Number of ictal segments.
    ratio : int
        Maximum interictal segments per ictal segment.
    rng : numpy.random.Generator
        Seeded random number generator.

    Returns
    -------
    retained : list of int
        Retained interictal indices after under-sampling.
    """
    quota = n_ictal * ratio                                    # compute interictal quota
    if len(interictal_indices) <= quota:                        # pool smaller than quota
        logger.warning(f"Interictal pool ({len(interictal_indices)}) <= quota ({quota}); "
                       f"retaining all.")
        return list(interictal_indices)                         # return all without sampling
    retained = rng.choice(                                     # random under-sample
        interictal_indices, size=quota, replace=False           # no replacement
    ).tolist()                                                 # convert to plain list
    return retained                                            # return retained subset


# -- save_segment --------------------------------------------------------------

# Label-to-folder mapping
LABEL_DIR_MAP = {
    "ictal":      "seizure",       # ictal segments go into seizure/
    "preictal":   "preictal",      # pre-ictal segments go into preictal/
    "postictal":  "postictal",     # post-ictal segments go into postictal/
    "interictal": "non_seizure"    # interictal segments go into non_seizure/
}


def save_segment(segment, output_dir, mouse_id, label, index):
    """
    Save a single 1-D segment as a .npy file in the appropriate subdirectory.

    Parameters
    ----------
    segment : np.ndarray, shape (2500,)
        Preprocessed 1-D EEG segment.
    output_dir : pathlib.Path
        Root output directory.
    mouse_id : str
        Subject identifier (e.g. 'm1').
    label : str
        One of 'ictal', 'preictal', 'postictal', 'interictal'.
    index : int
        1-based index for this class and mouse.

    Returns
    -------
    filepath : pathlib.Path
        Resolved path to the saved file.
    """
    folder_name = LABEL_DIR_MAP[label]                         # map label to folder name
    class_dir = output_dir / folder_name                       # build full folder path
    class_dir.mkdir(parents=True, exist_ok=True)               # create if needed
    filename = f"{mouse_id}_{label}_{index:05d}.npy"           # e.g. m1_ictal_00001.npy
    filepath = class_dir / filename                            # full output path
    np.save(filepath, segment.astype(np.float32))              # save as float32
    return filepath.resolve()                                  # return resolved path


# -- Section 6: Main processing loop ------------------------------------------

results = {}  # per-mouse counts: {mouse_id: {label: count}}

edf_files = sorted(input_dir.glob("*.edf"))                   # find all EDF files
logger.info(f"Found {len(edf_files)} EDF file(s) in {input_dir}")

for edf_path in edf_files:                                    # iterate over each recording

    mouse_id = edf_path.stem                                   # extract mouse ID from filename
    xlsx_path = annotation_dir / f"{mouse_id}_xlsx.xlsx"            # look up annotation separately

    logger.info(f"--- {mouse_id} ---")
    logger.info(f"  EDF:        {edf_path}")
    logger.info(f"  Annotation: {xlsx_path}")

    if not xlsx_path.exists():                                 # guard: no annotation file
        logger.error(f"  Annotation file not found: {xlsx_path} -- skipping {mouse_id}")
        continue                                               # skip to next EDF

    # -- Open EDF header (no signal loaded) ------------------------------------
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
    except Exception as e:
        logger.error(f"  Failed to open EDF: {e} -- skipping {mouse_id}")
        continue                                               # skip on failure

    n_samples = int(raw.n_times)                               # total samples from header
    fs_file = raw.info['sfreq']                                # sampling rate from header
    rec_start = raw.info['meas_date'].replace(tzinfo=None)     # timezone-naive start datetime

    logger.info(f"  Samples: {n_samples} @ {fs_file} Hz ({n_samples/fs_file:.1f} s)")

    # -- Load annotations ------------------------------------------------------
    seizure_intervals = load_annotations(xlsx_path, rec_start) # list of (start_sec, end_sec)
    if not seizure_intervals:                                  # no seizures found
        logger.warning(f"  No seizure annotations found for {mouse_id}")
    logger.info(f"  Seizure intervals: {len(seizure_intervals)}")

    # -- Compute segment grid --------------------------------------------------
    segment_starts = compute_segment_grid(n_samples, win_len, step)  # all valid start indices
    n_segs = len(segment_starts)                               # total segments on grid
    logger.info(f"  Segment grid: {n_segs} segments (win={win_len}, step={step})")

    # ==========================================================================
    # PASS 1: Label every segment index (no signal data loaded)
    # ==========================================================================
    index_label_map = {}                                       # {start_index: label}
    ictal_idx = []                                             # ictal start indices
    preictal_idx = []                                          # pre-ictal start indices
    postictal_idx = []                                         # post-ictal start indices
    interictal_idx = []                                        # interictal start indices

    for si in segment_starts:                                  # iterate over every grid index
        seg_s = float(si) / fs                                 # segment start in seconds
        seg_e = float(si + win_len) / fs                       # segment end in seconds
        lbl = assign_label(seg_s, seg_e, seizure_intervals,    # assign label by priority
                           preictal_window_sec, postictal_window_sec)
        index_label_map[int(si)] = lbl                         # store index -> label mapping

        if lbl == "ictal":
            ictal_idx.append(int(si))                          # add to ictal list
        elif lbl == "preictal":
            preictal_idx.append(int(si))                       # add to pre-ictal list
        elif lbl == "postictal":
            postictal_idx.append(int(si))                      # add to post-ictal list
        else:
            interictal_idx.append(int(si))                     # add to interictal list

    logger.info(f"  Pass 1 (before reduction): "
                f"ictal={len(ictal_idx)} preictal={len(preictal_idx)} "
                f"postictal={len(postictal_idx)} interictal={len(interictal_idx)}")

    # -- Under-sample interictal -----------------------------------------------
    rng = np.random.default_rng(seed=random_seed)              # seeded RNG for reproducibility
    retained_inter = undersample_interictal(                    # under-sample interictal indices
        interictal_idx, len(ictal_idx),
        ictal_to_interictal_ratio, rng
    )

    logger.info(f"  Pass 1 (after reduction): "
                f"ictal={len(ictal_idx)} preictal={len(preictal_idx)} "
                f"postictal={len(postictal_idx)} interictal={len(retained_inter)}")

    # -- Merge and sort retained indices ---------------------------------------
    retained_indices = sorted(                                 # single sorted list of all retained
        ictal_idx + preictal_idx + postictal_idx + retained_inter
    )

    # Update index_label_map to only contain retained entries
    retained_set = set(retained_indices)                        # fast lookup set
    index_label_map = {k: v for k, v in index_label_map.items() if k in retained_set}

    logger.info(f"  Total retained segments: {len(retained_indices)}")

    # ==========================================================================
    # PASS 2: Load, preprocess, and save retained segments only
    # ==========================================================================
    class_counters = {"ictal": 0, "preictal": 0, "postictal": 0, "interictal": 0}

    # Group retained indices into chunks of segments_per_chunk
    chunk_groups = [                                           # split into groups of K
        retained_indices[i : i + segments_per_chunk]
        for i in range(0, len(retained_indices), segments_per_chunk)
    ]
    n_chunks = len(chunk_groups)                               # total chunks for this mouse

    for ci, group in enumerate(chunk_groups):                  # iterate over each chunk

        chunk_start = group[0]                                 # first sample in this chunk
        chunk_end = group[-1] + win_len                        # one-past-last sample

        raw_block = raw.get_data(start=chunk_start, stop=chunk_end)  # load only this chunk
        block = raw_block.squeeze().astype(np.float64)         # flatten to 1-D float64

        block = preprocess_chunk(block, fs, notch_freq, notch_Q)  # preprocess the block

        logger.info(f"  Chunk {ci+1:3d}/{n_chunks}: "
                    f"samples [{chunk_start}:{chunk_end}] ({len(group)} segs)")

        for si in group:                                       # iterate over each segment in chunk
            local_start = si - chunk_start                     # offset within loaded block
            segment = block[local_start : local_start + win_len]  # extract 2500-sample window
            label = index_label_map[si]                        # retrieve pre-assigned label
            class_counters[label] += 1                         # increment per-class counter
            save_segment(segment, output_dir, mouse_id,        # save immediately
                         label, class_counters[label])
            # segment is released here -- not accumulated

        del block                                              # release chunk block memory

    del raw                                                    # release MNE file handle

    results[mouse_id] = dict(class_counters)                   # store per-mouse counts

    logger.info(f"  {mouse_id} complete: "
                f"ictal={class_counters['ictal']} "
                f"preictal={class_counters['preictal']} "
                f"postictal={class_counters['postictal']} "
                f"interictal={class_counters['interictal']}")

logger.info("All recordings processed.")


# -- Section 7: Summary statistics ---------------------------------------------

# -- Build summary DataFrame ---------------------------------------------------
rows = []                                                      # accumulate rows for DataFrame
for mid, counts in results.items():                            # iterate over each mouse
    for label, count in counts.items():                        # iterate over each class
        rows.append({"mouse": mid, "class": label, "count": count})

df = pd.DataFrame(rows)                                        # create DataFrame from rows

if not df.empty:
    # -- Pivot for tabular display ---------------------------------------------
    pivot = df.pivot_table(index="mouse", columns="class",     # rows=mice, cols=classes
                           values="count", fill_value=0,
                           aggfunc="sum")
    # Ensure column order
    for col in ["ictal", "preictal", "postictal", "interictal"]:
        if col not in pivot.columns:
            pivot[col] = 0                                     # add missing columns
    pivot = pivot[["ictal", "preictal", "postictal", "interictal"]]  # reorder columns
    pivot.loc["TOTAL"] = pivot.sum()                           # add totals row

    logger.info("Summary table:\n" + pivot.to_string())        # log the full table

    # -- Grouped bar chart -----------------------------------------------------
    mice = [m for m in pivot.index if m != "TOTAL"]            # exclude TOTAL row
    x = np.arange(len(mice))                                   # x positions
    width = 0.2                                                # bar width

    fig, ax = plt.subplots(figsize=(max(8, len(mice) * 1.5), 6))

    colors = {"ictal": "tomato", "preictal": "orange",
              "postictal": "mediumpurple", "interictal": "steelblue"}

    for i, label in enumerate(["ictal", "preictal", "postictal", "interictal"]):
        vals = [pivot.loc[m, label] for m in mice]             # counts for this class
        ax.bar(x + i * width, vals, width, label=label,
               color=colors[label])                            # draw bars

    ax.set_xlabel("Mouse ID")                                  # x-axis label
    ax.set_ylabel("Segment count")                             # y-axis label
    ax.set_title("Segment Distribution by Class and Mouse")    # chart title
    ax.set_xticks(x + 1.5 * width)                            # centre tick marks
    ax.set_xticklabels(mice, rotation=45, ha="right")          # mouse ID labels
    ax.legend()                                                # show legend

    fig.tight_layout()                                         # adjust spacing
    chart_path = output_dir / "summary_chart.png"              # output path
    fig.savefig(chart_path, dpi=150)                           # save chart
    plt.close(fig)                                             # free memory
    logger.info(f"Summary chart saved to {chart_path}")
else:
    logger.warning("No results to summarise.")

logger.info("Pipeline complete.")


# -- End of notebook -----------------------------------------------------------
logger.info("Notebook execution finished.")
