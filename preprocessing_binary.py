# -- Section 2: Library imports ------------------------------------------------

import logging                                      # structured logging to file and console
import mne                                          # read EDF headers and load data on demand
import numpy as np                                  # numerical array operations
import pandas as pd                                 # read Excel annotation files
import matplotlib                                   # configure non-interactive backend for cluster use
matplotlib.use('Agg')                               # use non-interactive backend (no display needed)
import matplotlib.pyplot as plt                     # bar chart for summary statistics
from pathlib import Path                            # cross-platform path handling
from scipy.signal import iirnotch, filtfilt         # notch filter design and zero-phase filtering
from scipy.stats import median_abs_deviation        # robust scale estimator for MAD-based z-score

# -- Section 3: Configuration -- edit these values before running ---------------

# --- Paths (EDF files and annotation files are in SEPARATE folders) ---
edf_dir        = Path(r'/home/people/22206468/scratch/EEG_TRAINING')           # folder containing all .edf files
annotation_dir = Path(r'/home/people/22206468/scratch/seizure_times_updated')    # folder containing all matching .xlsx files
output_dir     = Path(r'/home/people/22206468/scratch/TRAIN_DATA')       # root output folder; seizure/ and non_seizure/ created inside

# --- Signal parameters ---
fs      = 500    # sampling rate in Hz -- must match the EDF file header
win_len = 2500   # window length in samples: 5 s * 500 Hz = 2500 samples per segment
step    = 1250   # step size in samples: 2.5 s * 500 Hz = 1250 samples (50% overlap)

# --- Chunking -- controls peak RAM usage ---
segments_per_chunk = 40  # number of segments grouped into one RAM-loaded block (K)

# --- Preprocessing ---
notch_freq = 50.0  # powerline interference frequency to suppress (Hz)
notch_Q    = 30     # Q-factor of the notch filter: higher = narrower, more targeted notch

# --- Logging setup -- writes to file AND console -------------------------------
output_dir.mkdir(parents=True, exist_ok=True)  # ensure output folder exists before creating log file

log = logging.getLogger('eeg_pipeline')        # create a named logger for this pipeline
log.setLevel(logging.INFO)                     # set minimum log level to INFO
log.handlers.clear()                           # remove any handlers from previous runs (re-run safe)

file_handler = logging.FileHandler(            # handler that writes log messages to a file
    output_dir / 'pipeline.log',               # log file lives next to the output segments
    mode='a',                                  # append mode -- new runs add to the same log file
    encoding='utf-8'                           # ensure Unicode characters are written correctly
)
file_handler.setFormatter(logging.Formatter(   # format: timestamp | message
    '%(asctime)s | %(message)s'                # asctime gives YYYY-MM-DD HH:MM:SS,mmm
))

console_handler = logging.StreamHandler()      # handler that prints log messages to stdout/console
console_handler.setFormatter(logging.Formatter(  # same format as the file handler
    '%(asctime)s | %(message)s'
))

log.addHandler(file_handler)                   # attach the file handler to the logger
log.addHandler(console_handler)                # attach the console handler to the logger

log.info('Pipeline configuration loaded')      # confirm logging is working
log.info(f'  EDF dir:        {edf_dir}')       # log the EDF directory path
log.info(f'  Annotation dir: {annotation_dir}')  # log the annotation directory path
log.info(f'  Output dir:     {output_dir}')    # log the output directory path
log.info(f'  Log file:       {output_dir / "pipeline.log"}')  # log the log file path itself

# -- load_annotations -- parse Excel seizure times into seconds -----------------

def load_annotations(xlsx_path, recording_start_dt):
    """
    Read a seizure annotation Excel file and convert timestamps to seconds
    relative to the EDF recording start.

    Parameters
    ----------
    xlsx_path : pathlib.Path
        Path to the .xlsx annotation file with columns 'start_time' and
        'end_time' in DD/MM/YYYY HH:MM:SS.fff format.
    recording_start_dt : datetime.datetime
        Timezone-naive recording start datetime from the EDF header.

    Returns
    -------
    intervals : list of tuple(float, float)
        Each tuple is (start_sec, end_sec) relative to recording start.
    """

    df = pd.read_excel(xlsx_path)  # load the Excel file into a DataFrame

    df['start_time'] = pd.to_datetime(  # parse the start_time column as datetime objects
        df['start_time'],               # column containing onset timestamp strings
        dayfirst=True                   # interpret DD/MM/YYYY format correctly
    )

    df['end_time'] = pd.to_datetime(    # parse the end_time column as datetime objects
        df['end_time'],                 # column containing offset timestamp strings
        dayfirst=True                   # interpret DD/MM/YYYY format correctly
    )

    intervals = []  # accumulate (start_sec, end_sec) tuples for each seizure

    for _, row in df.iterrows():  # iterate over each annotation row

        start_dt = row['start_time'].to_pydatetime()  # convert pandas Timestamp to Python datetime
        end_dt   = row['end_time'].to_pydatetime()    # convert pandas Timestamp to Python datetime

        start_dt = start_dt.replace(tzinfo=None)  # strip timezone info for safe subtraction
        end_dt   = end_dt.replace(tzinfo=None)    # strip timezone info for safe subtraction

        start_sec = (start_dt - recording_start_dt).total_seconds()  # seconds from recording start to seizure onset
        end_sec   = (end_dt   - recording_start_dt).total_seconds()  # seconds from recording start to seizure offset

        intervals.append((start_sec, end_sec))  # store this seizure interval as a tuple

    return intervals  # return all seizure intervals as seconds relative to recording start

# -- compute_segment_grid -- build the full list of segment start indices -------

def compute_segment_grid(n_samples, win_len, step):
    """
    Pre-compute every valid segment start index for the entire recording.

    A start index i is valid when i + win_len <= n_samples, ensuring that
    every segment contains exactly win_len samples with no truncation.

    Parameters
    ----------
    n_samples : int
        Total number of samples in the recording (from the EDF header).
    win_len : int
        Segment length in samples (e.g. 2500 for 5 s at 500 Hz).
    step : int
        Step size between consecutive segment starts (e.g. 1250 for 50% overlap).

    Returns
    -------
    starts : np.ndarray, dtype int64
        1-D array of every valid segment start index.
    """

    last_valid = n_samples - win_len  # largest sample index where a full window still fits

    starts = np.arange(             # generate evenly spaced start indices
        0,                          # first segment begins at sample 0
        last_valid + 1,             # +1 because arange excludes the upper bound
        step,                       # spacing equals the configured step size
        dtype=np.int64              # integer indices for array slicing
    )

    return starts  # return the complete grid of segment start indices


# -- Preprocessing functions -- notch filter, DC removal, robust z-score --------

def apply_notch_filter(signal, fs, freq, Q):
    """
    Remove powerline interference with a zero-phase IIR notch filter.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D raw EEG signal (one chunk block).
    fs : float
        Sampling rate in Hz.
    freq : float
        Centre frequency of the notch in Hz (e.g. 50.0).
    Q : float
        Quality factor -- higher values produce a narrower notch.

    Returns
    -------
    filtered : np.ndarray, shape (n,)
        Signal with the specified frequency suppressed.
    """

    w0 = freq / (fs / 2.0)            # normalise notch frequency to Nyquist (must be in (0, 1))
    b, a = iirnotch(w0, Q)            # design the second-order IIR notch filter coefficients
    filtered = filtfilt(b, a, signal)  # apply forward-backward (zero-phase) filtering

    return filtered  # return the notch-filtered signal


def remove_dc(signal):
    """
    Remove the DC offset by subtracting the signal mean.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D EEG signal (already notch-filtered).

    Returns
    -------
    centred : np.ndarray, shape (n,)
        Signal with zero mean.
    """

    dc = np.mean(signal)   # compute the DC component (mean amplitude of the chunk)
    centred = signal - dc  # subtract it from every sample

    return centred  # return the DC-corrected signal


def robust_zscore(signal):
    """
    Normalise using the robust z-score: z = (x - median) / (1.4826 * MAD).

    The 1.4826 factor makes the denominator equal to the standard deviation
    for Gaussian data. If MAD = 0 (constant signal), the input is returned
    unchanged to avoid division by zero.

    Parameters
    ----------
    signal : np.ndarray, shape (n,)
        1-D EEG signal (already notch-filtered and DC-corrected).

    Returns
    -------
    normed : np.ndarray, shape (n,)
        Robustly normalised signal.
    """

    med = np.median(signal)                       # robust location estimate (centre of the distribution)
    mad = median_abs_deviation(signal, scale=1)   # MAD without the 1.4826 scaling factor

    if mad == 0.0:               # constant signal -- cannot normalise
        return signal            # return unchanged to avoid division by zero

    normed = (signal - med) / (1.4826 * mad)  # apply the robust z-score formula

    return normed  # return the normalised signal


def preprocess_chunk(chunk, fs, notch_freq, notch_Q):
    """
    Apply the full three-step preprocessing pipeline to one chunk block.

    Steps: (1) notch filter  (2) DC removal  (3) robust z-score.

    Parameters
    ----------
    chunk : np.ndarray, shape (n,)
        Raw 1-D signal for one chunk loaded from the EDF file.
    fs : float
        Sampling rate in Hz.
    notch_freq : float
        Notch filter centre frequency in Hz.
    notch_Q : float
        Notch filter Q-factor.

    Returns
    -------
    processed : np.ndarray, shape (n,)
        Fully preprocessed chunk ready for segment extraction.
    """

    step1 = apply_notch_filter(chunk, fs, notch_freq, notch_Q)  # step 1: remove 50 Hz powerline
    step2 = remove_dc(step1)                                     # step 2: subtract DC offset
    processed = robust_zscore(step2)                              # step 3: robust z-score normalisation

    return processed  # return the fully preprocessed chunk

# -- Labelling and saving functions --------------------------------------------

def is_ictal(seg_start_sec, seg_end_sec, seizure_intervals):
    """
    Determine whether a segment overlaps with any seizure interval.

    Overlap exists when:
        seg_start_sec < seizure_end_sec  AND  seg_end_sec > seizure_start_sec

    Parameters
    ----------
    seg_start_sec : float
        Segment start time in seconds relative to recording start.
    seg_end_sec : float
        Segment end time in seconds relative to recording start.
    seizure_intervals : list of tuple(float, float)
        Each tuple is (seizure_start_sec, seizure_end_sec).

    Returns
    -------
    overlaps : bool
        True if the segment overlaps with at least one seizure.
    """

    for sz_start, sz_end in seizure_intervals:  # check each annotated seizure

        if seg_start_sec < sz_end and seg_end_sec > sz_start:  # overlap condition
            return True  # at least one seizure overlaps -- label as ictal

    return False  # no overlap found with any seizure -- label as non-ictal


def save_segment(segment, output_dir, mouse_id, label, index):
    """
    Save a single segment as a .npy file in the appropriate class folder.

    Directory structure:
        output_dir/seizure/{mouse_id}_ictal_{index:05d}.npy
        output_dir/non_seizure/{mouse_id}_nonictal_{index:05d}.npy

    Parameters
    ----------
    segment : np.ndarray, shape (win_len,)
        The preprocessed 1-D segment to save.
    output_dir : pathlib.Path
        Root output directory.
    mouse_id : str
        Subject identifier (e.g. 'm1').
    label : str
        Either 'ictal' or 'nonictal'.
    index : int
        1-based index for this class and mouse.

    Returns
    -------
    filepath : pathlib.Path
        Full path to the saved .npy file.
    """

    if label == 'ictal':                                    # seizure segments go into seizure/
        class_dir = output_dir / 'seizure'                  # build the seizure subfolder path
        filename  = f'{mouse_id}_ictal_{index:05d}.npy'     # e.g. m1_ictal_00001.npy
    else:                                                   # non-seizure segments go into non_seizure/
        class_dir = output_dir / 'non_seizure'              # build the non_seizure subfolder path
        filename  = f'{mouse_id}_nonictal_{index:05d}.npy'  # e.g. m1_nonictal_00001.npy

    class_dir.mkdir(parents=True, exist_ok=True)            # create the folder if it does not exist

    filepath = class_dir / filename                         # full path to the output file
    np.save(filepath, segment)                              # write the 1-D NumPy array to disk

    return filepath  # return the path for logging or verification


# -- Main loop -- iterate over all EDF files, chunk-process, label, save --------

results = {}  # accumulate per-mouse segment counts: {mouse_id: {'ictal': int, 'nonictal': int}}

edf_files = sorted(edf_dir.glob('*.edf'))  # find every EDF file in the EDF directory
log.info(f'Found {len(edf_files)} EDF file(s) in {edf_dir}')  # report the total number of recordings

for edf_path in edf_files:  # iterate over each mouse recording

    mouse_id  = edf_path.stem                             # extract mouse ID from filename (e.g. 'm1' from 'm1.edf')
    xlsx_path = annotation_dir / f'{mouse_id}_xlsx.xlsx'       # look up the annotation in the SEPARATE folder

    if not xlsx_path.exists():                            # guard: skip if no matching annotation file
        log.warning(f'[SKIP] {mouse_id}: no annotation file found at {xlsx_path}')
        continue                                          # move on to the next EDF file

    # -- Open EDF header only (preload=False keeps signal on disk) ---------
    raw = mne.io.read_raw_edf(   # open the EDF file
        str(edf_path),           # MNE expects a string path
        preload=False,           # do NOT load signal data into RAM yet
        verbose=False            # suppress MNE's internal log messages
    )

    # -- Discard non-EEG channels (e.g. Activity, accelerometry, respiration, temperature) --
    # This eliminates mixed-sampling-frequency warnings and ensures only the EEG channel remains.
    raw.pick('EEG')  # keep only channels with 'EEG' in their type (case-insensitive match)

    # -- Guard: ensure at least one EEG channel survived the pick ----------
    if len(raw.ch_names) == 0:
        raise ValueError(
            f"No EEG channels identified in {edf_path} after pick('eeg'). "
            f"Consider using raw.pick([<explicit_channel_name>]) as a fallback."
        )
    log.info(f"[{mouse_id}] EEG channel retained: {raw.ch_names} @ {raw.info['sfreq']} Hz")

    n_samples = int(raw.n_times)                # total samples in the recording (from header)
    fs_file   = raw.info['sfreq']               # sampling rate reported in the EDF header
    meas_date = raw.info['meas_date']           # recording start datetime (may have timezone)
    rec_start = meas_date.replace(tzinfo=None)  # make timezone-naive for annotation comparison

    log.info(f'[{mouse_id}] Loaded header: {n_samples} samples @ {fs_file} Hz '
             f'({n_samples / fs_file:.1f} s)')  # report file stats

    # -- Load seizure annotations from the separate annotation folder ------
    seizure_intervals = load_annotations(xlsx_path, rec_start)  # list of (start_sec, end_sec)
    log.info(f'[{mouse_id}] {len(seizure_intervals)} seizure interval(s) loaded from {xlsx_path.name}')

    # -- Phase 1: Compute the segment grid (no data loaded) ---------------
    segment_starts = compute_segment_grid(n_samples, win_len, step)  # all valid segment start indices
    n_segs = len(segment_starts)                # total number of segments on the grid
    log.info(f'[{mouse_id}] Segment grid: {n_segs} segments '
             f'(win={win_len}, step={step})')   # confirm grid dimensions

    # -- Phase 2: Group grid into RAM-safe chunks and process --------------
    chunk_groups = [                            # split the grid into groups of K segments
        segment_starts[i : i + segments_per_chunk]  # each group is a contiguous slice of the grid
        for i in range(0, n_segs, segments_per_chunk)  # step through in groups of K
    ]
    n_chunks = len(chunk_groups)                # total number of chunks for this recording

    ictal_count    = 0  # per-mouse running count of ictal segments (1-indexed in filenames)
    nonictal_count = 0  # per-mouse running count of non-ictal segments (1-indexed in filenames)

    for chunk_idx, chunk_seg_starts in enumerate(chunk_groups):  # iterate over each chunk

        chunk_start = int(chunk_seg_starts[0])            # first sample index in this chunk
        chunk_end   = int(chunk_seg_starts[-1]) + win_len # one-past-last sample in this chunk

        raw_block = raw.get_data(    # load ONLY this chunk's samples from disk
            start=chunk_start,       # first sample to read
            stop=chunk_end           # one-past-last sample to read (exclusive)
        )
        block = raw_block.squeeze().astype(np.float64)  # flatten to 1-D and ensure float64 precision

        block = preprocess_chunk(    # apply the full preprocessing pipeline
            block, fs, notch_freq, notch_Q  # notch -> DC removal -> robust z-score
        )

        log.info(f'  Chunk {chunk_idx + 1:3d}/{n_chunks}: '
                 f'samples [{chunk_start}:{chunk_end}]  '
                 f'({len(chunk_seg_starts)} segments)')  # report chunk progress

        for seg_start_idx in chunk_seg_starts:  # iterate over each segment in this chunk

            local_start = int(seg_start_idx) - chunk_start          # offset within the loaded block
            segment = block[local_start : local_start + win_len]    # extract the 2500-sample window
            segment = segment.astype(np.float32)                    # cast to float32 to halve file size

            seg_start_sec = seg_start_idx / fs                      # segment start in seconds
            seg_end_sec   = (seg_start_idx + win_len) / fs          # segment end in seconds

            if is_ictal(seg_start_sec, seg_end_sec, seizure_intervals):  # check seizure overlap
                ictal_count += 1                                    # increment the 1-based ictal counter
                save_segment(segment, output_dir, mouse_id, 'ictal', ictal_count)
            else:                                                   # no overlap -- non-ictal
                nonictal_count += 1                                 # increment the 1-based non-ictal counter
                save_segment(segment, output_dir, mouse_id, 'nonictal', nonictal_count)

    del raw  # close the file handle and release the memory-mapped EDF

    results[mouse_id] = {            # store per-mouse segment counts
        'ictal': ictal_count,        # total ictal segments for this mouse
        'nonictal': nonictal_count   # total non-ictal segments for this mouse
    }

    log.info(f'[{mouse_id}] Done: {ictal_count} ictal | '
             f'{nonictal_count} non-ictal segments saved')  # per-file summary

log.info('All recordings processed.')  # signal that the main loop is complete



# -- Summary table and bar chart ----------------------------------------------

# -- Formatted text table -----------------------------------------------------
header = f'{"Mouse":<12s} {"Ictal":>8s} {"Non-ictal":>12s} {"Total":>8s}'  # column headers
sep    = '-' * len(header)  # separator line matching header width

log.info(header)  # log the header row
log.info(sep)     # log the separator

total_ictal    = 0  # accumulate grand total of ictal segments across all mice
total_nonictal = 0  # accumulate grand total of non-ictal segments across all mice

for mouse_id, counts in results.items():  # iterate over each mouse's results

    ic    = counts['ictal']     # ictal count for this mouse
    ni    = counts['nonictal']  # non-ictal count for this mouse
    total = ic + ni             # total segments for this mouse

    log.info(f'{mouse_id:<12s} {ic:>8d} {ni:>12d} {total:>8d}')  # log one row per mouse

    total_ictal    += ic  # add to grand ictal total
    total_nonictal += ni  # add to grand non-ictal total

log.info(sep)  # log separator before totals row
log.info(f'{"TOTAL":<12s} {total_ictal:>8d} {total_nonictal:>12d} '
         f'{total_ictal + total_nonictal:>8d}')  # log grand totals

# -- Grouped bar chart (saved to file -- no display needed on cluster) ----------
mice    = list(results.keys())                     # list of mouse IDs in processing order
ic_vals = [results[m]['ictal'] for m in mice]      # ictal counts per mouse
ni_vals = [results[m]['nonictal'] for m in mice]   # non-ictal counts per mouse

x     = np.arange(len(mice))  # x positions for the grouped bars
width = 0.35                   # width of each bar

fig, ax = plt.subplots(figsize=(max(6, len(mice) * 1.2), 5))  # scale width to number of mice

bars_ic = ax.bar(x - width / 2, ic_vals, width,   # draw ictal bars shifted left
                 label='Ictal', color='tomato')     # red for seizure

bars_ni = ax.bar(x + width / 2, ni_vals, width,   # draw non-ictal bars shifted right
                 label='Non-ictal', color='steelblue')  # blue for non-seizure

ax.set_xlabel('Mouse ID')                           # label the x-axis
ax.set_ylabel('Segment count')                      # label the y-axis
ax.set_title('Ictal vs Non-ictal Segments per Mouse')  # chart title
ax.set_xticks(x)                                    # place ticks at each mouse position
ax.set_xticklabels(mice)                            # label ticks with mouse IDs
ax.legend()                                         # show the legend

ax.bar_label(bars_ic, padding=3)   # label each ictal bar with its count
ax.bar_label(bars_ni, padding=3)   # label each non-ictal bar with its count

fig.tight_layout()                                           # adjust spacing so nothing is clipped
chart_path = output_dir / 'summary_chart.png'                # save path for the bar chart image
fig.savefig(chart_path, dpi=150)                             # save the chart as a PNG file
plt.close(fig)                                               # close the figure to free memory
log.info(f'Summary chart saved to {chart_path}')             # confirm chart was saved
