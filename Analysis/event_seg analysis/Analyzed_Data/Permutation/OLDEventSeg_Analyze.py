import os
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ========== CONFIG ==========
MASTER_FILE = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\master_data.csv"
OUTPUT_DIR  = r"C:\Users\Smallwood Lab\friends-event-segmentation\Analysis\event_seg analysis\Permutation\Results"

MIN_PARTICIPANTS = 5      # minimum participants to call a consensus candidate
PEAK_DISTANCE = 5         # minimum distance between peaks (seconds)
WINDOW_SIZE = 4           # tolerance window in seconds (±2 s)
HALF_WINDOW = WINDOW_SIZE // 2

N_PERMUTATIONS = 1000     # number of permutations
SHUFFLE_MODE = "uniform"  # "uniform" or "circular"
RANDOM_SEED = 42
# ===========================

rng = np.random.default_rng(RANDOM_SEED)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def build_binary_timeseries(df: pd.DataFrame, video_name: str) -> pd.DataFrame:
    """Return a time x participants 0/1 matrix at 1-second resolution for one video."""
    sub = df[df["VideoName"] == video_name].copy()
    if sub.empty:
        return pd.DataFrame()

    sub = sub.dropna(subset=["BoundaryTime(s)"])
    sub["Sec"] = sub["BoundaryTime(s)"].round().astype(int)
    sub = sub[sub["Sec"] >= 0]
    if sub.empty:
        return pd.DataFrame()

    duration = int(sub["Sec"].max()) + 1
    participants = sub["ParticipantID"].astype(str).unique()

    ts = pd.DataFrame(0, index=np.arange(duration), columns=participants, dtype=np.uint8)
    for pid, times in sub.groupby("ParticipantID")["Sec"]:
        ts.loc[times.unique(), str(pid)] = 1
    return ts


def dilated_unique_participants_curve(ts: pd.DataFrame, window_size: int) -> np.ndarray:
    """
    For each participant, mark their presses and dilate with a ones kernel of length 'window_size',
    then clip to 0/1. Summing across participants yields, per second,
    the number of UNIQUE participants who pressed within ±(window_size//2) seconds.
    """
    if ts.empty:
        return np.array([])

    kernel = np.ones(window_size, dtype=np.uint8)
    M = ts.values  # shape T x P
    conv = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, M)
    dilated = (conv > 0).astype(np.uint8)
    return dilated.sum(axis=1)


def shuffle_press_times_column(col: np.ndarray, mode: str, rng: np.random.Generator) -> np.ndarray:
    """Shuffle one participant's press vector preserving count."""
    T = col.shape[0]
    out = np.zeros(T, dtype=np.uint8)
    idx = np.where(col == 1)[0]
    k = idx.size
    if k == 0:
        return out
    if mode == "circular":
        shift = rng.integers(0, T)
        out[(idx + shift) % T] = 1
    else:
        new_idx = rng.choice(T, size=k, replace=False)
        out[new_idx] = 1
    return out


def permuted_curve_once(ts: pd.DataFrame, window_size: int, mode: str, rng: np.random.Generator) -> np.ndarray:
    """One permutation: shuffle each participant's presses, then compute the agreement curve."""
    T, P = ts.shape
    shuffled = np.zeros((T, P), dtype=np.uint8)
    M = ts.values
    for j in range(P):
        shuffled[:, j] = shuffle_press_times_column(M[:, j], mode=mode, rng=rng)
    kernel = np.ones(window_size, dtype=np.uint8)
    conv = np.apply_along_axis(lambda x: np.convolve(x, kernel, mode="same"), 0, shuffled)
    dil = (conv > 0).astype(np.uint8)
    return dil.sum(axis=1)


def permutation_test_curves(ts: pd.DataFrame, window_size: int, n_perm: int, mode: str, rng: np.random.Generator):
    """
    Build:
      - null_max: max agreement per permutation (for FWER p-values)
      - null_curves: agreement at every second per permutation (optional for diagnostics)
    """
    T = ts.shape[0]
    null_curves = np.zeros((n_perm, T), dtype=np.uint8)
    for p in range(n_perm):
        null_curves[p, :] = permuted_curve_once(ts, window_size, mode, rng)
    null_max = null_curves.max(axis=1)
    return null_max, null_curves


# ---------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------

def analyze_video(df: pd.DataFrame, video: str, out_dir: str):
    ts = build_binary_timeseries(df, video)
    if ts.empty:
        print(f"No data for {video}")
        return None

    agree = dilated_unique_participants_curve(ts, WINDOW_SIZE)

    peaks, _ = find_peaks(agree, height=MIN_PARTICIPANTS, distance=PEAK_DISTANCE)
    if peaks.size == 0:
        print(f"🎬 {video}: 0 candidates (≥{MIN_PARTICIPANTS})")
        return pd.DataFrame(columns=["VideoName", "BoundaryTime(s)", "NumParticipants"])

    print(f"   Permuting ({N_PERMUTATIONS}x, {SHUFFLE_MODE}) …")
    null_max, _ = permutation_test_curves(ts, WINDOW_SIZE, N_PERMUTATIONS, SHUFFLE_MODE, rng)

    peak_heights = agree[peaks].astype(int)

    # --- FWER p-values ---
    p_fwer = (1 + (null_max[:, None] >= peak_heights[None, :]).sum(axis=0)) / (1 + N_PERMUTATIONS)

    out = pd.DataFrame({
        "VideoName": video,
        "BoundaryTime(s)": peaks.astype(int),
        "NumParticipants": peak_heights.astype(int),
        "p_fwer": p_fwer,
        "Significant_FWER": p_fwer < 0.05
    }).sort_values("BoundaryTime(s)")

    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{video}_agreement_perm.csv")
    out.to_csv(out_file, index=False)

    # Save full agreement curve (for plotting/tuning)
    curve_file = os.path.join(out_dir, f"{video}_agreement_curve.csv")
    pd.DataFrame({"Second": np.arange(len(agree)), "Agreement": agree}).to_csv(curve_file, index=False)

    sig_count = int((out["p_fwer"] < 0.05).sum())
    print(f"🎬 {video}: {len(out)} candidates → {sig_count} FWER-significant")
    print(f"   Saved: {out_file}")
    return out


if __name__ == "__main__":
    df = pd.read_csv(MASTER_FILE)
    results = []
    for video in df["VideoName"].dropna().unique():
        print(f"\n=== Processing {video} ===")
        res = analyze_video(df, video, OUTPUT_DIR)
        if res is not None:
            results.append(res)

    if results:
        combined = pd.concat(results, ignore_index=True)
        combined.to_csv(os.path.join(OUTPUT_DIR, "all_videos_agreement_perm.csv"), index=False)
        print("\n✅ Combined results saved.")
