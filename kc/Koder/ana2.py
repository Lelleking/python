import re
import numpy as np
import glob
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter


############################################
# PARSE EN FIL
############################################

def parse_file(filename):
    chromatics = {}
    current_chromatic = None

    with open(filename, "r", encoding="latin-1") as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        if line.startswith("Chromatic:"):
            current_chromatic = line.split(":")[1].strip()
            chromatics[current_chromatic] = {"time": [], "wells": {}}
            i += 1
            continue

        if line.startswith("Time"):

            if current_chromatic is None:
                i += 1
                continue

            i += 1
            time_values = []

            while i < len(lines):
                tline = lines[i].strip()
                if not re.match(r'^[\d\s]+$', tline):
                    break
                time_values.extend([int(x) for x in tline.split()])
                i += 1

            chromatics[current_chromatic]["time"] = time_values
            continue

        if re.match(r'^[A-H]\d{2}', line) and current_chromatic is not None:
            parts = line.split()
            well = parts[0]
            values = list(map(int, parts[1:]))
            chromatics[current_chromatic]["wells"][well] = values

        i += 1

    return chromatics


############################################
# MERGE FILER
############################################

def merge_files(file_list):

    merged = {}
    time_offset = 0

    for file in file_list:
        data = parse_file(file)

        for chrom in data:

            if chrom not in merged:
                merged[chrom] = {"time": [], "wells": {}}

            original_time = data[chrom]["time"]
            adjusted_time = [t + time_offset for t in original_time]

            merged[chrom]["time"].extend(adjusted_time)

            for well in data[chrom]["wells"]:
                if well not in merged[chrom]["wells"]:
                    merged[chrom]["wells"][well] = []

                merged[chrom]["wells"][well].extend(
                    data[chrom]["wells"][well]
                )

            if adjusted_time:
                time_offset = adjusted_time[-1]

    return merged


############################################
# CHROMATIC VAL
############################################

def select_chromatic(data):

    valid_chromatics = []

    for chrom in data:
        saturated = False
        for well in data[chrom]["wells"]:
            if 260000 in data[chrom]["wells"][well]:
                saturated = True
                break
        if not saturated:
            valid_chromatics.append(chrom)

    all_chromatics = sorted(data.keys(), key=int)

    if valid_chromatics:
        return min(valid_chromatics, key=int)
    else:
        return max(all_chromatics, key=int)


############################################
# 4PL MODEL
############################################

def logistic_4pl(t, A, B, k, t_half):
    z = np.clip(-k * (t - t_half), -500, 500)
    return A + (B - A) / (1 + np.exp(z))


def calculate_halftime_trimmed(time, signal, start_idx, end_idx):

    try:
        t_trim = time[start_idx:end_idx]
        y_trim = signal[start_idx:end_idx]

        if len(t_trim) < 20:
            return np.nan

        A0 = np.percentile(y_trim, 5)
        B0 = np.percentile(y_trim, 95)
        k0 = 0.5
        t_half0 = t_trim[len(t_trim)//2]

        bounds = (
            [0, 0, 0, 0],
            [np.inf, np.inf, 10, np.max(t_trim)]
        )

        popt, _ = curve_fit(
            logistic_4pl,
            t_trim,
            y_trim,
            p0=[A0, B0, k0, t_half0],
            bounds=bounds,
            maxfev=20000
        )

        return popt[3]

    except:
        return np.nan


############################################
# FEATURE EXTRACTION (ROBUST)
############################################

def extract_features_from_current_folder():

    files = sorted(glob.glob("*.csv"))
    if not files:
        return {}

    data = merge_files(files)
    if not data:
        return {}

    selected = select_chromatic(data)
    time = np.array(data[selected]["time"]) / 3600.0
    feature_dict = {}

    for well, raw_signal in data[selected]["wells"].items():

        signal = np.array(raw_signal, dtype=float)

        if len(signal) != len(time):
            continue

        N = len(signal)
        if N < 50:
            continue

        # 1️⃣ Smooth signal
        window = min(101, N-1 if N % 2 == 0 else N)
        if window < 11:
            continue

        smooth_signal = savgol_filter(signal, window_length=window, polyorder=3)

        # 2️⃣ Baseline (same interval-logic as app.py estimate_baseline_plateau_from_signal)
        baseline_end_idx = max(1, int(0.05 * N))
        baseline_y_slice = smooth_signal[:baseline_end_idx]
        b_min, b_max = np.min(baseline_y_slice), np.max(baseline_y_slice)

        last_idx_in_baseline = 0
        for i in range(N):
            if b_min <= smooth_signal[i] <= b_max:
                last_idx_in_baseline = i
            else:
                if i > baseline_end_idx:
                    break
        baseline = float(smooth_signal[last_idx_in_baseline])
        noise = float(np.std(baseline_y_slice))

        # 3️⃣ Plateau (same interval-logic as app.py estimate_baseline_plateau_from_signal)
        plateau_start_idx = max(0, N - max(1, int(0.05 * N)))
        plateau_y_slice = smooth_signal[plateau_start_idx:]
        p_min, p_max = np.min(plateau_y_slice), np.max(plateau_y_slice)

        first_idx_in_plateau = N - 1
        for i in range(N - 1, -1, -1):
            if p_min <= smooth_signal[i] <= p_max:
                first_idx_in_plateau = i
            else:
                if i < plateau_start_idx:
                    break
        plateau = float(smooth_signal[first_idx_in_plateau])

        amplitude = plateau - baseline
        max_signal = np.max(signal)

        if amplitude <= 0:
            continue

        # 4️⃣ Windowed slope
        W = max(20, int(0.03 * N))

        slopes = np.array([
            (smooth_signal[i+W] - smooth_signal[i]) /
            (time[i+W] - time[i])
            for i in range(N - W)
        ])

        if len(slopes) == 0:
            continue

        max_slope = np.max(slopes)
        if max_slope <= 0:
            continue

        # 5️⃣ Identify sigmoid region (longest continuous slope block)
        slope_threshold = 0.1 * max_slope
        active = slopes > slope_threshold

        blocks = []
        start = None

        for i, val in enumerate(active):
            if val and start is None:
                start = i
            elif not val and start is not None:
                blocks.append((start, i))
                start = None

        if start is not None:
            blocks.append((start, len(active)))

        if not blocks:
            continue

        lengths = [b[1] - b[0] for b in blocks]
        longest = blocks[np.argmax(lengths)]

        if (longest[1] - longest[0]) < int(0.05 * N):
            continue

        start_idx = longest[0]
        end_idx = longest[1] + W

        # Padding
        pad = int(0.03 * N)
        start_idx = max(0, start_idx - pad)
        end_idx = min(N, end_idx + pad)

        if end_idx - start_idx < 20:
            continue

        # 6️⃣ Normalized signal
        norm_signal = (smooth_signal - baseline) / amplitude

        time_10 = time[np.argmax(norm_signal >= 0.1)] if np.any(norm_signal >= 0.1) else 0
        time_50 = time[np.argmax(norm_signal >= 0.5)] if np.any(norm_signal >= 0.5) else 0
        time_90 = time[np.argmax(norm_signal >= 0.9)] if np.any(norm_signal >= 0.9) else 0

        # 7️⃣ Tangent-based lag time at max slope point
        i_max = int(np.argmax(slopes))
        i_tan = int(min(N - 1, i_max + (W // 2)))
        t_max = float(time[i_tan])
        y_max = float(smooth_signal[i_tan])
        lag_time = t_max - ((y_max - baseline) / max_slope)
        if not np.isfinite(lag_time) or lag_time < 0:
            lag_time = 0.0

        # 8️⃣ Biphasic ratio: second slope peak after a dip below 50% of peak1
        peak1 = float(max_slope)
        peak1_idx = i_max
        min_sep = max(3, int(0.05 * len(slopes)))
        peak2 = 0.0
        for j in range(peak1_idx + min_sep, len(slopes)):
            between = slopes[peak1_idx:j + 1]
            if between.size == 0:
                continue
            if np.min(between) < 0.5 * peak1 and slopes[j] > peak2:
                peak2 = float(slopes[j])
        biphasic_ratio = (peak2 / peak1) if peak2 > 0 else 0.0

        # 9️⃣ Trimmed 4PL fit
        t_half_fit = calculate_halftime_trimmed(
            time,
            smooth_signal,
            start_idx,
            end_idx
        )

        if np.isnan(t_half_fit):
            t_half_fit = 0

        feature_dict[well] = {
            "amplitude": amplitude,
            "max_slope": max_slope,
            "lag_time": lag_time,
            "biphasic_ratio": biphasic_ratio,
            "baseline_noise": noise,
            "baseline_level": baseline,
            "plateau_level": plateau,
            "time_10": time_10,
            "time_50": time_50,
            "time_90": time_90,
            "t_half_fit": t_half_fit,
            "max_signal": max_signal
        }

    return feature_dict
############################################
# RULE-BASED AGGREGATION
############################################

def rule_based_aggregation(features):

    max_signal = features["max_signal"]

    # HARD PHYSICAL CUTOFF
    if max_signal < 10000:
        return False

    amplitude = features["amplitude"]
    slope = features["max_slope"]
    noise = features["baseline_noise"]
    t10 = features["time_10"]
    t90 = features["time_90"]

    cond1 = amplitude > 5000
    cond2 = slope > 10
    cond3 = (t90 - t10) > 1
    cond4 = noise < amplitude * 0.4

    score = sum([cond1, cond2, cond3, cond4])

    return score >= 3
