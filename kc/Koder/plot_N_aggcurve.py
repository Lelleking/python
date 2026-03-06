import sys
import os
import re
import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===============================
# PARSE EN FIL (Chromatic)
# ===============================
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

        if line.startswith("Time") and current_chromatic is not None:
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


# ===============================
# MERGE FILER
# ===============================
def merge_files(file_list):
    merged = {}

    for file in file_list:
        data = parse_file(file)

        for chrom in data:
            if chrom not in merged:
                merged[chrom] = {"time": [], "wells": {}}

            original_time = data[chrom]["time"]

            offset = merged[chrom]["time"][-1] if merged[chrom]["time"] else 0
            adjusted_time = [t + offset for t in original_time]
            merged[chrom]["time"].extend(adjusted_time)

            for well in data[chrom]["wells"]:
                merged[chrom]["wells"].setdefault(well, [])
                merged[chrom]["wells"][well].extend(
                    data[chrom]["wells"][well]
                )

    return merged


# ===============================
# VÄLJ CHROMATIC
# ===============================
def select_chromatic(data):
    valid = []

    for chrom in data:
        saturated = any(
            260000 in data[chrom]["wells"][well]
            for well in data[chrom]["wells"]
        )
        if not saturated:
            valid.append(chrom)

    all_chroms = sorted(data.keys(), key=int)

    return min(valid, key=int) if valid else max(all_chroms, key=int)


# ===============================
# ARGUMENT
# ===============================
if len(sys.argv) < 3:
    print("Usage: python plot_N_aggcurve.py <labname> <antibody>")
    sys.exit(1)

lab = sys.argv[1]
antibody = sys.argv[2]

# ===============================
# PATH
# ===============================
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
lab_dir = os.path.join(base_dir, lab)

files = sorted(glob.glob(os.path.join(lab_dir, "*_file*.csv")))
if not files:
    print("No raw files found.")
    sys.exit(1)

data = merge_files(files)
selected = select_chromatic(data)
print("Selected chromatic:", selected)

time = np.array(data[selected]["time"])
wells_dict = data[selected]["wells"]

# sekunder → timmar
time = (time - time[0]) / 3600

# ===============================
# MAP
# ===============================
map_df = pd.read_csv(os.path.join(lab_dir, f"{lab}_map.csv"))
selected_rows = map_df[
    (map_df["antibody"] == antibody) &
    (map_df["half_time"].notna())
]

if selected_rows.empty:
    print(f"No wells found for {antibody}")
    sys.exit(1)

# ===============================
# GRUPPERA PER KONCENTRATION
# ===============================
conc_groups = {}
for _, row in selected_rows.iterrows():
    well = f"{row['well'][0]}{int(row['well'][1:]):02d}"
    conc_groups.setdefault(row["conc_uM"], []).append(well)

# ===============================
# MATCHA ANTAL REPLIKAT (0 µM)
# ===============================
rep_counts = [
    len(wells) for conc, wells in conc_groups.items()
    if conc != 0
]

if rep_counts:
    target_reps = min(rep_counts)
    if 0 in conc_groups:
        conc_groups[0] = conc_groups[0][:target_reps]

sorted_concs = sorted(conc_groups.keys())

# ===============================
# COLORBLIND-SAFE PALETTE
# ===============================
OKABE_ITO = [
    "#000000",  # 0 µM
    "#E69F00",
    "#56B4E9",
    "#009E73",
    "#F0E442",
    "#0072B2",
    "#D55E00",
    "#CC79A7",
    "#999999",
    "#8B4513",
]

color_map = {}
color_index = 1

for conc in sorted_concs:
    if conc == 0:
        color_map[conc] = "#000000"
    else:
        color_map[conc] = OKABE_ITO[color_index % len(OKABE_ITO)]
        color_index += 1

# ===============================
# PLOT FULL RANGE
# ===============================
plt.figure(figsize=(7,5))

for conc in sorted_concs:
    color = color_map[conc]
    wells_in_group = conc_groups[conc]

    for idx, w in enumerate(wells_in_group):
        if w in wells_dict:
            raw = np.array(wells_dict[w], dtype=float)

            min_val = np.min(raw)
            max_val = np.max(raw)
            if max_val - min_val == 0:
                continue

            norm = (raw - min_val) / (max_val - min_val)

            plt.plot(
                time,
                norm,
                color=color,
                alpha=0.7,
                linewidth=2 if conc == 0 else 1.5,
                label=f"{conc} µM" if idx == 0 else None
            )

plt.xlabel("Time (h)")
plt.ylabel("Normalized fluorescence (0–1)")
plt.title(f"{antibody} (normalized)")
plt.legend(title="Concentration", loc="upper left")
plt.grid(True, linestyle="--", linewidth=0.5)
plt.tight_layout()
plt.show()

# ===============================
# SELECT MAX TIME
# ===============================
cutoff_input = input(
    "Enter max time in hours to replot (press Enter to keep full range): "
)

if cutoff_input.strip() != "":
    try:
        cutoff = float(cutoff_input)
        mask = time <= cutoff
        time_cut = time[mask]

        plt.figure(figsize=(7,5))

        for conc in sorted_concs:
            color = color_map[conc]
            wells_in_group = conc_groups[conc]

            for idx, w in enumerate(wells_in_group):
                if w in wells_dict:
                    raw = np.array(wells_dict[w], dtype=float)
                    raw_cut = raw[mask]

                    min_val = np.min(raw_cut)
                    max_val = np.max(raw_cut)
                    if max_val - min_val == 0:
                        continue

                    norm_cut = (raw_cut - min_val) / (max_val - min_val)

                    plt.plot(
                        time_cut,
                        norm_cut,
                        color=color,
                        alpha=0.7,
                        linewidth=2 if conc == 0 else 1.5,
                        label=f"{conc} µM" if idx == 0 else None
                    )

        plt.xlabel("Time (h)")
        plt.ylabel("Normalized fluorescence (0–1)")
        plt.title(f"{antibody} (normalized, 0–{cutoff} h)")
        plt.legend(title="Concentration", loc="upper left")
        plt.grid(True, linestyle="--", linewidth=0.5)
        plt.tight_layout()
        plt.show()

    except ValueError:
        print("Invalid number. Keeping full range.")