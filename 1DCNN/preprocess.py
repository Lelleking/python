import os
import re
import math
import glob

RAW_FOLDER = "raw data"
COMPRESSED_FOLDER = "compressed data"


############################################
# PARSE ONE RAW CSV FILE
# Identical logic to amyloconvert.py
############################################

def parse_file(filepath):
    chromatics = {}
    current_chromatic = None

    with open(filepath, "r", encoding="latin-1") as f:
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
                if not re.match(r'^[\d\s,]+$', tline):
                    break
                time_values.extend([int(x) for x in re.findall(r'\d+', tline)])
                i += 1
            chromatics[current_chromatic]["time"] = time_values
            continue

        if re.match(r'^[A-H]\d{2}', line) and current_chromatic is not None:
            parts = line.split()
            well = parts[0].rstrip(':')
            values = [int(v.rstrip(',')) for v in parts[1:]]
            chromatics[current_chromatic]["wells"][well] = values

        i += 1

    return chromatics


############################################
# MERGE FILE PARTS IN ORDER
# Handles file1, file2, file3, ... with
# time offset correction per chromatic
############################################

def merge_files(file_list):
    merged = {}

    for filepath in file_list:
        data = parse_file(filepath)

        for chrom in data:
            if chrom not in merged:
                merged[chrom] = {"time": [], "wells": {}}

            original_time = data[chrom]["time"]

            if merged[chrom]["time"]:
                time_offset = merged[chrom]["time"][-1]
            else:
                time_offset = 0

            adjusted_time = [t + time_offset for t in original_time]
            merged[chrom]["time"].extend(adjusted_time)

            for well in data[chrom]["wells"]:
                if well not in merged[chrom]["wells"]:
                    merged[chrom]["wells"][well] = []
                merged[chrom]["wells"][well].extend(data[chrom]["wells"][well])

    return merged


############################################
# SELECT BEST CHROMATIC
# Picks lowest-numbered chromatic with no
# saturated wells (value 260000).
# Falls back to highest-numbered if all saturated.
############################################

def select_chromatic(data):
    valid = []

    for chrom in data:
        saturated = False
        for well in data[chrom]["wells"]:
            if 260000 in data[chrom]["wells"][well]:
                saturated = True
                break
        if not saturated:
            valid.append(chrom)

    all_chroms = sorted(data.keys(), key=int)

    if valid:
        return min(valid, key=int)
    else:
        return max(all_chroms, key=int)


############################################
# GROUP RAW FILES BY EXPERIMENT PREFIX
# Groups files sharing the same prefix
# (everything before "file1", "file2" etc.)
# and sorts each group by file number.
############################################

def group_files_by_experiment(raw_folder):
    all_files = sorted(
        glob.glob(os.path.join(raw_folder, "*.csv")) +
        glob.glob(os.path.join(raw_folder, "*.DAT")) +
        glob.glob(os.path.join(raw_folder, "*.dat"))
    )

    groups = {}

    for filepath in all_files:
        fname = os.path.basename(filepath)

        match = re.search(r'(file\d+)', fname, re.IGNORECASE)
        if match:
            file_tag = match.group(1)
            prefix = fname[:fname.lower().index(file_tag.lower())]
            prefix = prefix.rstrip('_- ')
        else:
            prefix = fname.replace('.csv', '').rstrip('_- ')

        if prefix not in groups:
            groups[prefix] = []
        groups[prefix].append(filepath)

    for prefix in groups:
        def sort_key(fp):
            m = re.search(r'file(\d+)', os.path.basename(fp), re.IGNORECASE)
            return int(m.group(1)) if m else 0
        groups[prefix].sort(key=sort_key)

    return groups


############################################
# SAVE COMPRESSED .TXT
# One tab-separated file per experiment.
# Time column in hours, then one col per well.
############################################

def save_compressed(prefix, time_list, wells_dict, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    out_name = os.path.join(output_folder, f"{prefix}_compressed.txt")
    wells_sorted = sorted(wells_dict.keys())
    start_time = time_list[0]

    with open(out_name, "w") as f:
        header = "Time\t" + "\t".join(wells_sorted) + "\n"
        f.write(header)

        for i in range(len(time_list)):
            time_hours = (time_list[i] - start_time) / 3600
            row = [f"{time_hours:.6f}"]
            for well in wells_sorted:
                row.append(str(wells_dict[well][i]))
            f.write("\t".join(row) + "\n")

    return out_name


############################################
# MAIN
############################################

if __name__ == "__main__":

    print(f"Scanning {RAW_FOLDER} for raw CSV files...\n")

    groups = group_files_by_experiment(RAW_FOLDER)

    if not groups:
        print("No CSV files found in data/raw/. Exiting.")
        exit()

    print(f"Found {len(groups)} experiment(s):\n")
    for prefix, files in groups.items():
        fnames = [os.path.basename(f) for f in files]
        print(f"  [{prefix}]")
        for fn in fnames:
            print(f"    {fn}")
    print()

    files_to_delete = []

    for prefix, file_list in groups.items():
        print(f"Processing: {prefix}")

        data = merge_files(file_list)

        if not data:
            print(f"  WARNING: No chromatic data found. Skipping.\n")
            continue

        selected = select_chromatic(data)
        print(f"  Selected chromatic: {selected}")

        time_list = data[selected]["time"]
        wells_dict = data[selected]["wells"]

        if not time_list or not wells_dict:
            print(f"  WARNING: Empty time or wells data. Skipping.\n")
            continue

        out_path = save_compressed(prefix, time_list, wells_dict, COMPRESSED_FOLDER)
        print(f"  Saved compressed file: {out_path}")
        print(f"  Wells: {len(wells_dict)}  Timepoints: {len(time_list)}")

        files_to_delete.extend(file_list)
        print()

    if files_to_delete:
        print("Deleting raw files...")
        for fp in files_to_delete:
            os.remove(fp)
            print(f"  Deleted: {os.path.basename(fp)}")
        print(f"\nDone. {len(files_to_delete)} raw file(s) deleted.")
    else:
        print("No files were processed.")