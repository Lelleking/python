import re
import math
import glob

MAX_WELLS_PER_FILE = 25


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


############################################
# MERGE FLERA FILER
############################################
def merge_files(file_list):
    merged = {}

    for file in file_list:
        data = parse_file(file)

        for chrom in data:

            if chrom not in merged:
                merged[chrom] = {"time": [], "wells": {}}

            original_time = data[chrom]["time"]

            # beräkna offset för JUST denna chromatic
            if merged[chrom]["time"]:
                time_offset = merged[chrom]["time"][-1]
            else:
                time_offset = 0

            adjusted_time = [t + time_offset for t in original_time]
            merged[chrom]["time"].extend(adjusted_time)

            for well in data[chrom]["wells"]:
                if well not in merged[chrom]["wells"]:
                    merged[chrom]["wells"][well] = []

                merged[chrom]["wells"][well].extend(
                    data[chrom]["wells"][well]
                )

    return merged

############################################
# VÄLJ CHROMATIC
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
# EXPORTERA I BLOCK OM 25 WELLS
############################################
def export_split_files(time, wells_dict):

    import os

    # Hämta mappens namn
    lab_name = os.path.basename(os.getcwd())

    wells_sorted = sorted(wells_dict.keys())
    total_wells = len(wells_sorted)

    n_files = math.ceil(total_wells / MAX_WELLS_PER_FILE)

    print(f"Totalt wells: {total_wells}")
    print(f"Skapar {n_files} filer")

    start_time = time[0]

    for file_index in range(n_files):
        start = file_index * MAX_WELLS_PER_FILE
        end = start + MAX_WELLS_PER_FILE
        subset_wells = wells_sorted[start:end]

        filename = f"{lab_name}_amylo_part{file_index+1}.txt"

        with open(filename, "w") as f:
            header = "Time\t" + "\t".join(subset_wells) + "\n"
            f.write(header)

            for i in range(len(time)):
                row = [str((time[i] - start_time) / 3600)]
                for well in subset_wells:
                    row.append(str(wells_dict[well][i]))
                f.write("\t".join(row) + "\n")

        print(f"Export klar: {filename}")

############################################
# MAIN – KÖR MED MAPPNAMN
############################################

if __name__ == "__main__":

    import sys
    import os
    import glob

    if len(sys.argv) != 2:
        print("Använd: python amyloconvert.py <mappnamn>")
        exit()

    folder_name = sys.argv[1]

    # Vi står i Koder → gå upp till kc
    base_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    target_folder = os.path.join(base_path, folder_name)

    if not os.path.isdir(target_folder):
        print(f"Hittar inte mappen: {target_folder}")
        exit()

    # Byt temporärt arbetsmapp
    os.chdir(target_folder)

    files = sorted(glob.glob("*.csv"))

    if not files:
        print("Inga CSV-filer hittades i denna mapp.")
        exit()

    print("Filer som används:", files)

    data = merge_files(files)
    selected = select_chromatic(data)

    print("Vald chromatic:", selected)

    time = data[selected]["time"]
    wells = data[selected]["wells"]

    export_split_files(time, wells)