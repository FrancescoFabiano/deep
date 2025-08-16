import re

def parse_latex_table(latex):
    rows = []
    for line in latex.splitlines():
        if "&" in line and "\\" in line:
            columns = [col.strip() for col in line.split("&")]
            if columns[0].startswith("\\textbf{"):  # Skip average line
                continue
            rows.append(columns)
    return rows

def build_data_map(new_rows):
    data_map = {}
    for row in new_rows:
        name = row[0].replace("\\_", "_")
        length = row[2]
        nodes = row[3]
        total = row[4]
        search = row[6]
        data_map[name] = (length, nodes, total, search)
    return data_map

def update_original_table(orig_latex, new_data_map):
    updated_lines = []
    missing = []
    for line in orig_latex.splitlines():
        if "&" in line and "\\" in line:
            parts = [part.strip() for part in line.split("&")]
            name_raw = parts[0]
            name_clean = name_raw.replace("\\_", "_")
            if name_clean in new_data_map:
                length, nodes, total, search = new_data_map[name_clean]
                # Replace last four columns
                parts[-4] = length
                parts[-3] = nodes
                parts[-2] = total
                parts[-1] = search + " \\\\"
                updated_lines.append(" & ".join(parts))
            else:
                missing.append(name_clean)
                updated_lines.append(line)
        else:
            updated_lines.append(line)
    return "\n".join(updated_lines), missing

# --- Paste your original table string here ---
with open("original_table.tex", "r") as f:
    original_latex = f.read()

# --- Paste the new table content here ---
with open("SPG.txt", "r") as f:
    new_latex = f.read()

# Process and merge
new_rows = parse_latex_table(new_latex)
data_map = build_data_map(new_rows)
updated_table, missing_instances = update_original_table(original_latex, data_map)

# Output updated table
with open("merged_table.tex", "w") as f:
    f.write(updated_table)

# Print missing instances
if missing_instances:
    print("⚠️ Missing instances in new data:")
    for name in missing_instances:
        print(" -", name)
else:
    print("✅ All instances matched.")
