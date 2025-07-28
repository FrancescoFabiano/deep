import csv
import os

ROOT_DIR = os.path.join(os.getcwd(), 'exp', 'aaai26')

def clean_csv_and_delete_files(csv_path):
    # Determine which column to use for the file path
    use_path_column = "Path Hash" if "/batch3/" in csv_path.replace("\\", "/") else "Path Hash Merged"
    new_rows = []

    with open(csv_path, mode='r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        headers = reader.fieldnames

        if "Distance From Goal" not in headers or use_path_column not in headers:
            print(f"Skipping {csv_path}: missing required columns.")
            return

        for row in reader:
            try:
                distance = float(row["Distance From Goal"])
            except ValueError:
                new_rows.append(row)
                continue

            if distance >= 1_000_000:
                rel_path = row[use_path_column]
                if os.path.isfile(rel_path):
                    try:
                        os.remove(rel_path)
                        print(f"Deleted file: {rel_path}")
                    except Exception as e:
                        print(f"Error deleting {rel_path}: {e}")
                else:
                    print(f"File not found: {rel_path}")
            else:
                new_rows.append(row)

    # Overwrite CSV with filtered rows
    with open(csv_path, mode='w', newline='', encoding='utf-8') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=headers)
        writer.writeheader()
        writer.writerows(new_rows)

    print(f"Cleaned CSV: {csv_path}")

def find_and_process_csvs(root):
    for dirpath, _, filenames in os.walk(root):
        for file in filenames:
            if file.endswith(".csv"):
                full_path = os.path.join(dirpath, file)
                clean_csv_and_delete_files(full_path)

if __name__ == "__main__":
    find_and_process_csvs(ROOT_DIR)
