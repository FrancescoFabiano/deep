import re
import csv
import argparse

def extract_tables_from_latex(latex_text):
    subsection_pattern = re.compile(r'\\subsection\*\{([^\}]*)\}')
    subsections = [(m.start(), m.group(1)) for m in subsection_pattern.finditer(latex_text)]

    tabular_pattern = re.compile(r'\\begin\{tabular\}\{[^\}]+\}(.+?)\\end\{tabular\}', re.DOTALL)
    tabulars = [(m.start(), m.group(1)) for m in tabular_pattern.finditer(latex_text)]

    results = []
    for pos, table_content in tabulars:
        title = ""
        for start, t in subsections:
            if start < pos:
                title = t
            else:
                break
        results.append((title, table_content))
    return results

def parse_tabular_content(table_content):
    lines = table_content.splitlines()
    rows = []
    for line in lines:
        line = line.strip()
        if not line or any(x in line for x in [r'\toprule', r'\midrule', r'\bottomrule']):
            continue
        line = re.sub(r'\\\\$', '', line).strip()
        if not line:
            continue
        cols = [col.strip() for col in line.split('&')]
        rows.append(cols)
    return rows

def write_tables_to_csv(tables, output_csv_path):
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for title, table_content in tables:
            writer.writerow([title])
            rows = parse_tabular_content(table_content)
            for row in rows:
                writer.writerow(row)
            writer.writerow([])  # blank line between tables

def main():
    parser = argparse.ArgumentParser(description="Convert LaTeX tabular tables to CSV with titles")
    parser.add_argument("input_path", help="Path to input LaTeX file")
    parser.add_argument("output_path", help="Path to output CSV file")

    args = parser.parse_args()

    with open(args.input_path, "r", encoding="utf-8") as f:
        latex_text = f.read()

    tables = extract_tables_from_latex(latex_text)
    write_tables_to_csv(tables, args.output_path)
    print(f"CSV created at {args.output_path}")

if __name__ == "__main__":
    main()
