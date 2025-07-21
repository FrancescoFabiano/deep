import itertools
import re
import os

# Load the template file
with open("template.txt", "r") as f:
    template_text = f.read()

# Fluents for positions for each agent (exactly one of the two should be true)
at_a = ["at_a_1", "at_a_2"]
at_b = ["at_b_1", "at_b_2"]
at_c = ["at_c_1", "at_c_2"]

# Fixed fluents always true
fixed_true = ["sa", "sb", "sc"]

# All position fluents combined for easy false calc
all_ats = at_a + at_b + at_c
all_fluents = all_ats + fixed_true

# Output directory
os.makedirs("variants", exist_ok=True)

# Generate all combinations where for each agent exactly one 'at_*' fluent is true
# For each agent, select exactly one from their 2 location fluents:
pos_combinations = list(itertools.product(
    [(at_a[0],), (at_a[1],)],
    [(at_b[0],), (at_b[1],)],
    [(at_c[0],), (at_c[1],)],
))

for combo in pos_combinations:
    # Flatten tuple of tuples
    true_ats = set(sum(combo, ()))
    true_fluents = true_ats.union(fixed_true)
    false_fluents = set(all_fluents) - true_fluents

    # Compose initially fluent line
    initial_line = (
        "initially "
        + ", ".join(sorted(true_fluents))
        + ", "
        + ", ".join(f"-{f}" for f in sorted(false_fluents))
        + ";"
    )

    # Compose belief lines for initially C([a,b,c], ...)
    c_lines = []
    for f in sorted(true_fluents):
        c_lines.append(f"initially C([a,b,c], {f});")
    for f in sorted(false_fluents):
        c_lines.append(f"initially C([a,b,c], -{f});")

    # --- Step 1: Replace initial 'initially ...;' line in template ---
    modified = re.sub(
        r"^initially\s+[-\w_,\s\(\)\|]+;",
        initial_line,
        template_text,
        count=1,
        flags=re.MULTILINE,
    )

    # --- Step 2: Remove old initially C(...) lines ---
    modified = re.sub(
        r"^\s*initially C\(\[a,b,c\],\s*-?[\w(),\s\|\-]*\);\s*\n?",
        "",
        modified,
        flags=re.MULTILINE,
    )

    # --- Step 3: Insert new C lines after initial_line ---
    match = re.search(re.escape(initial_line), modified)
    if not match:
        raise RuntimeError("Couldn't find initially line to insert C lines after.")

    insert_index = match.end()
    modified = modified[:insert_index] + "\n" + "\n".join(c_lines) + "\n" + modified[insert_index:]

    # --- Step 4: Save to file ---
    # Generate filename suffix based on positions, e.g., a1b2c1
    pos_suffix = "".join([f"{f.split('_')[2]}" for f in sorted(true_ats)])  # extracts '1' or '2'
    fname = f"variants/pos_{pos_suffix}.txt"

    with open(fname, "w") as out:
        out.write(modified)

    print(f"Generated: {fname}")
