import itertools
import re

# Load the template
with open("template.txt", "r") as f:
    template_text = f.read()

# Fluent sets
at_a = ["at_a_1", "at_a_2", "at_a_3", "at_a_4"]
at_b = ["at_b_1", "at_b_2", "at_b_3", "at_b_4"]
at_b1 = ["at_b1_1", "at_b1_3", "at_b1_4"]  # Only used for full list of fluents
at_b2 = ["at_b2_1", "at_b2_3", "at_b2_4"]
all_fluents = at_a + at_b + at_b1 + at_b2

# Regex to extract and preserve the original full fluent line
fluents_pattern = r"initially\s+([-\w_,\s]+);"
original_fluent_line = re.search(fluents_pattern, template_text)
if not original_fluent_line:
    raise RuntimeError("Could not find initial fluents line in template.")

# Parse original fluent truth assignments from template
fluent_assignments = {}
for fluent in original_fluent_line.group(1).split(","):
    f = fluent.strip()
    if f.startswith("-"):
        fluent_assignments[f[1:]] = False
    else:
        fluent_assignments[f] = True

# Generate all combinations of at_a and at_b
for a, b in itertools.product(at_a, at_b):
    # Copy original truth assignment
    updated_assignments = fluent_assignments.copy()

    # Update only at_a and at_b values
    for aa in at_a:
        updated_assignments[aa] = (aa == a)
    for bb in at_b:
        updated_assignments[bb] = (bb == b)

    # Reconstruct the initial fluents line
    new_fluents_line = "initially " + ", ".join(
        [f if val else f"-{f}" for f, val in updated_assignments.items()]
    ) + ";"

    # Step 1: Replace the initial fluent line
    modified_text = re.sub(fluents_pattern, new_fluents_line, template_text, count=1)

    # Step 2: Remove old belief lines for at_a and at_b only
    modified_text = re.sub(
        r"^\s*initially C\(\[a,b\],\s*(-?at_[ab]_\d)\);\s*",
        "",
        modified_text,
        flags=re.MULTILINE
    )

    # Step 3: Insert updated belief lines for at_a and at_b
    c_lines = []
    for aa in at_a:
        c_lines.append(f"initially C([a,b], {'-' if aa != a else ''}{aa});")
    for bb in at_b:
        c_lines.append(f"initially C([a,b], {'-' if bb != b else ''}{bb});")

    # Step 4: Insert belief lines after new fluent line
    match = re.search(re.escape(new_fluents_line), modified_text)
    if match:
        insert_index = match.end()
        modified_text = (
            modified_text[:insert_index]
            + "\n"
            + "\n".join(c_lines)
            + "\n"
            + modified_text[insert_index:]
        )
    else:
        raise RuntimeError("Couldn't find the new fluents line to insert beliefs after.")

    # Step 5: Save to file
    x1, x2 = a[-1], b[-1]
    filename = f"a{x1}_b{x2}.txt"
    with open(filename, "w") as f:
        f.write(modified_text)

    print(f"Generated: {filename}")
