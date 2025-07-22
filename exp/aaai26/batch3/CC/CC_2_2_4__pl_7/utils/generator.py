import itertools
import re

# Load the template
with open("template.txt", "r") as f:
    template_text = f.read()

# Fluent sets
at_a = ["at_a_1", "at_a_2", "at_a_3", "at_a_4"]
at_b = ["at_b_1", "at_b_2", "at_b_3", "at_b_4"]
at_b1 = ["at_b1_1", "at_b1_3", "at_b1_4"]
at_b2 = ["at_b2_1", "at_b2_3", "at_b2_4"]
all_fluents = at_a + at_b + at_b1 + at_b2

# Regex to match the first 'initially ... ;' fluents line
fluents_pattern = r"(initially\s+[-\w_,\s]+;)"

# Generate all permutations
for a, b, b1, b2 in itertools.product(at_a, at_b, at_b1, at_b2):
    pos = {a, b, b1, b2}

    # Construct new initial fluent values
    initial_fluents = [f if f in pos else f"-{f}" for f in all_fluents]
    new_fluents_line = "initially " + ", ".join(initial_fluents) + ";"

    # Step 1: Replace the original fluents line
    modified_text = re.sub(fluents_pattern, new_fluents_line, template_text, count=1)

    # Step 2: Remove old C([a,b], ...) belief lines about at_a_* or at_b_*
    modified_text = re.sub(
        r"^\s*initially C\(\[a,b\],\s*(-?at_[ab]_\d)\);\s*",
        "",
        modified_text,
        flags=re.MULTILINE
    )

    # Step 3: Generate new C-lines
    c_lines = []
    for aa in at_a:
        c_lines.append(f"initially C([a,b], {'-' if aa != a else ''}{aa});")
    for bb in at_b:
        c_lines.append(f"initially C([a,b], {'-' if bb != b else ''}{bb});")

    # Step 4: Insert new C-lines AFTER the new fluent line
    match = re.search(re.escape(new_fluents_line), modified_text)
    if match:
        insert_index = match.end()
        modified_text = modified_text[:insert_index] + "\n" + "\n".join(c_lines) + "\n" + modified_text[insert_index:]
    else:
        raise RuntimeError("Couldn't find the new fluents line to insert beliefs after.")

    # Step 5: Filename
    x1, x2, x3, x4 = a[-1], b[-1], b1[-1], b2[-1]
    filename = f"a{x1}_b{x2}-p{x3}_p{x4}.txt"

    with open(filename, "w") as f:
        f.write(modified_text)

    print(f"Generated: {filename}")
