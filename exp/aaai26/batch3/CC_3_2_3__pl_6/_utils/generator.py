import itertools
import re

# Load the template
with open("template.txt", "r") as f:
    template_text = f.read()

# Extract all declared fluents from the 'fluent ... ;' line
fluent_line_match = re.search(r"fluent\s+([-\w_,\s]+);", template_text)
if not fluent_line_match:
    raise RuntimeError("Could not find 'fluent' line in template.")
declared_fluents = [f.strip() for f in fluent_line_match.group(1).split(",")]

# Fluent groups
at_a = [f for f in declared_fluents if f.startswith("at_a_")]
at_b = [f for f in declared_fluents if f.startswith("at_b_")]
at_c = [f for f in declared_fluents if f.startswith("at_c_")]
at_b1 = [f for f in declared_fluents if f.startswith("at_b1_")]
at_b2 = [f for f in declared_fluents if f.startswith("at_b2_")]
all_fluents = at_a + at_b + at_c + at_b1 + at_b2

# Parse the original initially line for truth values
initial_line_match = re.search(r"initially\s+([-\w_,\s]+);", template_text)
if not initial_line_match:
    raise RuntimeError("Could not find 'initially ...;' line in template.")
initial_assignments = {}
for token in initial_line_match.group(1).split(","):
    token = token.strip()
    if token.startswith("-"):
        initial_assignments[token[1:]] = False
    else:
        initial_assignments[token] = True

# Generate permutations for at_a, at_b, at_c
for a, b, c in itertools.product(at_a, at_b, at_c):
    current_assignments = initial_assignments.copy()

    # Override at_a, at_b, at_c values based on current permutation
    for f in at_a:
        current_assignments[f] = (f == a)
    for f in at_b:
        current_assignments[f] = (f == b)
    for f in at_c:
        current_assignments[f] = (f == c)

    # Rebuild the fluent line
    new_fluents_line = "initially " + ", ".join(
        [f if current_assignments.get(f, False) else f"-{f}" for f in all_fluents]
    ) + ";"

    # Step 1: Replace the fluent line
    modified_text = re.sub(
        r"^initially\s+[-\w_,\s]+;",
        new_fluents_line,
        template_text,
        count=1,
        flags=re.MULTILINE
    )

    # Step 2: Remove all C-lines for at_a/b/c fluents
    modified_text = re.sub(
        r"^\s*initially C\(\[a,b,c\],\s*(-?at_[abc]_\d+)\);\s*",
        "",
        modified_text,
        flags=re.MULTILINE
    )

    # Step 3: Generate new C-lines for at_a, at_b, at_c
    c_lines = []
    for f in at_a:
        c_lines.append(f"initially C([a,b,c], {'-' if f != a else ''}{f});")
    for f in at_b:
        c_lines.append(f"initially C([a,b,c], {'-' if f != b else ''}{f});")
    for f in at_c:
        c_lines.append(f"initially C([a,b,c], {'-' if f != c else ''}{f});")

    # Step 4: Insert C-lines after the fluents line
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
        raise RuntimeError("Couldn't find where to insert belief lines.")

    # Step 5: Create and save file
    x1, x2, x3 = a[-1], b[-1], c[-1]
    filename = f"a{x1}_b{x2}_c{x3}.txt"
    with open(filename, "w") as f:
        f.write(modified_text)

    print(f"Generated: {filename}")
