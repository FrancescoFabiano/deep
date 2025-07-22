import itertools
import re

# Load the template
with open("template.txt", "r") as f:
    template_text = f.read()

# Define fluent sets
at_a = ["at_a_1", "at_a_2", "at_a_3"]
at_b = ["at_b_1", "at_b_2", "at_b_3"]
at_c = ["at_c_1", "at_c_2", "at_c_3"]
at_b1 = ["at_b1_1", "at_b1_3"]
at_b2 = ["at_b2_1", "at_b2_3"]
all_fluents = at_a + at_b + at_c + at_b1 + at_b2

# Generate permutations
for a, b, c, b1, b2 in itertools.product(at_a, at_b, at_c, at_b1, at_b2):
    pos = {a, b, c, b1, b2}

    # Rebuild fluent list
    initial_fluents = [f if f in pos else f"-{f}" for f in all_fluents]
    new_fluents_line = "initially " + ", ".join(initial_fluents) + ";"

    # Step 1: Replace original fluent line
    modified_text = re.sub(
        r"^initially\s+[-\w_,\s]+;",
        new_fluents_line,
        template_text,
        count=1,
        flags=re.MULTILINE
    )

    # Step 2: Remove all lines like: initially C([a,b,c], <at_a_*>); etc.
    modified_text = re.sub(
        r"^\s*initially C\(\[a,b,c\],\s*(-?at_[abc]_\d+)\);\s*",
        "",
        modified_text,
        flags=re.MULTILINE
    )

    # Step 3: Generate new C([a,b,c], ...) lines for at_a_*, at_b_*, at_c_*
    c_lines = []
    for aa in at_a:
        c_lines.append(f"initially C([a,b,c], {'-' if aa != a else ''}{aa});")
    for bb in at_b:
        c_lines.append(f"initially C([a,b,c], {'-' if bb != b else ''}{bb});")
    for cc in at_c:
        c_lines.append(f"initially C([a,b,c], {'-' if cc != c else ''}{cc});")

    # Step 4: Insert the new C lines right after the fluent line we just replaced
    match = re.search(re.escape(new_fluents_line), modified_text)
    if match:
        insert_index = match.end()
        modified_text = modified_text[:insert_index] + "\n" + "\n".join(c_lines) + "\n" + modified_text[insert_index:]
    else:
        raise RuntimeError("Could not find insertion point for C lines.")

    # Step 5: Create filename
    x1, x2, x3 = a[-1], b[-1], c[-1]
    x4, x5 = b1[-1], b2[-1]
    filename = f"a{x1}_b{x2}_c{x3}-p{x4}_p{x5}.txt"

    # Step 6: Save
    with open(filename, "w") as f:
        f.write(modified_text)

    print(f"Generated: {filename}")
