import itertools
import re
import os

# Load your template
with open("template.txt", "r") as f:
    template_text = f.read()

# Agents and their position fluents
agents = ["a", "b", "c", "d", "e"]
at_fluents = {ag: [f"at_{ag}_1", f"at_{ag}_2"] for ag in agents}

# Fixed fluents always true
fixed_true = ["sa", "sb", "sc", "sd", "se"]

# Flatten all position fluents for false calculation
all_ats = sum(at_fluents.values(), [])
all_fluents = all_ats + fixed_true

# Output folder
os.makedirs("variants", exist_ok=True)

# Generate all combinations: exactly one position fluent true per agent
pos_combinations = list(itertools.product(
    *[[pos[0], pos[1]] for pos in at_fluents.values()]
))

for combo in pos_combinations:
    true_ats = set(combo)
    true_fluents = true_ats.union(fixed_true)
    false_fluents = set(all_fluents) - true_fluents

    # Compose initial line
    initial_line = (
        "initially "
        + ", ".join(sorted(true_fluents))
        + ", "
        + ", ".join(f"-{f}" for f in sorted(false_fluents))
        + ";"
    )

    # Compose belief lines for initially C([a,b,c,d,e], ...)
    c_lines = []
    # For positive and negative position fluents
    for f in sorted(true_fluents):
        c_lines.append(f"initially C([a,b,c,d,e], {f});")
    for f in sorted(false_fluents):
        c_lines.append(f"initially C([a,b,c,d,e], -{f});")
    # Beliefs about agents knowing their respective s-fluents
    for ag, s in zip(agents, fixed_true):
        c_lines.append(f"initially C([a,b,c,d,e], (B({ag},{s}) | B({ag},(-{s}))));")

    # Replace initial fluent line in template
    modified = re.sub(
        r"^initially\s+[-\w_,\s\(\)\|\-]+;",
        initial_line,
        template_text,
        count=1,
        flags=re.MULTILINE,
    )

    # Remove old initially C(...) lines
    modified = re.sub(
        r"^\s*initially C\(\[a,b,c,d,e\],\s*-?[\w(),\s\|\-]*\);\s*\n?",
        "",
        modified,
        flags=re.MULTILINE,
    )

    # Insert new C lines after the initial line
    match = re.search(re.escape(initial_line), modified)
    if not match:
        raise RuntimeError("Couldn't find initially line to insert C lines after.")

    insert_index = match.end()
    modified = modified[:insert_index] + "\n" + "\n".join(c_lines) + "\n" + modified[insert_index:]

    # Filename suffix: e.g. a1b2c1d1e2
    pos_suffix = "".join([f"{f.split('_')[2]}" for f in sorted(true_ats)])
    fname = f"variants/pos_{pos_suffix}.txt"

    with open(fname, "w") as out:
        out.write(modified)

    print(f"Generated: {fname}")
