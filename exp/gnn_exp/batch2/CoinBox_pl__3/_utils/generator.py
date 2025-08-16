import itertools
import re
import os

# Load the template
with open("template.txt", "r") as f:
    template_text = f.read()

# --- Fluent Groups ---
has_keys = ["has_key_a", "has_key_b", "has_key_c"]
looking = ["looking_a", "looking_b", "looking_c"]

# --- Constants ---
fixed_true = ["tail"]  # always true

# opened can be true or false — we treat it separately
opened_states = [True, False]

all_fluents = has_keys + looking + ["opened"] + fixed_true

# --- Output directory ---
os.makedirs("variants", exist_ok=True)

# Generate permutations: exactly 1 keyholder, at least 1 looking, and opened True/False
key_perms = [(k,) for k in has_keys]

look_perms = []
for r in range(1, len(looking) + 1):
    look_perms.extend(itertools.combinations(looking, r))

for key_holder in key_perms:
    for lookers in look_perms:
        for opened_true in opened_states:
            # Build true fluents
            true_fluents = set(list(key_holder) + list(lookers) + fixed_true)
            if opened_true:
                true_fluents.add("opened")
            # Fluents that are false = all fluents - true fluents
            false_fluents = set(all_fluents) - true_fluents

            # Generate fluent line
            initial_line = (
                "initially "
                + ", ".join(sorted(true_fluents))
                + ", "
                + ", ".join(f"-{f}" for f in sorted(false_fluents))
                + ";"
            )

            # Generate C belief lines
            c_lines = []
            for f in sorted(true_fluents):
                if f != "tail":
                    c_lines.append(f"initially C([a,b,c], {f});")
            for f in sorted(false_fluents):
                if f != "tail":
                    c_lines.append(f"initially C([a,b,c], -{f});")

            # Step 1: Replace fluent line
            modified = re.sub(
                r"^initially\s+[-\w_,\s]+;",
                initial_line,
                template_text,
                count=1,
                flags=re.MULTILINE,
            )

            # Step 2: Remove all old initially C([a,b,c], ...) lines
            modified = re.sub(
                r"^\s*initially C\(\[a,b,c\],\s*-?[\w(),\s\-|]*\);\s*\n?",
                "",
                modified,
                flags=re.MULTILINE,
            )

            # Step 3: Insert new C lines after fluent line
            match = re.search(re.escape(initial_line), modified)
            if not match:
                raise RuntimeError("Couldn't find fluent line to insert C lines after.")

            insert_index = match.end()
            modified = modified[:insert_index] + "\n\n" + "\n".join(c_lines) + "\n\n" + modified[insert_index:]

            # Step 4: Save file
            key = key_holder[0][-1]  # last char: a/b/c
            looks = "".join([l[-1] for l in lookers])  # e.g., "ac"
            opened_str = "open" if opened_true else "closed"
            fname = f"variants/key_{key}_look_{looks}_{opened_str}.txt"

            with open(fname, "w") as out:
                out.write(modified)

            print(f"Generated: {fname}")
