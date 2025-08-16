data = r"""
Assemble\_B2-pl\_5 & 5 & 14 & 140 & 5 & 14 & 47 & 5 & 8 & 176 & \SPG \ \
    Assemble\_B4-pl\_5 & 5 & 14 & 243 & 5 & 14 & 77 & 5 & 8 & 209 & \SPG \ \
    Assemble\_B6-pl\_5 & 5 & 14 & 760 & 5 & 14 & 461 & 5 & 8 & 3140 & \SPG \ \
    Assemble\_B8-pl\_5 & 5 & 14 & 31750 & 5 & 14 & 24112 & 5 & 8 & 239144 & \SPG \ \
    Assemble\_B9-pl\_5 & 5 & 14 & 585443 & 5 & 14 & 390542 & - & - & - & - \ \
    Assemble\_B10-pl\_5 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & - & - & - & - \ \
    Assemble\_C-pl\_5 & 5 & 14 & 132 & 5 & 14 & 49 & 5 & 8 & 178 & \SPG \ \
    CC\_2\_2\_3-pl\_3 & 3 & 9 & 62 & 3 & 9 & 17 & 3 & 3 & 89 & \SPG \ \
    CC\_2\_2\_3-pl\_5 & 5 & 76 & 362 & 5 & 77 & 176 & 5 & 5 & 92 & \SPG \ \
    CC\_2\_2\_3-pl\_7 & 7 & 451 & 2245 & 7 & 1052 & 2776 & 7 & 18 & 389 & \SPG \ \
    CC\_2\_2\_3-pl\_8 & 8 & 1934 & 11128 & 8 & 2045 & 5186 & - & - & - & - \ \
    CC\_2\_2\_4-pl\_3 & 3 & 11 & 182 & 3 & 6 & 60 & 3 & 3 & 147 & \SPG \ \
    CC\_2\_2\_4-pl\_4 & 4 & 23 & 378 & 4 & 24 & 232 & 4 & 4 & 209 & \SPG \ \
    CC\_2\_2\_4-pl\_6 & 6 & 432 & 4769 & 6 & 824 & 5030 & 6 & 6 & 296 & \SPG \ \
    CC\_2\_2\_4-pl\_7 & 7 & 1911 & 21141 & 7 & 4171 & 16361 & 10 & 26 & 1542 & \SPG \ \
    CC\_2\_3\_4-pl\_4 & 4 & 42 & 8451 & 4 & 29 & 2504 & 4 & 4 & 1531 & \SPG \ \
    CC\_2\_3\_4-pl\_5 & 5 & 127 & 30013 & 5 & 277 & 18700 & 5 & 5 & 2390 & \SPG \ \
    CC\_2\_3\_4-pl\_6 & 6 & 756 & 94259 & 6 & 1644 & 50540 & 6 & 6 & 1731 & \SPG \ \
    CC\_3\_2\_3-pl\_3 & 3 & 14 & 123 & 3 & 8 & 49 & 3 & 3 & 167 & \SPG \ \
    CC\_3\_2\_3-pl\_6 & 6 & 368 & 4453 & 6 & 471 & 2205 & 6 & 6 & 284 & \SPG \ \
    CC\_3\_2\_3-pl\_7 & 7 & 1596 & 25601 & 7 & 3329 & 17005 & 10 & 27 & 1077 & \SPG \ \
    CC\_3\_3\_3-pl\_3 & 3 & 15 & 699 & 3 & 8 & 180 & 3 & 3 & 337 & \SPG \ \
    CC\_3\_3\_3-pl\_5 & 5 & 253 & 6863 & 5 & 538 & 3136 & 5 & 5 & 471 & \SPG \ \
    CC\_3\_3\_3-pl\_6 & 6 & 1955 & 52929 & 6 & 1643 & 15139 & 6 & 8 & 737 & \SPG \ \
    CC\_3\_3\_3-pl\_7 & 7 & 11769 & 304660 & 7 & 12143 & 86997 & - & - & - & - \ \
    Coin\_Box-pl\_2 & 2 & 2 & 61 & 2 & 2 & 11 & 2 & 2 & 72 & \SPG \ \
    Coin\_Box-pl\_5 & 5 & 77 & 425 & 5 & 101 & 377 & 5 & 5 & 350 & \SPG \ \
    Coin\_Box-pl\_7 & 7 & 1816 & 18684 & 7 & 2490 & 7580 & 8 & 9 & 705 & \SPG \ \
    Grapevine\_3-pl\_2 & 2 & 4 & 111 & 2 & 6 & 60 & 2 & 2 & 91 & \SPG \ \
    Grapevine\_3-pl\_5 & 5 & 568 & 10866 & 5 & 821 & 4755 & 7 & 15 & 867 & \SPG \ \
    Grapevine\_3-pl\_6 & 6 & 1599 & 21958 & 6 & 2113 & 9928 & 6 & 7 & 536 & \SPG \ \
    Grapevine\_3-pl\_7 & 7 & 6561 & 92670 & 7 & 12014 & 61716 & 11 & 26 & 2193 & \SPG \ \
    Grapevine\_4-pl\_3 & 3 & 13 & 824 & 3 & 40 & 1351 & 3 & 3 & 537 & \SPG \ \
    Grapevine\_4-pl\_4 & 4 & 141 & 7062 & 4 & 233 & 5094 & 4 & 4 & 553 & \SPG \ \
    Grapevine\_4-pl\_5 & 5 & 434 & 17301 & 5 & 1445 & 16673 & - & - & - & - \ \
    Grapevine\_4-pl\_6 & 6 & 6790 & 310258 & 6 & 4066 & 40681 & - & - & - & - \ \
    Grapevine\_5-pl\_2 & 2 & 4 & 1344 & 2 & 8 & 1906 & 2 & 2 & 1226 & \SPG \ \
    Grapevine\_5-pl\_3 & 3 & 27 & 17341 & 3 & 71 & 26331 & 3 & 3 & 1750 & \SPG \ \
    Grapevine\_5-pl\_5 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & 7 & 13 & 10355 & \SPG \ \
    Grapevine\_5-pl\_6 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & 6 & 7 & 12755 & \SPG \ \
    SC\_4\_1-pl\_5 & 5 & 17 & 73 & 5 & 11 & 21 & 5 & 6 & 55 & \SPG \ \
    SC\_4\_2-pl\_5 & 5 & 15 & 215 & 5 & 21 & 49 & 5 & 6 & 99 & \SPG \ \
    SC\_4\_2-pl\_8 & 8 & 339 & 2484 & 8 & 306 & 627 & - & - & - & - \ \
    SC\_4\_3-pl\_6 & 6 & 21 & 162 & 6 & 21 & 19 & - & - & - & - \ \
    SC\_4\_3-pl\_8 & 8 & 59 & 273 & 8 & 66 & 53 & - & - & - & - \ \
    SC\_4\_4-pl\_5 & 5 & 17 & 119 & 5 & 11 & 11 & 5 & 6 & 68 & \SPG \ \
    SC\_8\_10-pl\_8 & 8 & 542 & 4565 & 8 & 388 & 1412 & - & - & - & - \ \
    SC\_8\_10-pl\_9 & 9 & 1660 & 20349 & 9 & 1332 & 4478 & - & - & - & - \ \
    SC\_8\_10-pl\_12 & 12 & 18541 & 187834 & 12 & 21604 & 35400 & 13 & 13 & 996 & \SPG \ \
    SC\_9\_11-pl\_4 & 4 & 8 & 89 & 4 & 6 & 23 & 4 & 4 & 127 & \SPG \ \
    SC\_9\_11-pl\_5 & 5 & 11 & 157 & 5 & 10 & 32 & 6 & 7 & 220 & \SPG \ \
    SC\_9\_11-pl\_7 & 7 & 41 & 424 & 7 & 32 & 96 & 7 & 11 & 337 & \SPG \ \
    SC\_9\_11-pl\_9 & 9 & 238 & 2497 & 9 & 146 & 439 & 10 & 16 & 737 & \SPG \ \
    SC\_9\_11-pl\_10 & 10 & 663 & 6789 & 10 & 356 & 1364 & 20 & 52 & 3455 & \SPG \ \
    SC\_9\_11-pl\_11 & 11 & 1262 & 14034 & 11 & 913 & 2892 & 16 & 23 & 1294 & \SPG \ \
    SC\_10\_8-pl\_9 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & - & - & - & - \ \
    SC\_10\_8-pl\_14 & 14 & 93 & 544 & 14 & 51 & 127 & 14 & 14 & 395 & \SPG \ \
    SC\_10\_10-pl\_9 & 9 & 10 & 151 & 9 & 10 & 35 & 9 & 9 & 126 & \SPG \ \
    SC\_10\_10-pl\_10 & 10 & 15 & 223 & 10 & 12 & 29 & 10 & 10 & 169 & \SPG \ \
    SC\_10\_10-pl\_17 & 17 & 946 & 6582 & 17 & 1078 & 3180 & 22 & 22 & 763 & \SPG \ \
    gossip\_3\_3\_3 & 4 & 35 & 1257 & 4 & 41 & 170 & 4 & 10 & 475 & \SPG \ \
    gossip\_3\_3\_6 & 2 & 3 & 195 & 2 & 4 & 50 & 2 & 3 & 149 & \SPG \ \
    gossip\_3\_3\_9 & 6 & 356 & 3296 & 6 & 356 & 479 & 6 & 21 & 696 & \SPG \ \
    gossip\_4\_3\_9 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & 9 & 50 & 6988 & \SPG \ \
    gossip\_4\_4\_1 & 2 & 2 & 249 & 2 & 2 & 128 & 2 & 2 & 541 & \SPG \ \
    gossip\_4\_4\_3 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & 8 & 45 & 28848 & \SPG \ \
    gossip\_4\_4\_7 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & 12 & 80 & 42795 & \SPG \ \
    gossip\_5\_3\_3 & 4 & 168 & 16236 & 4 & 179 & 2572 & 6 & 27 & 7701 & \SPG \ \
    gossip\_5\_3\_5 & 0 & 0 & 114 & 0 & 0 & 6 & 0 & 0 & 18 & \SPG \ \
    gossip\_5\_3\_7 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & 12 & 80 & 23180 & \SPG \ \
    gossip\_5\_4\_3 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & 10 & 69 & 35226 & \SPG \ \
    gossip\_5\_4\_4 & 1 & 1 & 267 & 1 & 1 & 68 & 1 & 1 & 198 & \SPG \ \
    gossip\_5\_4\_9 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & 16 & 129 & 58642 & \SPG \ \
    gossip\_5\_5\_4 & 1 & 1 & 731 & 1 & 1 & 172 & 1 & 1 & 663 & \SPG \ \
    gossip\_5\_5\_8 & \unsolvedColumn & \unsolvedColumn & \myTO & \unsolvedColumn & \unsolvedColumn & \myTO & - & - & - & - \\"""

# --- PROCESSING ---

lines = [line.strip().rstrip("\\") for line in data.strip().split("\n") if line.strip()]
total_rows = len(lines)

# Detect how many sets of tool columns there are
num_columns = len(lines[0].split("&"))
num_tools = (num_columns - 1) // 3

# Initialize solved counts for each tool
solved_counts = [0] * num_tools

# Define what counts as unsolved
unsolved_tokens = {"TO", r"\myTO", r"\unsolvedColumn", "-", "inf"}

for line in lines:
    parts = [p.strip() for p in line.split("&")]
    for i in range(num_tools):
        time_col = 1 + i * 3 + 2
        if time_col < len(parts) and parts[time_col] not in unsolved_tokens:
            solved_counts[i] += 1

# --- GENERATE LaTeX OUTPUT ---

latex_parts = []
for i, solved in enumerate(solved_counts):
    percent = (solved / total_rows) * 100
    end = "}" if i == num_tools - 1 else "|}"
    value = 3
    if i == 2:
        value = 4
    latex_parts.append(
        f"\\multicolumn{{{value}}}{{c{end}{{{solved}/{total_rows} (${percent:.2f}\\%$)}}"
    )

latex_output = "\\hline\nSolved Instances & " + " & ".join(latex_parts)
print(latex_output)
