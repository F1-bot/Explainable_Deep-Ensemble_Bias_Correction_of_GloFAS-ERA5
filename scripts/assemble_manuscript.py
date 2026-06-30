"""Complete, regenerable manuscript assembly for the MDPI Water article.

Single source of truth = the section drafts (citekey placeholders) + the table
files + figure_captions.md. This script:
  - concatenates sections 00-06,
  - normalizes the KGE-prime glyph and Results heading depth,
  - reconciles Table 3 <-> Table 4 (UQ vs positioning) to order of appearance,
  - resolves [@citekey] -> MDPI numbers in order of first appearance + ref list,
  - inserts the four tables and the fourteen figure captions at first citation,
  - writes docs/manuscript/MANUSCRIPT.md + a citation audit.
Re-run after editing any section to regenerate the full manuscript.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(r"G:/MDPI Q1-2026")
SEC = ROOT / "docs/manuscript/sections"
TBL = ROOT / "docs/manuscript/tables"
REFDB = ROOT / "references/references_database.md"
CAPS = ROOT / "docs/manuscript/figure_captions.md"
OUT = ROOT / "docs/manuscript/MANUSCRIPT.md"

SECTION_ORDER = ["00_frontmatter.md", "01_introduction.md", "02_methods.md",
                 "03_results.md", "04_discussion.md", "05_conclusions.md", "06_backmatter.md"]

# --- reference database: citekey -> full MDPI entry -------------------------
db = {}
for line in REFDB.read_text(encoding="utf-8").splitlines():
    m = re.match(r"^\*\*\[@([A-Za-z0-9_]+)\]\*\*\s*(.+)$", line.strip())
    if m:
        db[m.group(1)] = m.group(2).strip()

# --- concatenate sections ---------------------------------------------------
body = "\n\n".join((SEC / f).read_text(encoding="utf-8").strip() for f in SECTION_ORDER)

# --- glyph + heading normalization ------------------------------------------
body = re.sub(r"KGE[′'’ʹ]", "KGE′", body)
body = body.replace("ΔKGE-prime", "ΔKGE′").replace("KGE-prime", "KGE′").replace("KGE prime", "KGE′")
body = re.split(r"\n##\s+References\b", body)[0].rstrip()           # drop placeholder refs
body = re.sub(r"(?m)^##\s+(3\.\d+\.?\s)", r"### \1", body)            # Results subsections -> h3

# --- reconcile Table 3 <-> Table 4 (UQ in 3.6 precedes positioning in 4.1) ---
body = body.replace("Table 3", "Table @@T@@").replace("Table 4", "Table 3").replace("Table @@T@@", "Table 4")
#   now: UQ table = Table 3, positioning table = Table 4

# --- insert tables + figure captions at first citation (BEFORE citation
#     resolution, so [@citekey] inside tables are numbered too) --------------
def load_table(fn, num):
    t = (TBL / fn).read_text(encoding="utf-8").strip()
    t = re.sub(r"\*\*Table\s+\w+\.\*\*", f"**Table {num}.**", t, count=1)
    t = re.sub(r"(?m)^Table\s+\w+\.\s", f"Table {num}. ", t, count=1)
    return t


TABLES = {1: load_table("table1_skill_by_split.md", 1), 2: load_table("table2_ablation.md", 2),
          3: load_table("table4_uq_suite.md", 3), 4: load_table("table3_positioning.md", 4)}
FIGCAPS = {int(m.group(1)): m.group(0).strip()
           for m in re.finditer(r"\*\*Figure (\d+)\.\*\* .+?(?=\n\n\*\*Figure |\Z)",
                                 CAPS.read_text(encoding="utf-8"), re.S)}

blocks = body.split("\n\n")


def insert_after(label, content):
    pat = re.compile(r"\b" + label.replace(" ", r"\s+") + r"\b(?!\d)")
    for i, b in enumerate(blocks):
        if content and content.split("**", 2)[1:2] and b.strip().startswith("**" + label):
            return  # already present
        if pat.search(b):
            blocks.insert(i + 1, content)
            return


for n in range(1, 15):
    if n in FIGCAPS:
        insert_after(f"Figure {n}", FIGCAPS[n])
for n in range(1, 5):
    insert_after(f"Table {n}", TABLES[n])
body = "\n\n".join(blocks)

# --- resolve [@citekey] -> numbers in order of first appearance -------------
CITE = re.compile(r"\[@([A-Za-z0-9_;@\s]+?)\]")
order, key2num = [], {}


def keys_of(group):
    return [k.strip().lstrip("@").strip() for k in group.split(";") if k.strip()]


for mm in CITE.finditer(body):
    for k in keys_of(mm.group(1)):
        if k not in key2num:
            order.append(k); key2num[k] = len(order)
missing = [k for k in order if k not in db]


def fmt_nums(nums):
    nums = sorted(set(nums)); parts, i = [], 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        parts.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return "[" + ",".join(parts) + "]"


body = CITE.sub(lambda mm: fmt_nums([key2num[k] for k in keys_of(mm.group(1)) if k in key2num]) or mm.group(0), body)

# --- numbered reference list ------------------------------------------------
refs = ["", "## References", ""]
for k in order:
    refs.append(f"{key2num[k]}. {db.get(k, '**MISSING: ' + k + '**')}"); refs.append("")
manuscript = body + "\n" + "\n".join(refs) + "\n"
OUT.write_text(manuscript, encoding="utf-8")

nfig = len(re.findall(r"(?m)^\*\*Figure \d+\.\*\*", manuscript))
ntab = len(re.findall(r"(?m)^\*\*Table \d+\.\*\*", manuscript))
print(f"ASSEMBLED -> {OUT}")
print(f"  references: {len(order)} numbered (in order of appearance); missing: {missing}")
print(f"  figure captions placed: {nfig}/14 ; table blocks placed: {ntab}/4")
print(f"  unused DB keys: {len(set(db) - set(order))}")
print(f"  approx words: {len(re.findall(chr(92)+'w+', manuscript)):,}")
