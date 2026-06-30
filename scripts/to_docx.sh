#!/usr/bin/env bash
# Convert the assembled Markdown manuscript to an MDPI-styled .docx.
# Requires pandoc (>=3) on PATH. Run from the project root.
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
# 1) embed figure PNGs at their caption blocks
python - <<'PY'
import re
from pathlib import Path
ROOT = Path('.')
src = (ROOT/'docs/manuscript/MANUSCRIPT.md').read_text(encoding='utf-8')
def repl(m):
    cap, path = m.group(1), m.group(2).strip()
    ap = str((ROOT/path).resolve()).replace('\','/')
    return f'![]({ap}){{width=85%}}\n\n{cap}'
src, n = re.subn(r'(\*\*Figure \d+\.\*\* .*?)\s*\(Image:\s*([^)]+)\)', repl, src, flags=re.S)
Path('docs/manuscript/MANUSCRIPT_pandoc.md').write_text(src, encoding='utf-8')
print('embedded', n, 'figures')
PY
# 2) pandoc -> docx with the MDPI-styled reference doc
pandoc docs/manuscript/MANUSCRIPT_pandoc.md \
  -o docs/manuscript/MANUSCRIPT.docx \
  --reference-doc=docs/manuscript/water-reference.docx \
  --from=markdown+pipe_tables+tex_math_dollars+raw_html \
  --resource-path="$ROOT"
echo "wrote docs/manuscript/MANUSCRIPT.docx"
