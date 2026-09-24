"""
Render REPORT.md to a typeset PDF.

Markdown -> styled HTML -> Chrome headless print. Chrome is used rather than a
LaTeX pipeline because the figures are SVG and the browser already renders them
correctly at print resolution; rasterizing them first would throw away the one
advantage of having made them vector.

Two substitutions happen on the way through. The `<picture>` elements that let
GitHub serve a dark figure to dark-mode readers collapse to the light variant,
since paper has one theme. And relative figure paths become absolute `file://`
URLs, because the intermediate HTML lives in a temp directory rather than beside
the images.
"""

from __future__ import annotations

import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date

import markdown

ROOT = pathlib.Path(__file__).resolve().parents[1]
SOURCE = ROOT / "REPORT.md"
OUTPUT = ROOT / "docs" / "culprit-report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 19mm 18mm 20mm 18mm; }

:root {
  --ink:       #12120f;
  --ink-soft:  #4a4945;
  --ink-faint: #76746c;
  --rule:      #e0dfda;
  --accent:    #2a78d6;
  --surface:   #f7f7f4;
}

* { box-sizing: border-box; }

body {
  font-family: Charter, "Bitstream Charter", "Iowan Old Style", Georgia, serif;
  font-size: 10.6pt;
  line-height: 1.58;
  color: var(--ink);
  margin: 0;
  -webkit-font-smoothing: antialiased;
}

/* ---- title block ---- */
.titleblock { border-bottom: 2px solid var(--ink); padding-bottom: 14px; margin-bottom: 26px; }
.titleblock h1 {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 23pt; line-height: 1.15; letter-spacing: -0.4px;
  margin: 0 0 8px; font-weight: 700; border: 0; padding: 0;
}
.titleblock .sub {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 10.5pt; color: var(--ink-soft); margin: 0 0 12px;
}
.titleblock .meta {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 8.8pt; color: var(--ink-faint);
  display: flex; gap: 18px; flex-wrap: wrap;
}
.titleblock .meta a { color: var(--accent); text-decoration: none; }

/* ---- headings ---- */
h1, h2, h3, h4 {
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  color: var(--ink); font-weight: 700; letter-spacing: -0.2px;
  break-after: avoid; page-break-after: avoid;
}
h2 {
  font-size: 15pt; margin: 30px 0 10px; padding-bottom: 6px;
  border-bottom: 1px solid var(--rule);
}
h3 { font-size: 11.8pt; margin: 22px 0 7px; }
h4 { font-size: 10.6pt; margin: 16px 0 5px; color: var(--ink-soft); }

p { margin: 0 0 10px; }
strong { font-weight: 700; }

a { color: var(--accent); text-decoration: none; }

/* ---- the headline callout ---- */
blockquote {
  margin: 18px 0; padding: 14px 18px;
  background: var(--surface); border-left: 3px solid var(--accent);
  border-radius: 0 4px 4px 0;
  font-size: 10.2pt;
  break-inside: avoid; page-break-inside: avoid;
}
blockquote p { margin: 0 0 8px; }
blockquote p:last-child { margin-bottom: 0; }

/* ---- tables ---- */
table {
  border-collapse: collapse; width: 100%; margin: 14px 0 18px;
  font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
  font-size: 9.1pt;
  break-inside: avoid; page-break-inside: avoid;
}
th {
  text-align: left; font-weight: 700; color: var(--ink);
  border-bottom: 1.5px solid var(--ink); padding: 7px 9px 6px;
}
td { padding: 6px 9px; border-bottom: 1px solid var(--rule); color: var(--ink-soft); }
td strong { color: var(--ink); }
tbody tr:last-child td { border-bottom: 1.5px solid var(--ink); }

/* ---- figures ---- */
img { max-width: 100%; height: auto; display: block; margin: 6px auto 4px; }
p:has(> picture), p:has(> img) {
  break-inside: avoid; page-break-inside: avoid; margin: 16px 0 18px;
}

/* ---- code ---- */
code {
  font-family: "SF Mono", Menlo, Consolas, monospace;
  font-size: 0.86em; background: var(--surface);
  padding: 1px 4px; border-radius: 3px; color: var(--ink);
}
pre {
  background: var(--surface); border: 1px solid var(--rule); border-radius: 4px;
  padding: 10px 12px; overflow: hidden; margin: 12px 0 16px;
  break-inside: avoid; page-break-inside: avoid;
}
pre code {
  background: none; padding: 0; font-size: 8.3pt; line-height: 1.48;
  white-space: pre-wrap; word-break: break-word;
}

ul, ol { margin: 0 0 12px; padding-left: 22px; }
li { margin-bottom: 5px; }

hr { border: 0; border-top: 1px solid var(--rule); margin: 26px 0; }

/* keep a heading with the text under it */
h2 + p, h3 + p, h2 + table, h3 + table { break-before: avoid; page-break-before: avoid; }
"""


def to_html(md_text: str) -> str:
    """Markdown -> HTML, with the print-only substitutions applied first."""
    # Paper has one theme: collapse <picture> to its light source.
    md_text = re.sub(
        r'<picture>.*?<img alt="([^"]*)" src="([^"]*)"[^>]*>.*?</picture>',
        r'<img alt="\1" src="\2">',
        md_text,
        flags=re.DOTALL,
    )
    # The intermediate HTML is written to a temp dir, so relative paths break.
    md_text = re.sub(
        r'src="(docs/figures/[^"]+)"',
        lambda m: f'src="file://{ROOT / m.group(1)}"',
        md_text,
    )
    # The title block is rendered by hand below; drop the source's own heading
    # and subtitle. Without MULTILINE the `^` only anchors at string start, and
    # by this point the string starts with the newline left by the heading.
    md_text = re.sub(r"^# .*?\n", "", md_text, count=1)
    md_text = re.sub(
        r"^\*\*An applied research report on `culprit`.*?\*\*\s*\n", "", md_text,
        count=1, flags=re.DOTALL | re.MULTILINE,
    )
    md_text = md_text.lstrip("\n")
    # The source writes double hyphens where a dash is meant, which is fine in a
    # plain-text diff and looks like a typo once typeset.
    md_text = re.sub(r"(?<=\s)--(?=\s)", "\u2013", md_text)
    return markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "md_in_html", "attr_list"],
    )


def build() -> pathlib.Path:
    if not shutil.which(CHROME) and not pathlib.Path(CHROME).exists():
        sys.exit(f"Chrome not found at {CHROME}")

    body = to_html(SOURCE.read_text())
    title_block = f"""
<div class="titleblock">
  <h1>Localizing the cause of agent failures by intervention</h1>
  <p class="sub">An applied research report on <strong>culprit</strong>, a root-cause
     localization tool for Arize Phoenix.</p>
  <div class="meta">
    <span>Laxya Kumar</span>
    <span>{date.today():%B %Y}</span>
    <a href="https://github.com/Lkumar209/culprit-agent-rca">github.com/Lkumar209/culprit-agent-rca</a>
  </div>
</div>
"""
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>culprit — applied research report</title>"
            f"<style>{CSS}</style></head><body>{title_block}{body}</body></html>")

    with tempfile.TemporaryDirectory() as tmp:
        src = pathlib.Path(tmp) / "report.html"
        src.write_text(html)
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
             "--virtual-time-budget=10000",
             f"--print-to-pdf={OUTPUT}", f"file://{src}"],
            check=True, capture_output=True,
        )
    return OUTPUT


if __name__ == "__main__":
    out = build()
    print(f"wrote {out.relative_to(ROOT)}  ({out.stat().st_size / 1024:.0f} KB)")
