"""Markdown -> styled HTML -> PDF through headless Google Chrome (no extra packages) - for the printable runbooks.

    python tools/docs/md2pdf.py docs/hardware/HW2_RUNBOOK.md docs/hardware/HW2_RUNBOOK.pdf --landscape --font-pt 7.6
    python tools/docs/md2pdf.py docs/hardware/HW2_CHECKLIST.md docs/hardware/HW2_CHECKLIST.pdf
    python tools/docs/md2pdf.py --html-only docs/hardware/HW2_RUNBOOK.md /tmp/runbook.html     # no Chrome needed

Why a tool here: the operator reads the runbook on paper / a tablet on the day (the session-10 decision "markdown +
PDF"). It follows an earlier converter of ours (headings, paragraphs, nested lists, fenced code, tables, bold, inline
code, links, rules) with what an 8-column runbook needs: a LANDSCAPE page, a smaller font, table cells that wrap long
words and paths, a table's header repeated on every page, and no lab fonts (system fonts only: the repository is
public). The output is deterministic for a given input and Chrome version (the PDF's own metadata aside).

Chrome: `$CHROME`, else the macOS app path, else `google-chrome` / `chromium` on PATH.
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
FENCE_RE = re.compile(r"^\s*```")
TABLE_SEP_RE = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
MAC_CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def split_row(line: str) -> list[str]:
    """A markdown table row -> its cells; `\\|` is a literal bar inside a cell (the runbook's shell pipes)."""
    s = line.strip()
    if s.startswith("|"):
        s = s[1:]
    if s.endswith("|") and not s.endswith("\\|"):
        s = s[:-1]
    cells = re.split(r"(?<!\\)\|", s)
    return [c.strip().replace("\\|", "|") for c in cells]


def inline(text: str) -> str:
    """Inline markup: `code`, **bold** (also around code: **`x` true**), [text](link); everything else escaped."""
    codes: list[str] = []

    def stash(m: re.Match[str]) -> str:
        code = html.escape(m.group(1)).replace("/", "/<wbr>").replace("_", "_<wbr>")  # long paths may wrap
        codes.append(f"<code>{code}</code>")
        return f"\x00{len(codes) - 1}\x00"

    t = re.sub(r"`([^`]+)`", stash, text)
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', t)
    return re.sub("\x00(\\d+)\x00", lambda m: codes[int(m.group(1))], t)


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def dedent(lines: list[str]) -> list[str]:
    ind = min((indent_of(ln) for ln in lines if ln.strip()), default=0)
    return [ln[ind:] if len(ln) >= ind else ln.lstrip() for ln in lines]


def is_block_start(line: str) -> bool:
    s = line.strip()
    return bool(
        FENCE_RE.match(line)
        or s.startswith("#")
        or LIST_RE.match(line)
        or s.startswith("|")
        or re.fullmatch(r"-{3,}|\*{3,}", s)
    )


def cell(k: int, text: str) -> str:
    """A table cell; a SHORT first cell (a step id: S1b, S3.2a-e, FS4-FS8) is kept on one line."""
    short = k == 0 and len(text) <= 16 and not text.startswith("`")
    return f'<td class="id">{inline(text)}</td>' if short else f"<td>{inline(text)}</td>"


def render(lines: list[str]) -> str:
    """Block-level markdown -> HTML."""
    out: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if not s:
            i += 1
            continue
        if FENCE_RE.match(line):
            j, body = i + 1, []
            while j < len(lines) and not FENCE_RE.match(lines[j]):
                body.append(lines[j])
                j += 1
            code = html.escape("\n".join(dedent(body))).replace("/", "/<wbr>")
            out.append(f"<pre><code>{code}</code></pre>")
            i = j + 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            n = len(m.group(1))
            out.append(f"<h{n}>{inline(m.group(2))}</h{n}>")
            i += 1
            continue
        if re.fullmatch(r"-{3,}|\*{3,}", s):
            out.append("<hr>")
            i += 1
            continue
        if s.startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1]):
            head = split_row(s)
            rows, j = [], i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append(split_row(lines[j]))
                j += 1
            t = "<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
            t += "".join("<tr>" + "".join(cell(k, c) for k, c in enumerate(r)) + "</tr>" for r in rows)
            out.append(t + "</tbody></table>")
            i = j
            continue
        if LIST_RE.match(line):
            html_list, i = render_list(lines, i)
            out.append(html_list)
            continue
        para = [s]
        i += 1
        while i < len(lines) and lines[i].strip() and not is_block_start(lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


def render_list(lines: list[str], i: int) -> tuple[str, int]:
    base = indent_of(lines[i])
    m0 = LIST_RE.match(lines[i])
    assert m0 is not None
    ordered = m0.group(2)[0].isdigit()
    items = []
    while i < len(lines):
        m = LIST_RE.match(lines[i])
        if not m or indent_of(lines[i]) != base or m.group(2)[0].isdigit() != ordered:
            break
        body = [m.group(3)]
        i += 1
        while i < len(lines):
            ln = lines[i]
            if ln.strip() and indent_of(ln) <= base:
                break
            if not ln.strip():  # a blank line ends the item unless indented text follows
                k = i + 1
                while k < len(lines) and not lines[k].strip():
                    k += 1
                if k >= len(lines) or indent_of(lines[k]) <= base:
                    break
            body.append(ln)
            i += 1
        first, rest = body[0], dedent(body[1:])
        lead, j = [first], 0
        while j < len(rest) and rest[j].strip() and not is_block_start(rest[j]):
            lead.append(rest[j].strip())
            j += 1
        items.append("<li>" + inline(" ".join(lead)) + render(rest[j:]) + "</li>")
        while i < len(lines) and not lines[i].strip():
            k = i + 1
            if k < len(lines) and LIST_RE.match(lines[k]) and indent_of(lines[k]) == base:
                i = k
            break
    tag = "ol" if ordered else "ul"
    return f"<{tag}>" + "".join(items) + f"</{tag}>", i


CSS = """
@page { size: Letter %(orient)s; margin: 10mm 9mm 11mm; }
body { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif; font-size: %(pt).1fpt; line-height: 1.35;
       color: #20262a; }
h1 { font-size: %(h1).1fpt; color: #193657; margin: 0 0 5pt; }
h2 { font-size: %(h2).1fpt; color: #193657; margin: 12pt 0 4pt; padding-top: 4pt; border-top: 1px solid #c1ced8;
     break-after: avoid; }
h3 { font-size: %(h3).1fpt; color: #193657; margin: 9pt 0 3pt; break-after: avoid; }
p { margin: 3pt 0; }
ul, ol { margin: 2pt 0 4pt; padding-left: 14pt; }
li { margin: 1pt 0; }
code { font-family: Menlo, Consolas, monospace; font-size: %(code).1fpt; background: #eef0f3; padding: 0 1.5pt; }
pre { background: #f3f4f2; border-left: 3px solid #193657; padding: 4pt 6pt; margin: 3pt 0 5pt; white-space: pre-wrap;
      overflow-wrap: anywhere; break-inside: avoid; }
pre code { background: none; padding: 0; }
table { border-collapse: collapse; width: 100%%; margin: 4pt 0 7pt; table-layout: auto; }
thead { display: table-header-group; }
th, td { border: 1px solid #d0d3cf; padding: 2pt 3.5pt; text-align: left; vertical-align: top;
         overflow-wrap: break-word; word-break: normal; hyphens: manual; }
td.id { white-space: nowrap; }
th { background: #eaf0f4; color: #193657; }
tr { break-inside: avoid; }
strong { color: #10294a; }
a { color: #193657; }
hr { border: 0; border-top: 1px solid #c1ced8; margin: 8pt 0; }
.foot { margin-top: 10pt; font-size: 7pt; color: #626d73; }
"""


def to_html(src_text: str, *, title: str, source_name: str, landscape: bool, font_pt: float) -> str:
    css = CSS % {
        "orient": "landscape" if landscape else "portrait",
        "pt": font_pt,
        "h1": font_pt * 2.0,
        "h2": font_pt * 1.45,
        "h3": font_pt * 1.15,
        "code": font_pt * 0.9,
    }
    body = render(src_text.splitlines())
    return (
        f"<!doctype html><html><head><meta charset='utf-8'><title>{html.escape(title)}</title>"
        f"<style>{css}</style></head><body>{body}"
        f"<p class='foot'>Generated from {html.escape(source_name)} by tools/docs/md2pdf.py - edit the .md and "
        "re-run the tool to update.</p></body></html>"
    )


def chrome_path() -> str:
    env = os.environ.get("CHROME")
    if env:
        return env
    if Path(MAC_CHROME).exists():
        return MAC_CHROME
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            return p
    raise SystemExit("md2pdf: Google Chrome not found (set $CHROME)")


def md_to_pdf(src: Path, dst: Path, *, landscape: bool = False, font_pt: float = 9.5) -> Path:
    doc = to_html(src.read_text(), title=src.stem, source_name=src.name, landscape=landscape, font_pt=font_pt)
    with tempfile.TemporaryDirectory() as d:
        h = Path(d) / (src.stem + ".html")
        h.write_text(doc)
        subprocess.run(
            [chrome_path(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--print-to-pdf-no-header",
             f"--print-to-pdf={dst.resolve()}", h.as_uri()],  # no --user-data-dir: a fresh profile hung (macOS, 3 min)
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120,
        )  # fmt: skip
    return dst


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python tools/docs/md2pdf.py", description=(__doc__ or "").splitlines()[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--landscape", action="store_true", help="a landscape page (wide tables: the runbooks)")
    ap.add_argument("--font-pt", type=float, default=9.5)
    ap.add_argument("--html-only", action="store_true", help="write the HTML to dst instead of a PDF (no Chrome)")
    a = ap.parse_args(argv)
    if a.html_only:
        a.dst.write_text(
            to_html(a.src.read_text(), title=a.src.stem, source_name=a.src.name, landscape=a.landscape,
                    font_pt=a.font_pt)
        )  # fmt: skip
    else:
        md_to_pdf(a.src, a.dst, landscape=a.landscape, font_pt=a.font_pt)
    print(a.dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
