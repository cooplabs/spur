"""tools/docs/md2pdf.py (session 11): the runbook's markdown -> HTML (the PDF is Chrome's print of it).

Pins what an 8-column runbook needs: a table row with a shell pipe written `\\|` stays ONE cell (the HW2 runbook's
`journalctl ... \\| grep ...`), the header repeats per page, long paths may wrap, the landscape page, text escaped.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def md2pdf() -> ModuleType:
    spec = importlib.util.spec_from_file_location("md2pdf", REPO / "tools" / "docs" / "md2pdf.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_escaped_pipe_stays_in_one_cell(md2pdf: ModuleType) -> None:
    row = "| R2 | robot shell | `journalctl -u r1rt -o cat \\| grep status` | nothing |"
    assert md2pdf.split_row(row) == ["R2", "robot shell", "`journalctl -u r1rt -o cat | grep status`", "nothing"]


def test_table_heading_code_and_escaping(md2pdf: ModuleType) -> None:
    md = "\n".join(
        [
            "# Title <x>",
            "",
            "| ID | Exact words |",
            "|---|---|",
            "| **S1b** | `r1 on-stand` & co |",
            "",
            "- one",
            "- two",
            "",
            "```zsh",
            "r1() { ... }",
            "```",
        ]
    )
    out = md2pdf.render(md.splitlines())
    assert "<h1>Title &lt;x&gt;</h1>" in out
    assert "<thead><tr><th>ID</th><th>Exact words</th></tr></thead>" in out
    assert '<td class="id"><strong>S1b</strong></td>' in out
    assert "<code>r1 on-stand</code> &amp; co" in out
    assert "<ul><li>one</li><li>two</li></ul>" in out
    assert "<pre><code>r1() { ... }</code></pre>" in out


def test_page_setup(md2pdf: ModuleType) -> None:
    doc = md2pdf.to_html("x", title="t", source_name="t.md", landscape=True, font_pt=7.6)
    assert "size: Letter landscape" in doc
    assert "font-size: 7.6pt" in doc
    assert "thead { display: table-header-group; }" in doc  # the header row repeats on every page
    assert "size: Letter portrait" in md2pdf.to_html("x", title="t", source_name="t.md", landscape=False, font_pt=9)


def test_long_paths_may_wrap(md2pdf: ModuleType) -> None:
    out = md2pdf.inline("`/opt/r1mc3/current/deploy/bin/r1mc3-svc`")
    assert out.count("<wbr>") >= 4


def test_bold_around_code_and_short_ids(md2pdf: ModuleType) -> None:
    assert md2pdf.inline("**`rt_granted` true**") == "<strong><code>rt_<wbr>granted</code> true</strong>"
    out = md2pdf.render(
        ["| ID | x |", "|---|---|", "| S3.2a-e | y |", "| the stand, the remote (L2+B), the E-stop | z |"]
    )
    assert '<td class="id">S3.2a-e</td>' in out
    assert "<td>the stand, the remote (L2+B), the E-stop</td>" in out
