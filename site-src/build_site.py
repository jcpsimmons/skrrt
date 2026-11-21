#!/usr/bin/env python3
"""Stamps out the skrrt docs pages (shared head/tabline/statusline) into ~/work/skrrt/site.
The output is plain static HTML; this script is a convenience, not a required build step."""
import html, re, sys
from pathlib import Path

OUT = Path(sys.argv[1])
BASE = "https://skrrt.drjoshcsimmons.com"
REPO = "https://github.com/jcpsimmons/skrrt"
PAPER = "https://zenodo.org/records/24428116"
TWEET = "https://x.com/_lyraaaa_/status/2113833330429083662"

PAGES = [  # slug, tab label, menu label
    ("index", "home", "Home"),
    ("how-it-works", "how-it-works", "How it works"),
    ("personas", "personas", "Personas"),
    ("judges", "judges", "Judges"),
    ("backends", "backends", "Backends"),
    ("cli", "cli", "CLI reference"),
    ("results", "results", "Results"),
    ("faq", "faq", "FAQ & limits"),
]

FAVICON = ("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'%3E"
           "%3Crect width='32' height='32' fill='%23000'/%3E"
           "%3Cpath d='M4 9h12q9 0 9 9v10' fill='none' stroke='%2387ff87' stroke-width='3'/%3E"
           "%3Cpath d='M4 16h8q6 0 6 6v6' fill='none' stroke='%2387ff87' stroke-width='3'/%3E"
           "%3Crect x='22' y='3' width='6' height='6' fill='%23ff00ff'/%3E%3C/svg%3E")

# ---------------------------------------------------------------- highlighting
def esc(s): return html.escape(s, quote=False)
def span(c, s): return f'<span class="{c}">{s}</span>'

CMDS = r"(?:skrrt|git|cd|uv|export|llama-server|llamacpp/build\.sh|cat|ls|python3)"
TOK = re.compile(r'(?P<s>"[^"]*"|\'[^\']*\')|(?P<f>(?<![\w-])--?[A-Za-z][\w-]*)')

def _sh_seg(seg):
    m = re.match(r"(\s*)(" + CMDS + r")(?=\s|$)(.*)$", seg, re.S)
    head, rest = "", seg
    if m:
        head, rest = m.group(1) + span("k", m.group(2)), m.group(3)
    def rep(mm):
        return span("s", mm.group(0)) if mm.group("s") else span("i", mm.group(0))
    return head + TOK.sub(rep, rest)

def hl_sh(line):
    line = esc(line)
    code, com = line, ""
    m = re.search(r"(^|\s)#.*$", line)
    if m:
        code, com = line[:m.start() + len(m.group(1))], span("c", line[m.start() + len(m.group(1)):])
    parts = re.split(r"( &amp;&amp; | \| )", code)
    return "".join(p if i % 2 else _sh_seg(p) for i, p in enumerate(parts)) + com

def hl_usage(line):
    line = esc(line)
    code, com = line, ""
    m = re.search(r"(^|\s)#.*$", line)
    if m:
        code, com = line[:m.start() + len(m.group(1))], span("c", line[m.start() + len(m.group(1)):])
    code = re.sub(r'(?P<s>"[^"]*")|(?P<f>(?<![\w-])--?[a-z][\w-]*)|(?P<p>\b[A-Z][A-Z_]+(?:\.yaml|\.gguf)?\b)|(?P<k>^skrrt \S+)',
                  lambda m: span("s" if m.group("s") else "i" if m.group("f") else "p" if m.group("p") else "k", m.group(0)), code)
    return code + com

def hl_yaml(line):
    line = esc(line)
    code, com = line, ""
    m = re.search(r"(^|\s)#.*$", line)
    if m:
        code, com = line[:m.start() + len(m.group(1))], span("c", line[m.start() + len(m.group(1)):])
    m = re.match(r"^(\s*)(- )?([\w-]+)(:)(\s*)(.*)$", code)
    if m:
        ind, dash, key, colon, sp, val = m.groups()
        code = ind + (span("k", dash) if dash else "") + span("i", key) + span("x", colon) + sp + (span("s", val) if val.strip() else val)
    return code + com

def hl_md(line):
    e = esc(line)
    if line.startswith("#"):
        return span("h", e)
    e = e.replace("CONFIRMED", span("ok", "CONFIRMED"))
    if line.lstrip().startswith("&gt;") or line.lstrip().startswith(">"):
        return span("c", e)
    return e

def hl_bib(line):
    e = esc(line)
    m = re.match(r"^(\s*)(\w+)(\s*=\s*)(.*)$", e)
    if m:
        return m.group(1) + span("i", m.group(2)) + m.group(3) + span("s", m.group(4))
    return re.sub(r"^(@\w+)", lambda m: span("k", m.group(1)), e)

HL = {"sh": hl_sh, "yaml": hl_yaml, "usage": hl_usage, "md": hl_md, "text": esc, "bibtex": hl_bib}

def code(lang, text, label=None, copy=True, wrap=False):
    lines = text.strip("\n").split("\n")
    def cls(ln):
        return "l tr" if wrap and ln.lstrip().startswith("|") else "l"
    body = ("" if wrap else "\n").join(f'<span class="{cls(ln)}">{HL[lang](ln)}</span>' for ln in lines)
    nocopy = "" if copy else " data-nocopy"
    if wrap:  # long prose lines (a quoted report): wrap them on wide screens, scroll inside on phones
        nocopy += " data-wrap"
    return (f'<figure class="code"{nocopy}><div class="code-bar"><span>{esc(label or lang)}</span></div>'
            f'<pre tabindex="0"><code>{body}\n</code></pre></figure>\n')

def art(text, cls="", label=None, marks=()):
    """ASCII art <pre>. Sets --cols so CSS can scale it to the viewport."""
    lines = text.strip("\n").split("\n")
    cols = max(len(l) for l in lines)
    for l in lines:
        assert all(32 <= ord(ch) < 127 for ch in l), f"non-ASCII in art: {l!r}"
    body = esc("\n".join(lines))
    for needle, c in marks:
        assert esc(needle) in body, needle
        body = body.replace(esc(needle), span(c, esc(needle)))
    a11y = f'role="img" aria-label="{html.escape(label)}"' if label else 'aria-hidden="true"'
    return f'<pre class="ascii {cls}" style="--cols:{cols}" {a11y}>{body}</pre>\n', cols

# ---------------------------------------------------------------- ASCII art
def box(title, rows, w):
    inner = w - 4
    assert all(len(x) <= inner for x in [title] + list(rows)), (title, rows)
    out = ["+" + "-" * (w - 2) + "+", "| " + title.ljust(inner) + " |", "+" + "-" * (w - 2) + "+"]
    out += ["| " + r.ljust(inner) + " |" for r in rows]
    out.append("+" + "-" * (w - 2) + "+")
    return out

STAGES = [
    ("1 RANK", ["every neuron", "scored by", "cosine with", "the persona", "direction", ""], "model only"),
    ("2 SCREEN", ["top 8/layer", "+ random", "  controls", "6 prompts", "doses 1, 2", ""], "model + judge"),
    ("3 CONFIRM", ["best 10", "+ random", "  controls", "12 held-out", "  prompts", "6 doses"], "model + judge"),
    ("4 REPORT", ["fixed rules", "pick winner:", "CONFIRMED,", "WEAK or", "NOT FOUND", ""], "arithmetic"),
]

def pipeline_wide():
    w = 16
    boxes = [box(t, r, w) for t, r, _ in STAGES]
    lines = []
    for i in range(len(boxes[0])):
        sep = "-->" if i == 1 else "   "
        lines.append(sep.join(b[i] for b in boxes))
    lines.append("   ".join(("  " + n).ljust(w) for _, _, n in STAGES).rstrip())
    top = ["  persona.yaml + model", "        |", "        v"]
    bot = ["", " " * 57 + "       |", " " * 57 + "       v", " " * 57 + "report.md + .json"]
    return "\n".join(top + lines + bot)

def pipeline_narrow():
    w = 32
    out = ["   persona.yaml + model", "              |", "              v"]
    flat = [
        ("1 RANK        (model only)", ["every neuron scored by its", "cosine with the persona", "direction"]),
        ("2 SCREEN   (model + judge)", ["top 8 per layer + random", "controls, 6 prompts,", "doses 1 and 2"]),
        ("3 CONFIRM  (model + judge)", ["best 10 + random controls,", "12 held-out prompts,", "6 doses"]),
        ("4 REPORT      (arithmetic)", ["fixed rules pick the winner:", "CONFIRMED, WEAK or NOT FOUND"]),
    ]
    for i, (t, rows) in enumerate(flat):
        out += box(t, rows, w)
        out += ["              |", "              v"]
    out.append("      report.md + .json")
    return "\n".join(out)

PUSH_WIDE = r"""
 1. what skrrt does: push one neuron, at every token

 layer L-1 =====+===========================(+)=====> layer L+1
                |      residual stream       ^
           read |                            | write
                v                            |
          +------------+     neurons     +----------+
          | W_up, then |        o        |  W_down  |
          | activation | --->   o   ---> | (linear) |
          +------------+       [n]       +----------+
                                ^
                                |
                             + delta

 2. the same thing, because W_down is linear

 layer L-1 =================================(+)=====> layer L+1
                                             ^
                                             |
                                 + delta * W_down[:, n]
                                   (n's "write arrow")
"""

PUSH_NARROW = r"""
          layer L-1
              |
              +--- read ---.
              |            v
              |      +----------+
   residual   |      | W_up +   |
   stream     |      | activ.   |
              |      +----------+
              |            |
              |      o  o [n] o  o
              |            ^
              |         + delta
              |            |
              |      +----------+
              |      |  W_down  |
              |      +----------+
              |            |
             (+)<-- write -'
              |
              v
          layer L+1

 same as adding the fixed vector
 delta * W_down[:, n] at the (+)
"""

GLYPH = {
 "s": [" ####", "#    ", " ### ", "    #", "#### "],
 "k": ["#   #", "#  # ", "###  ", "#  # ", "#   #"],
 "r": ["#### ", "#   #", "#### ", "#  # ", "#   #"],
 "t": ["#####", "  #  ", "  #  ", "  #  ", "  #  "],
}
def wordmark(word="skrrt", px="//", gap=2):
    rows = []
    for r in range(5):
        line = (" " * gap).join("".join(px if c == "#" else " " * len(px) for c in GLYPH[ch][r]) for ch in word)
        rows.append(" " * (4 - r) + line)          # shear right: the word leans into the turn
    return "\n".join(x.rstrip() for x in rows)
WORDMARK = wordmark()

MARKS = r"""
===============================================.
======================================.         `===.
                                       `===.         `==.
                                            `==.        `=.
                                               `=.        \\
                                                 \\       ||
                                                  ||       ||
"""

MARKS_NARROW = r"""
===========================.
====================.       `==.
                     `==.      `=.
                         `=.     \\\\
                           \\\\     ||
                            ||     ||
"""

# ---------------------------------------------------------------- page shell
def shell(slug, title, desc, body, root=""):
    is404 = slug == "404"
    fname = "index.html" if slug == "index" else f"{slug}.html"
    clean = "" if slug == "index" else slug
    def href(s): return (root + "./" if not root else "/") if s == "index" else f"{root}{s}.html"
    tabs = "".join(
        f'<a href="{href(s)}"{" aria-current=\"page\"" if s == slug else ""}>{lab}</a>'
        for s, lab, _ in PAGES if s != "index")
    menu = "".join(
        f'<a href="{href(s)}"{" aria-current=\"page\"" if s == slug else ""}><span>{m}</span><span class="n">{"index" if s == "index" else s}.html</span></a>'
        for s, _, m in PAGES)
    idx = [s for s, _, _ in PAGES].index(slug) if not is404 else -1
    pager = ""
    if not is404:
        prev = PAGES[idx - 1] if idx > 0 else None
        nxt = PAGES[idx + 1] if idx < len(PAGES) - 1 else None
        pager = '<nav class="pager" aria-label="Previous and next page">'
        if prev: pager += f'<a class="prev" href="{href(prev[0])}" rel="prev"><span><span class="n">&lt;-</span> {prev[2]}</span></a>'
        if nxt: pager += f'<a class="next" href="{href(nxt[0])}" rel="next"><span>{nxt[2]} <span class="n">-&gt;</span></span></a>'
        pager += "</nav>"
    og = "" if is404 else f"""<link rel="canonical" href="{BASE}/{clean}">
<meta property="og:type" content="website">
<meta property="og:site_name" content="skrrt">
<meta property="og:title" content="{html.escape(title)}">
<meta property="og:description" content="{html.escape(desc)}">
<meta property="og:url" content="{BASE}/{clean}">
<meta property="og:image" content="{BASE}/og.png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="skrrt in ASCII art with tire marks veering off: find the one neuron that yanks a model's steering wheel">
<meta name="twitter:card" content="summary_large_image">
"""
    robots = '<meta name="robots" content="noindex">\n' if is404 else ""
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>{html.escape(title)}</title>
<meta name="description" content="{html.escape(desc)}">
<meta name="theme-color" content="#000000">
<meta name="color-scheme" content="dark">
{robots}{og}<link rel="icon" href="{FAVICON}">
<link rel="stylesheet" href="{root}site.css">
<script>window.va = window.va || function () {{ (window.vaq = window.vaq || []).push(arguments); }};</script>
<script defer src="/_vercel/insights/script.js"></script>
<script defer src="{root}site.js"></script>
</head>
<body>
<a class="skip" href="#buffer">Skip to content</a>
<header class="tabline">
<a class="brand" href="{href('index')}"{' aria-current="page"' if slug == 'index' else ''} aria-label="skrrt home">skrrt</a>
<span class="here" aria-hidden="true">{fname}</span>
<nav aria-label="Pages">{tabs}</nav>
<span class="fill"></span>
<nav aria-label="Source"><a href="{REPO}" rel="noopener">github</a></nav>
</header>
<main id="buffer">
{body}
{pager}
</main>
<div class="eob" aria-hidden="true"><span>~</span><span>~</span><span>~</span></div>
<footer>
<p>skrrt is open source, by Josh Simmons. <a href="{REPO}" rel="noopener">github.com/jcpsimmons/skrrt</a></p>
<p>Recipe credit: <a href="{TWEET}" rel="noopener">@_lyraaaa_</a> and @chiefofautism. Colors: Vim's murphy.</p>
</footer>
<div class="statusline">
<span class="file">{fname}</span>
<span class="enc" aria-hidden="true">utf-8 | html</span>
<span class="ruler" id="ruler" aria-hidden="true">Top</span>
<details class="menu">
<summary aria-label="Open the page menu">:menu</summary>
<nav class="panel" aria-label="Pages (menu)">{menu}<a href="{REPO}" rel="noopener"><span>GitHub</span><span class="n">jcpsimmons/skrrt</span></a></nav>
</details>
</div>
</body>
</html>
"""

def diagram(wide, narrow, label, caption, marks=()):
    w, _ = art(wide, "wide", label, marks)
    n, _ = art(narrow, "narrow", label, marks)
    return f'<figure class="art">{w}{n}<figcaption>{caption}</figcaption></figure>\n'

BIB = """
@misc{simmons2026personaneurons,
  author    = {Simmons, Josh C.},
  title     = {Persona Neurons: Single-Neuron Steering of Personality in a 27B Language Model},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24428116},
  url       = {https://zenodo.org/records/24428116}
}
"""

P = {}
exec((Path(__file__).parent / "content.py").read_text())

OUT.mkdir(parents=True, exist_ok=True)
for slug, (title, desc, body) in P.items():
    root = "/" if slug == "404" else ""
    (OUT / f"{slug}.html").write_text(shell(slug, title, desc, body, root))
    print("wrote", slug)
