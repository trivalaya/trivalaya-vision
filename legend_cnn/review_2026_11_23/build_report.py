"""REPORT.md -> report.html (self-contained, figures embedded). The PDF is printed from report.html with headless Chrome:
  google-chrome --headless=new --no-pdf-header-footer --print-to-pdf=<out.pdf> file://<report.html>
A small Markdown subset only: headings, paragraphs, lists, tables, images on their own line, **bold**, `code`."""
import base64
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent


def inline(s):
    s = html.escape(s, quote=False)
    s = re.sub(r"`([^`]+)`", r"<code>\1</code>", s)
    s = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", s)
    return s


def figure(cap, path):
    data = base64.b64encode((HERE / path).read_bytes()).decode()
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    c = f"<figcaption>{inline(cap)}</figcaption>" if cap else ""
    return f'<figure><img src="data:{mime};base64,{data}" alt="{html.escape(cap)}">{c}</figure>'


def convert(md):
    out, lines, i = [], md.split("\n"), 0
    while i < len(lines):
        ln = lines[i]
        if not ln.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,3}) (.*)", ln)
        if m:
            out.append(f"<h{len(m.group(1))}>{inline(m.group(2))}</h{len(m.group(1))}>")
            i += 1
            continue
        m = re.match(r"^!\[(.*)\]\(([^)]+)\)$", ln)
        if m:
            out.append(figure(m.group(1), m.group(2)))
            i += 1
            continue
        if ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[2:]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body) + "</tbody></table>")
            continue
        m = re.match(r"^(- |\d+\. )", ln)
        if m:
            tag = "ul" if ln.startswith("- ") else "ol"
            items = []
            while i < len(lines) and (re.match(r"^(- |\d+\. )", lines[i]) or (lines[i].startswith("  ") and lines[i].strip())):
                if re.match(r"^(- |\d+\. )", lines[i]):
                    items.append(re.sub(r"^(- |\d+\. )", "", lines[i]))
                else:
                    items[-1] += " " + lines[i].strip()
                i += 1
            out.append(f"<{tag}>" + "".join(f"<li>{inline(x)}</li>" for x in items) + f"</{tag}>")
            continue
        para = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,3} |- |\d+\. |\||!\[)", lines[i]):
            para.append(lines[i].strip())
            i += 1
        out.append(f"<p>{inline(' '.join(para))}</p>")
    return "\n".join(out)


CSS = """
@page { size: Letter; margin: 18mm 17mm; }
* { box-sizing: border-box; }
body { font-family: "DejaVu Sans", "Liberation Sans", Arial, sans-serif; font-size: 10.2pt; line-height: 1.45; color: #0b0b0b; background: #fff; max-width: 780px; margin: 24px auto; padding: 0 16px; }
h1 { font-size: 18pt; line-height: 1.2; margin: 0 0 6pt; }
h2 { font-size: 13pt; margin: 18pt 0 5pt; padding-top: 6pt; border-top: 1px solid #d9d8d3; break-after: avoid; }
h3 { font-size: 10.8pt; margin: 12pt 0 3pt; break-after: avoid; }
p { margin: 0 0 6pt; } ul, ol { margin: 0 0 6pt; padding-left: 18pt; } li { margin-bottom: 2.5pt; }
code { font-family: "DejaVu Sans Mono", "Liberation Mono", monospace; font-size: 8.8pt; background: #f1f0ec; padding: 0 2px; border-radius: 2px; overflow-wrap: anywhere; }
table { border-collapse: collapse; margin: 4pt 0 8pt; font-size: 9.6pt; width: 100%; break-inside: avoid; }
th { text-align: left; color: #52514e; font-weight: 600; border-bottom: 1px solid #a9a8a2; padding: 3pt 8pt 3pt 0; }
td { border-bottom: 1px solid #e4e3df; padding: 3pt 8pt 3pt 0; vertical-align: top; }
figure { margin: 6pt 0 9pt; break-inside: avoid; } figure img { width: 100%; display: block; border: 1px solid #e4e3df; }
figcaption { font-size: 8.8pt; color: #52514e; margin-top: 3pt; }
@media print { body { margin: 0; padding: 0; max-width: none; } }
"""

if __name__ == "__main__":
    body = convert((HERE / "REPORT.md").read_text())
    (HERE / "report.html").write_text(f'<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Legend-ribbon experiment: review, 2026-11-23</title>'
                                      f"<style>{CSS}</style></head><body>{body}</body></html>")
    print("wrote", HERE / "report.html", (HERE / "report.html").stat().st_size, "bytes")
