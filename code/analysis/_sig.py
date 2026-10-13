"""Per-paper semantics on top of code/figstyle: colour follows the ENTITY, fixed across all
signature figures (Fig 1 sets SFT = teal, DPO = coral). Receipts are read from notes/."""
import os, sys, json, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(ROOT, "code", "figstyle"))
import figstyle as fs
NOTES = os.path.join(ROOT, "notes"); FIGS = os.path.join(ROOT, "figures")

# arms / stages
SFT, DPO, RL, HEAVY, SHAPED, NULL = fs.TEAL, fs.CORAL, fs.INDIGO, fs.AMBER, fs.VIOLET, fs.GREY
# families (levels plots)
QWEN, GEMMA, LLAMA = fs.INDIGO, fs.AMBER, fs.VIOLET
TEXTW = fs.TEXTWIDTH_IN

def receipt(name):
    return json.load(open(os.path.join(NOTES, name)))

def finish(fig, stem, check=True):
    """Write figures/<stem>.svg + .pdf at EXACTLY figsize (bbox_inches=None: a tight bbox would enlarge the
    canvas and shrink every font when LaTeX scales the figure to its \\includegraphics width), fonttype 42
    -> CID TrueType; refuse Type 3. Warns when any artist lies outside the canvas."""
    import matplotlib.pyplot as plt
    plt.rcParams["savefig.bbox"] = None
    fig.canvas.draw(); W, H = fig.get_size_inches(); tb = fig.get_tightbbox(fig.canvas.get_renderer())
    over = (max(0, -tb.x0), max(0, tb.x1 - W), max(0, -tb.y0), max(0, tb.y1 - H))
    if max(over) > 0.005: print(f"  WARN {stem}: content outside canvas by (l,r,b,t) in = {tuple(round(v, 3) for v in over)}")
    fs.save(fig, os.path.join(FIGS, stem + ".svg"))
    pdf = os.path.join(FIGS, stem + ".pdf")
    out = subprocess.run(["pdffonts", pdf], capture_output=True, text=True).stdout
    if check and "Type 3" in out:
        raise SystemExit(f"Type 3 font in {pdf}:\n{out}")
    print(f"wrote {pdf}  fonts: " + ", ".join(sorted({l.split()[1] + " " + l.split()[2] for l in out.splitlines()[2:] if l.strip()})))
    return pdf

def tidy(ax, grid="y"):
    ax.grid(False)
    if grid: ax.grid(True, axis=grid)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
