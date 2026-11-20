"""One shared visual identity for figures and the dashboard.

Colours follow the *method*, never its rank, so a method has the same colour in
every chart. Categorical slots come from a CVD-validated palette, assigned in a
fixed order. Marker shape encodes the family, so identity never relies on colour alone.
"""

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Slots 1..8 of the validated palette in their fixed order (adjacent pairs pass
# the CVD check). Most important methods take the earliest slots.
METHOD_COLORS = {
    "Predictive (Prop.)": "#2a78d6",          # 1 blue    (proposed)
    "Max-Min Fairness": "#eb6834",            # 2 orange
    "Oracle (Prop.)": "#1baf7a",              # 3 aqua    (benchmark)
    "Round Robin": "#eda100",                 # 4 yellow
    "Proportional Share": "#e87ba4",          # 5 magenta
    "Predictive (Max-Min)": "#008300",        # 6 green   (proposed)
    "Reactive Prop. + headroom": "#6250d6",   # 7 violet  (ablation)
    "Oracle (Max-Min)": "#e34948",            # 8 red     (benchmark)
}
METHOD_ORDER = list(METHOD_COLORS)

FAMILY_MARKERS = {"traditional": "o", "proposed": "D", "benchmark": "*", "ablation": "s"}
FAMILY_PLOTLY_SYMBOLS = {"traditional": "circle", "proposed": "diamond",
                         "benchmark": "star", "ablation": "square"}

DATASET_LABELS = {
    "milan": "Milan (mobile grid)",
    "mawi": "MAWI (backbone)",
    "caida": "CAIDA (backbone)",
    "fcc": "FCC (home broadband)",
}

SEQUENTIAL_BLUE = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


def apply_matplotlib_style():
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "font.family": "sans-serif", "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False,
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
    })
