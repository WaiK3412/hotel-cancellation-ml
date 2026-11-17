"""Small evidence helpers used by every phase: paths, seed, logging, tables, JSON and figures."""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SEED = 42

# One colour-blind-safe palette (Okabe-Ito) for every figure, so models keep the same colour
# in every chart of the report, the slides and the notebooks.
PALETTE = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MODEL_COLOURS = {
    "Majority baseline": "#999999",
    "Stratified baseline": "#BBBBBB",
    "Logistic Regression": "#0072B2",
    "Random Forest": "#009E73",
    "Hist Gradient Boosting": "#E69F00",
    "Voting": "#D55E00",
    "Stacking": "#CC79A7",
}
plt.rcParams.update({
    "figure.dpi": 100,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "legend.frameon": False,
})


def log(phase, step, message):
    """Echo a progress line and append it to work/run_log.md.

    The run log lives in the git-ignored work/ folder, so re-running a notebook never
    rewrites tracked files.
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = ROOT / "work" / "run_log.md"
    path.parent.mkdir(exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n### {phase} — {step} — {timestamp}\n\n{message}\n")
    print(f"{phase} | {step}: {message}", flush=True)


def table(frame, name):
    """Save a results table as results/<name> and return it unchanged.

    LF line endings keep the CSV byte-identical whether it is written on Windows or Linux.
    """
    frame.to_csv(ROOT / "results" / name, index=False, lineterminator="\n")
    return frame


def save_json(value, path):
    """Write JSON (UTF-8, indented) to a path relative to the repository root."""
    (ROOT / path).write_text(json.dumps(value, indent=2, default=str) + "\n", encoding="utf-8", newline='\n')


def load_json(path):
    """Read JSON from a path relative to the repository root."""
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def figure(name, title, xlabel, ylabel):
    """Label, save and close the current single-axis matplotlib figure."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(ROOT / "reports/figures" / name, bbox_inches="tight")
    plt.close()


def save_figure(fig, name, title=None):
    """Save and close a multi-axis figure with the shared style."""
    if title:
        fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / "reports/figures" / name, bbox_inches="tight")
    plt.close(fig)
