"""Where everything lives.

Every script in this package imports this module and asks it for paths, so the
package can be checked out anywhere and run from anywhere. Nothing is written
outside the package root.

    data/           input data and the trained model artefacts
    results/        measurement output, one JSON per script
    figures/output/ rendered figures
    work/           intermediate artefacts, gitignored and safe to delete
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data"
MODELS = DATA / "models"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures" / "output"
WORK = ROOT / "work"

for _d in (MODELS, RESULTS, FIGURES, WORK):
    _d.mkdir(parents=True, exist_ok=True)


def data(*parts):
    """A path inside data/."""
    return str(DATA.joinpath(*parts))


def model(*parts):
    """A path inside data/models/."""
    return str(MODELS.joinpath(*parts))


def results(name):
    """A path inside results/."""
    return str(RESULTS / name)


def figure(name):
    """A path inside figures/output/."""
    return str(FIGURES / name)


def work(name):
    """A path inside work/, for intermediate artefacts."""
    return str(WORK / name)
