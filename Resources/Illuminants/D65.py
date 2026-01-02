import sys
from pathlib import Path

import numpy as np
import colour

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
AGX_ROOT = REPO_ROOT / "external" / "agx-emulsion"
if AGX_ROOT.exists():
    sys.path.insert(0, str(AGX_ROOT))

from agx_emulsion.config import SPECTRAL_SHAPE

ILLUMINANT_MAP = {
    "D65": ("SDS_ILLUMINANTS", "D65"),
    "D55": ("SDS_ILLUMINANTS", "D55"),
    "D50": ("SDS_ILLUMINANTS", "D50"),
    "T": ("SDS_LIGHT_SOURCES", "Incandescent"),
    "K75P": ("SDS_LIGHT_SOURCES", "Kinoton 75P"),
}


def load_illuminant(key):
    source_type, name = ILLUMINANT_MAP[key]
    if source_type == "SDS_ILLUMINANTS":
        sd = colour.SDS_ILLUMINANTS[name]
    else:
        sd = colour.SDS_LIGHT_SOURCES[name]
    return sd.copy().align(SPECTRAL_SHAPE)


def normalize(values):
    mean_power = values.mean()
    if mean_power > 0.0:
        return values / mean_power
    return values


def write_csv(name, wavelengths, values):
    np.savetxt(
        f"{name}.csv",
        np.column_stack([wavelengths, values]),
        delimiter=",",
        fmt="%.6f",
    )


def main():
    for key in ILLUMINANT_MAP:
        sd = load_illuminant(key)
        values = normalize(sd.values.copy())
        write_csv(key, sd.wavelengths, values)


if __name__ == "__main__":
    main()
