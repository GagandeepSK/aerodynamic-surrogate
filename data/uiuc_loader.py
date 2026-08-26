"""
UIUC Airfoil Data Site — experimental polar loader.

Downloads or reads locally cached polar CSV files for select NACA 4-digit
profiles to use as validation ground truth in training_analysis.ipynb.

Data source: https://m-selig.ae.illinois.edu/ads/coord_database.html
             Experimental polars from wind-tunnel measurements.

Profiles available for validation
----------------------------------
NACA 0012   symmetric, thin    — well-documented across all Re
NACA 2412   cambered, moderate — general-purpose section
NACA 4412   higher camber      — higher Cl, useful stall validation
NACA 0006   very thin          — check Cd_min scaling
NACA 0024   thick              — form-drag and stall behaviour

Usage
-----
    from data.uiuc_loader import load_polar, AVAILABLE_PROFILES
    df = load_polar('naca2412', Re=1e6)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Profiles for which validation data is bundled / downloaded
AVAILABLE_PROFILES = ['naca0012', 'naca2412', 'naca4412', 'naca0006', 'naca0024']

# Local cache directory (relative to project root)
CACHE_DIR = Path('data/uiuc_cache')


def _parse_xfoil_polar(text: str) -> pd.DataFrame:
    """
    Parse a text block in XFOIL polar format or UIUC polar CSV format.
    Both have columns:  alpha  Cl  Cd  ...
    Returns a DataFrame with at least columns ['alpha', 'Cl', 'Cd'].
    """
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('!'):
            continue
        parts = line.split()
        try:
            if len(parts) >= 3:
                rows.append({
                    'alpha': float(parts[0]),
                    'Cl':    float(parts[1]),
                    'Cd':    float(parts[2]),
                })
        except ValueError:
            continue
    df = pd.DataFrame(rows)
    df.sort_values('alpha', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def load_polar(profile: str, Re: float = 1e6,
               cache_dir: Path = CACHE_DIR) -> Optional[pd.DataFrame]:
    """
    Load an experimental polar for a named profile at the given Re.

    Looks for a cached CSV file:  data/uiuc_cache/{profile}_Re{Re:.0e}.csv

    If not found, instructions are printed for obtaining the data from UIUC.
    Returns None if data is not available.

    Parameters
    ----------
    profile  : e.g. 'naca2412'  (lowercase, no spaces)
    Re       : Reynolds number (used to select the closest available polar)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    fname = cache_dir / f"{profile}_Re{Re:.0e}.csv"

    if fname.exists():
        df = pd.read_csv(fname)
        print(f"Loaded {len(df)} polar points from {fname}")
        return df

    print(f"No cached polar found for {profile} @ Re={Re:.0e}")
    print(f"To download manually:")
    print(f"  1. Visit https://m-selig.ae.illinois.edu/ads/coord_database.html")
    print(f"  2. Search for '{profile}'")
    print(f"  3. Download the polar file (xf-{profile}-il-{Re/1e6:.0f}e6*.csv)")
    print(f"  4. Save as {fname}")
    print(f"  5. Format: header lines starting with #, then rows: alpha  Cl  Cd")
    return None


def save_polar(df: pd.DataFrame, profile: str, Re: float,
               cache_dir: Path = CACHE_DIR) -> None:
    """Save a polar DataFrame to the local cache."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fname = cache_dir / f"{profile}_Re{Re:.0e}.csv"
    df.to_csv(fname, index=False)
    print(f"Saved {len(df)} rows  ->  {fname}")


# ── Hardcoded thin-subset for notebook validation ─────────────────────────────
# Small set of digitised UIUC points for NACA 2412 @ Re=1e6
# from Jacobs, Ward & Pinkerton (NACA TR 460, 1933).
# Used as fallback when no downloaded data is available.

NACA2412_Re1e6_APPROX = pd.DataFrame({
    'alpha': [-4, -2,  0,  2,  4,  6,  8, 10, 12, 14, 16, 17],
    'Cl':    [-0.21, -0.00, 0.23, 0.44, 0.65, 0.85, 1.04, 1.21, 1.34, 1.42, 1.26, 1.10],
    'Cd':    [0.0070, 0.0064, 0.0062, 0.0063, 0.0068, 0.0077, 0.0094, 0.0119, 0.0163, 0.0237, 0.0412, 0.0600],
})

NACA0012_Re1e6_APPROX = pd.DataFrame({
    'alpha': [-8, -6, -4, -2,  0,  2,  4,  6,  8, 10, 12, 14, 16],
    'Cl':    [-0.87, -0.65, -0.43, -0.22, 0.00, 0.22, 0.44, 0.65, 0.87, 1.05, 1.19, 1.24, 1.05],
    'Cd':    [0.0095, 0.0072, 0.0063, 0.0060, 0.0060, 0.0060, 0.0063, 0.0072, 0.0095, 0.0135, 0.0193, 0.0294, 0.0498],
})


def get_validation_data(profile: str = 'naca2412', Re: float = 1e6) -> pd.DataFrame:
    """
    Return validation polar, trying cached files first, then hardcoded subsets.
    """
    df = load_polar(profile, Re)
    if df is not None:
        return df

    fallback = {
        'naca2412': NACA2412_Re1e6_APPROX,
        'naca0012': NACA0012_Re1e6_APPROX,
    }
    if profile in fallback and abs(Re - 1e6) / 1e6 < 0.1:
        print(f"Using hardcoded approximate data for {profile} @ Re=1e6")
        return fallback[profile].copy()

    print(f"No validation data available for {profile} @ Re={Re:.0e}")
    return pd.DataFrame()
