"""Hand-crafted features that describe where and how the failed dies are spread."""
import numpy as np
from skimage.measure import label, regionprops


def region_densities(m):
    """Fail rate in 13 zones: 1 center, 4 inner-ring sectors, 8 outer-ring sectors.

    Center defects light up the center zone, Edge-Ring lights up all outer zones,
    Edge-Loc lights up only one or two outer zones.
    """
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    cy, cx = (h - 1) / 2, (w - 1) / 2
    r = np.sqrt(((yy - cy) / (h / 2)) ** 2 + ((xx - cx) / (w / 2)) ** 2)
    ang = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)  # 0..1

    die = m > 0
    fail = m == 2
    zones = [r < 0.33]
    zones += [(r >= 0.33) & (r < 0.66) & (ang >= k / 4) & (ang < (k + 1) / 4) for k in range(4)]
    zones += [(r >= 0.66) & (ang >= k / 8) & (ang < (k + 1) / 8) for k in range(8)]
    return [fail[z].sum() / max(die[z].sum(), 1) for z in zones]


def radial_profile(m, n_rings=8):
    """Fail rate at different distances from the wafer center (center -> edge)."""
    h, w = m.shape
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.sqrt(((yy - (h - 1) / 2) / (h / 2)) ** 2 + ((xx - (w - 1) / 2) / (w / 2)) ** 2)
    die, fail = m > 0, m == 2
    edges = np.linspace(0, 1.0001, n_rings + 1)
    out = []
    for a, b in zip(edges[:-1], edges[1:]):
        z = (r >= a) & (r < b)
        out.append(fail[z].sum() / max(die[z].sum(), 1))
    return out


def shape_features(m):
    """Shape of the biggest connected group of failed dies.

    A Scratch is long and thin (high eccentricity, low solidity),
    a Loc cluster is a compact blob, Random has no big group at all.
    """
    h, w = m.shape
    fail = m == 2
    n_die = max((m > 0).sum(), 1)
    lab = label(fail, connectivity=2)
    if lab.max() == 0:
        return [0.0] * 6
    regions = regionprops(lab)
    big = max(regions, key=lambda p: p.area)
    cy, cx = big.centroid
    dist = np.sqrt(((cy - (h - 1) / 2) / (h / 2)) ** 2 + ((cx - (w - 1) / 2) / (w / 2)) ** 2)
    return [
        big.area / n_die,          # size of biggest cluster
        big.eccentricity,          # 0 = round, 1 = line
        big.solidity,              # how filled-in the cluster is
        dist,                      # how far the cluster is from the center
        len(regions) / n_die,      # number of separate clusters
        big.area / max(fail.sum(), 1),  # share of all failures in the biggest cluster
    ]


FEATURE_NAMES = (
    ["defect_density"]
    + [f"zone_{i}" for i in range(13)]
    + [f"ring_{i}" for i in range(8)]
    + ["cluster_area", "cluster_eccentricity", "cluster_solidity",
       "cluster_center_dist", "n_clusters", "cluster_share"]
)


def extract_features(m):
    density = (m == 2).sum() / max((m > 0).sum(), 1)
    return [density] + region_densities(m) + radial_profile(m) + shape_features(m)
