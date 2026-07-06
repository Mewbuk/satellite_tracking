#!/usr/bin/env python3
"""
Matched-filter + local-background star detector for satellite-tracked frames.

INPUT  : a single .fits image (stars appear as parallel trails)
OUTPUT : star centroids (green) + satellite/compact sources (red, "SAT" on
         brightest), saved as one annotated PNG into PNG_DIR, plus .txt
         source lists next to the FITS, and an optional plate-solved WCS.

Install: pip install numpy scipy scikit-image astropy matplotlib skyfield
Run    : edit the SETTINGS block below, then:  python satellite_traking.py
"""

import math
import os
import numpy as np
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import fit_wcs_from_points
import astropy.units as u
from scipy.ndimage import median_filter, convolve, binary_closing
from scipy.spatial import cKDTree
from skimage.draw import line as draw_line
from skimage.measure import label, regionprops
from skimage.transform import rescale
import matplotlib.pyplot as plt


def robust_sigma(a):
    m = np.median(a)
    return np.std(a[a < m + 3 * np.std(a)])


def line_kernel(length, angle_deg):
    length = max(3, int(length) | 1)
    k = np.zeros((length, length)); c = length // 2
    dx, dy = np.cos(np.deg2rad(angle_deg)), np.sin(np.deg2rad(angle_deg))
    rr, cc = draw_line(int(c - dy * length / 2), int(c - dx * length / 2),
                       int(c + dy * length / 2), int(c + dx * length / 2))
    ok = (rr >= 0) & (rr < length) & (cc >= 0) & (cc < length)
    k[rr[ok], cc[ok]] = 1.0
    return k / k.sum()


def estimate_angle(data_sub, length, scale=0.25):
    """Self-correcting: scan all orientations, keep the strongest matched response."""
    small = rescale(data_sub, scale, anti_aliasing=True, preserve_range=True)
    klen = max(7, int(length * scale))
    best_ang, best = 0.0, -np.inf
    for ang in np.arange(0, 180, 3):
        s = np.percentile(convolve(small, line_kernel(klen, ang)), 99.9)
        if s > best: best, best_ang = s, ang
    for ang in np.arange(best_ang - 3, best_ang + 3.01, 1):
        s = np.percentile(convolve(small, line_kernel(klen, ang % 180)), 99.9)
        if s > best: best, best_ang = s, ang % 180
    return float(best_ang % 180)


def source_angle(p):
    """Major-axis orientation of a region in the SAME convention as line_kernel
    (degrees from +x, mod 180), via PCA on its pixel coords. Convention-safe."""
    ys, xs = p.coords[:, 0].astype(float), p.coords[:, 1].astype(float)
    pts = np.column_stack([xs - xs.mean(), ys - ys.mean()])
    if len(pts) < 2:
        return 0.0
    evals, evecs = np.linalg.eigh(np.cov(pts.T))
    vx, vy = evecs[:, int(np.argmax(evals))]
    return float(np.rad2deg(np.arctan2(vy, vx)) % 180)


def angle_diff(a, b):
    """Smallest separation between two undirected angles (0-90 deg)."""
    d = abs(a - b) % 180
    return min(d, 180 - d)


def spine_centroid(data_sub, p, frac=0.30):
    """Sub-pixel centroid using only the bright spine of the source (pixels above
    `frac` x peak), flux-weighted. Removes the bias from blurred edge pixels."""
    ys, xs = p.coords[:, 0], p.coords[:, 1]
    vals = data_sub[ys, xs]
    keep = vals >= frac * vals.max()
    if keep.sum() == 0:
        keep = slice(None)
    w = vals[keep]
    return (float(np.average(xs[keep], weights=w)),
            float(np.average(ys[keep], weights=w)))


def window_centroid(data_sub, x0, y0, box=4, frac=0.30):
    """Sub-pixel center-of-mass in a small window (for compact/point sources)."""
    H, W = data_sub.shape
    x0i, y0i = int(round(x0)), int(round(y0))
    x1, x2 = max(0, x0i - box), min(W, x0i + box + 1)
    y1, y2 = max(0, y0i - box), min(H, y0i + box + 1)
    sub = data_sub[y1:y2, x1:x2]
    if sub.size == 0 or sub.max() <= 0:
        return x0, y0
    m = sub >= frac * sub.max()
    yy, xx = np.mgrid[y1:y2, x1:x2]
    w = sub[m]
    return (float(np.average(xx[m], weights=w)),
            float(np.average(yy[m], weights=w)))


def detect_sources(fits_path, angle_deg=None, length=110, nsigma=4.0, bg_box=64,
                   min_elong=3.0, close_frac=0.3, max_angle_dev=12.0, min_len_frac=0.5,
                   border_margin=20, sat_border_margin=60):
    """Returns (stars, compact, angle, image)."""
    data = fits.getdata(fits_path).astype(float)
    H, W = data.shape
    data_sub = data - median_filter(data, size=bg_box)

    if angle_deg is None:
        angle_deg = estimate_angle(data_sub, length)
        print(f"  auto-estimated trail angle: {angle_deg:.1f} deg "
              f"(set ANGLE_DEG in SETTINGS if this looks wrong)")

    mask_raw = data_sub > nsigma * robust_sigma(data_sub)
    compact, comp_xy = [], []
    for p in regionprops(label(mask_raw), intensity_image=data_sub):
        if p.area < 4:
            continue
        elong = p.axis_major_length / max(p.axis_minor_length, 1e-6)
        if elong >= min_elong:
            continue
        minr, minc, maxr, maxc = p.bbox
        if (minr < sat_border_margin or minc < sat_border_margin or
                maxr > H - sat_border_margin or maxc > W - sat_border_margin):
            continue
        rough_y, rough_x = p.centroid_weighted
        xc, yc = window_centroid(data_sub, rough_x, rough_y)
        compact.append((xc, yc, float(p.image_intensity.sum())))
        comp_xy.append((xc, yc))

    mf = convolve(data_sub, line_kernel(length, angle_deg))
    mask_mf = mf > nsigma * robust_sigma(mf)
    if close_frac > 0:
        mask_mf = binary_closing(mask_mf, structure=(line_kernel(int(length * close_frac),
                                                                 angle_deg) > 0))
    stars = []
    for p in regionprops(label(mask_mf), intensity_image=data_sub):
        if p.area < 5:
            continue
        elong = p.axis_major_length / max(p.axis_minor_length, 1e-6)
        if elong < min_elong:
            continue
        minr, minc, maxr, maxc = p.bbox
        if (minr < border_margin or minc < border_margin or
                maxr > H - border_margin or maxc > W - border_margin):
            continue
        if angle_diff(source_angle(p), angle_deg) > max_angle_dev:
            continue
        if p.axis_major_length < min_len_frac * length:
            continue
        xc, yc = spine_centroid(data_sub, p)
        if comp_xy and min(np.hypot(xc - cx, yc - cy) for cx, cy in comp_xy) < 0.35 * length:
            continue
        stars.append((xc, yc, float(p.image_intensity.sum())))

    stars.sort(key=lambda s: -s[2])
    compact.sort(key=lambda s: -s[2])
    return (np.array(stars).reshape(-1, 3),
            np.array(compact).reshape(-1, 3), angle_deg, data)


def save_png(out_path, data, stars, compact, angle_deg):
    fig, ax = plt.subplots(figsize=(9, 9))
    vmin, vmax = np.percentile(data, [40, 99.5])
    ax.imshow(data, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
    if len(stars):
        X, Y = stars[:, 0], stars[:, 1]
        ax.scatter(X, Y, s=160, facecolors="none", edgecolors="#27e36b", lw=1.5)
        ax.scatter(X, Y, marker="+", c="#27e36b", s=60)
        for i, (x, y) in enumerate(zip(X, Y), start=1):
            ax.annotate(str(i), (x, y), xytext=(6, 6), textcoords="offset points",
                        color="#27e36b", fontsize=7, fontweight="bold")
    if len(compact):
        Xc, Yc = compact[:, 0], compact[:, 1]
        ax.scatter(Xc, Yc, s=200, facecolors="none", edgecolors="#ff3b3b", lw=2.2)
        ax.scatter(Xc, Yc, marker="+", c="#ff3b3b", s=80)
        for j, (x, y) in enumerate(zip(Xc, Yc)):
            tag = "SAT" if j == 0 else "compact"
            ax.annotate(tag, (x, y), xytext=(8, -12), textcoords="offset points",
                        color="#ff3b3b", fontsize=9, fontweight="bold")
    ax.set_title(f"green = {len(stars)} stars   |   red = {len(compact)} compact "
                 f"(SAT = brightest)   |   angle {angle_deg:.1f} deg")
    ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, facecolor="white")
    plt.close(fig)


# ======================================================================
# PLATE SOLVING — offline, via Gaia DR3 pattern matching (no astrometry.net)
# Ported from star_catalog_pipeline_3_1.ipynb: build a rough WCS from the
# header pointing centre + plate scale, query Gaia DR3, lock rotation/flip/
# shift by matching the star pattern to Gaia, then least-squares refine.
# The detected "stars" here are trail centroids (satellite-tracked frame);
# their relative geometry is a rigid transform of the Gaia field at mid-
# exposure, so pattern matching still works and the WCS is valid at mid-exp.
# ======================================================================

PLATE_SOLVE_CFG = dict(
    gaia_maglimit      = 16.0,   # faintest Gaia G to fetch
    gaia_radius_factor = 0.9,    # cone radius = factor * FOV (deg)
    n_match_stars      = 60,     # brightest detected stars used for pattern voting
    n_match_gaia       = 120,    # brightest Gaia stars used for pattern voting
    len_tol_px         = 2.0,    # pair-length tolerance when voting rotation
    rot_bin_deg        = 0.5,    # rotation-vote bin size
    coarse_tol_px      = 3.0,    # inlier tolerance at the coarse stage
    min_matches        = 8,      # minimum matched stars to trust the solve
)


def read_center(hdr):
    """Approximate pointing centre from the header (OBJCTRA/DEC, RA/DEC, or CRVAL)."""
    if hdr.get("OBJCTRA") and hdr.get("OBJCTDEC"):
        return SkyCoord(str(hdr["OBJCTRA"]), str(hdr["OBJCTDEC"]),
                        unit=(u.hourangle, u.deg))
    if hdr.get("RA") is not None and hdr.get("DEC") is not None:
        ra = str(hdr["RA"]); unit = u.hourangle if (":" in ra or " " in ra.strip()) else u.deg
        return SkyCoord(ra, str(hdr["DEC"]), unit=(unit, u.deg))
    if hdr.get("CRVAL1") is not None and hdr.get("CRVAL2") is not None:
        return SkyCoord(float(hdr["CRVAL1"]) * u.deg, float(hdr["CRVAL2"]) * u.deg)
    raise ValueError("No pointing centre in header (need OBJCTRA/DEC, RA/DEC, or CRVAL1/2).")


def plate_scale_arcsec(hdr):
    """arcsec/pixel from pixel pitch XPIXSZ (µm) and FOCALLEN (mm)."""
    pix_um = float(hdr.get("XPIXSZ", 0.0)); fl_mm = float(hdr.get("FOCALLEN", 0.0))
    if pix_um <= 0 or fl_mm <= 0:
        raise ValueError("Need XPIXSZ (µm) and FOCALLEN (mm) in the header to compute the scale.")
    return 206.265 * pix_um / fl_mm


def make_wcs(center, scale, nx, ny, theta_deg=0.0, flip=False):
    """Initial TAN WCS from centre + scale. `flip` negates the first CD column
    (a proper reflection, det -1), not a skew."""
    s = scale / 3600.0; th = np.deg2rad(theta_deg); sx = -1.0 if flip else 1.0
    cd = np.array([[sx * s * np.cos(th), -s * np.sin(th)],
                   [sx * s * np.sin(th),  s * np.cos(th)]])
    w = WCS(naxis=2)
    w.wcs.crpix = [(nx + 1) / 2, (ny + 1) / 2]
    w.wcs.crval = [center.ra.deg, center.dec.deg]
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]; w.wcs.cd = cd
    return w


def query_gaia(center, fov_deg, cfg):
    """Gaia DR3 cone query around the approximate centre."""
    from astroquery.gaia import Gaia
    r = cfg["gaia_radius_factor"] * fov_deg
    q = (f"SELECT ra, dec, phot_g_mean_mag FROM gaiadr3.gaia_source "
         f"WHERE 1=CONTAINS(POINT('ICRS',ra,dec),"
         f"CIRCLE('ICRS',{center.ra.deg},{center.dec.deg},{r})) "
         f"AND phot_g_mean_mag < {cfg['gaia_maglimit']} ORDER BY phot_g_mean_mag")
    t = Gaia.launch_job_async(q).get_results()
    return SkyCoord(np.array(t["ra"]) * u.deg, np.array(t["dec"]) * u.deg)


def _pair_vecs(xy):
    i, j = np.triu_indices(len(xy), k=1); v = xy[j] - xy[i]
    return np.hypot(v[:, 0], v[:, 1]), np.arctan2(v[:, 1], v[:, 0])


def coarse_align(det, cat, cfg):
    """Vote rotation + flip + shift that best overlays the detection pattern on Gaia."""
    dL, dA = _pair_vecs(det); best = None
    abin = np.deg2rad(cfg["rot_bin_deg"]); ltol = cfg["len_tol_px"]; tol = cfg["coarse_tol_px"]
    for flip in (False, True):
        C = cat * np.array([-1.0, 1.0]) if flip else cat
        cL, cA = _pair_vecs(C); order = np.argsort(cL); cLs = cL[order]
        lo = np.searchsorted(cLs, dL - ltol); hi = np.searchsorted(cLs, dL + ltol)
        votes = []
        for p in range(len(dL)):
            qa = cA[order[lo[p]:hi[p]]]
            votes.append(dA[p] - qa); votes.append(dA[p] - qa + np.pi)
        if not votes:
            continue
        votes = np.mod(np.concatenate(votes), 2 * np.pi)
        hist, edges = np.histogram(votes, bins=int(2 * np.pi / abin), range=(0, 2 * np.pi))
        for pk in np.argsort(hist)[-6:]:
            ang = edges[pk] + abin / 2
            R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
            RC = (R @ C.T).T
            shifts = (det[:, None, :] - RC[None, :, :]).reshape(-1, 2)
            key = np.round(shifts / tol).astype(int)
            uniq, cnt = np.unique(key, axis=0, return_counts=True)
            t = uniq[np.argmax(cnt)] * tol
            d, _ = cKDTree(RC + t).query(det, k=1)
            n = int((d < tol).sum())
            if best is None or n > best[0]:
                best = (n, flip, ang, t)
    return best


def refine(det, gaia, gaia_xy, flip, ang, t, center, cfg):
    """Iterate the least-squares WCS fit with shrinking tolerance + outlier rejection."""
    C = gaia_xy * np.array([-1.0, 1.0]) if flip else gaia_xy
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    pred = (R @ C.T).T + t
    d, j = cKDTree(pred).query(det, k=1)
    idx = np.where(d < cfg["coarse_tol_px"])[0]; match = j[d < cfg["coarse_tol_px"]]
    w = None
    for it in range(6):
        if len(idx) < 3:
            break
        w = fit_wcs_from_points((det[idx, 0], det[idx, 1]), gaia[match], proj_point=center)
        ax, ay = w.world_to_pixel(gaia)
        d, j = cKDTree(np.column_stack([ax, ay])).query(det, k=1)
        tol = max(1.5, 6.0 * (0.6 ** it))
        idx = np.where(d < tol)[0]; match = j[d < tol]
        oo = np.argsort(d[idx]); idx = idx[oo]; match = match[oo]
        _, uq = np.unique(match, return_index=True)
        idx = idx[uq]; match = match[uq]
    return w, idx, match


def plate_solve(stars, width, height, header, cfg=None,
                center=None, scale_arcsec=None):
    """Offline plate solve: match the detected star pattern to Gaia DR3 and
    least-squares refine a TAN WCS.

    `header` supplies the approximate pointing centre and the plate scale.
    Override with `center` (a SkyCoord or (ra_deg, dec_deg) tuple) and/or
    `scale_arcsec` if the header lacks OBJCTRA/DEC or XPIXSZ/FOCALLEN.
    Returns an astropy WCS (or None on failure)."""
    cfg = {**PLATE_SOLVE_CFG, **(cfg or {})}
    if stars is None or len(stars) < 4:
        print("    plate solve skipped: need ~4+ stars")
        return None

    try:
        if center is None:
            center = read_center(header)
        elif not isinstance(center, SkyCoord):
            center = SkyCoord(center[0] * u.deg, center[1] * u.deg)
        scale = scale_arcsec if scale_arcsec else plate_scale_arcsec(header)
    except (ValueError, KeyError, TypeError) as e:
        print(f"    plate solve skipped: {e}")
        return None

    fov_deg = scale * max(width, height) / 3600.0
    det = stars[:, :2].astype(float)          # (x, y) trail centroids
    flux = stars[:, 2].astype(float)
    print(f"    plate scale {scale:.4f} arcsec/px, FOV {fov_deg * 60:.2f} arcmin, "
          f"centre {center.to_string('hmsdms')} (approximate is fine)")

    try:
        gaia = query_gaia(center, fov_deg, cfg)
    except Exception as e:
        print(f"    plate solve failed: Gaia query error: {e}")
        return None
    if len(gaia) < 4:
        print(f"    plate solve failed: only {len(gaia)} Gaia stars in field")
        return None
    print(f"    Gaia DR3: {len(gaia)} stars in field")

    w0 = make_wcs(center, scale, width, height)
    gaia_xy = np.column_stack(w0.world_to_pixel(gaia))
    order = np.argsort(flux)[::-1]
    coarse = coarse_align(det[order[:cfg["n_match_stars"]]],
                          gaia_xy[:cfg["n_match_gaia"]], cfg)
    if coarse is None or coarse[0] < 3:
        print("    plate solve failed: no consistent rotation in pattern match")
        return None
    ninl, flip, ang, t = coarse
    print(f"    pattern lock: flip={flip}, rotation={np.rad2deg(ang):.2f} deg, "
          f"seed inliers={ninl}")

    w, idx, match = refine(det, gaia, gaia_xy, flip, ang, t, center, cfg)
    if w is None or len(idx) < cfg["min_matches"]:
        print(f"    plate solve failed: too few matched stars "
              f"({0 if w is None else len(idx)}, need {cfg['min_matches']})")
        return None
    ax, ay = w.world_to_pixel(gaia[match])
    rms = np.sqrt(np.mean((ax - det[idx, 0]) ** 2 + (ay - det[idx, 1]) ** 2)) * scale
    print(f"    SOLVED offline: refined on {len(idx)} stars, RMS {rms:.3f} arcsec")
    return w


# ======================================================================
# TRIANGULATION — two stations, same satellite, same time -> ECEF position
# ======================================================================
def radec_to_unit(ra_deg, dec_deg):
    """Unit line-of-sight vector in the ICRS/GCRS equatorial axes."""
    ra, dec = np.deg2rad(ra_deg), np.deg2rad(dec_deg)
    return np.array([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)])


def closest_point_between_rays(o1, u1, o2, u2):
    """o = ray origin (station GCRS xyz, km); u = unit line-of-sight direction.
    Returns (satellite_xyz_km, miss_distance_km)."""
    u1 = u1 / np.linalg.norm(u1)
    u2 = u2 / np.linalg.norm(u2)
    w0 = o1 - o2
    b, d, e = u1 @ u2, u1 @ w0, u2 @ w0
    denom = 1.0 - b * b
    if denom < 1e-12:
        return None, np.inf
    s = (b * e - d) / denom
    r = (e - b * d) / denom
    p1, p2 = o1 + s * u1, o2 + r * u2
    return 0.5 * (p1 + p2), float(np.linalg.norm(p1 - p2))


def triangulate(radec_a, station_a, radec_b, station_b, time_utc):
    """radec_* = (ra_deg, dec_deg); station_* = (lat_deg, lon_deg, elev_m);
    time_utc = (Y, M, D, h, m, s). Returns a dict with ECEF position, altitude,
    sub-point, etc.  Triangulation is done in the inertial (GCRS) frame, then the
    resulting point is rotated into Earth-fixed ECEF (ITRS) at that instant."""
    from skyfield.api import load, wgs84
    from skyfield.positionlib import Geocentric
    from skyfield.constants import AU_KM
    from skyfield.framelib import itrs

    ts = load.timescale()
    t = ts.utc(*time_utc)

    origins, dirs = [], []
    for (lat, lon, elev), (ra, dec) in ((station_a, radec_a), (station_b, radec_b)):
        st = wgs84.latlon(lat, lon, elevation_m=elev)
        origins.append(st.at(t).position.km)         # station GCRS position (km)
        dirs.append(radec_to_unit(ra, dec))          # line of sight, same axes

    sat_gcrs, miss = closest_point_between_rays(origins[0], dirs[0], origins[1], dirs[1])
    if sat_gcrs is None:
        raise ValueError("Lines of sight parallel -- baseline too short to triangulate.")

    geo = Geocentric(sat_gcrs / AU_KM, t=t)
    sat_ecef = geo.frame_xyz(itrs).km                # Earth-fixed ECEF (compare to calculator)
    sub = wgs84.geographic_position_of(geo)
    ranges = [float(np.linalg.norm(sat_gcrs - o)) for o in origins]
    return dict(xyz_ecef_km=sat_ecef, xyz_gcrs_km=sat_gcrs,
                altitude_km=sub.elevation.km,
                lat_deg=sub.latitude.degrees, lon_deg=sub.longitude.degrees,
                geocentric_km=float(np.linalg.norm(sat_gcrs)),
                slant_km=ranges, miss_km=miss)


def angsep_deg(ra1, dec1, ra2, dec2):
    """Angular separation between two RA/Dec points, in degrees."""
    u1, u2 = radec_to_unit(ra1, dec1), radec_to_unit(ra2, dec2)
    return float(np.rad2deg(np.arccos(np.clip(u1 @ u2, -1.0, 1.0))))


def read_date_obs(fits_path):
    """Read DATE-OBS from a FITS header -> timezone-aware UTC datetime, or None."""
    from datetime import datetime, timezone
    hdr = fits.getheader(fits_path)
    for key in ("DATE-OBS", "DATE_OBS"):
        if key in hdr:
            s = str(hdr[key]).strip().rstrip("Z")
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
    return None


def tle_predict(line1, line2, station, when):
    """Predict RA/Dec + altitude from a TLE at an exact instant (topocentric J2000)."""
    from skyfield.api import load, wgs84, EarthSatellite
    ts = load.timescale()
    t = ts.utc(*when) if isinstance(when, (tuple, list)) else ts.from_datetime(when)
    sat = EarthSatellite(line1, line2, "target", ts)
    lat, lon, elev = station
    obs = wgs84.latlon(lat, lon, elevation_m=elev)
    ra, dec, dist = (sat - obs).at(t).radec()
    geo = sat.at(t)
    gp = wgs84.geographic_position_of(geo)
    return dict(ra_deg=ra._degrees, dec_deg=dec.degrees, range_km=dist.km,
                altitude_km=gp.elevation.km,
                sub_lat=gp.latitude.degrees, sub_lon=gp.longitude.degrees)


def tle_check_frame(label, fits_path, station, measured_radec, line1, line2):
    """Print predicted (TLE at this frame's exact DATE-OBS) vs measured RA/Dec."""
    when = read_date_obs(fits_path)
    if when is None:
        print(f"  [{label}] no DATE-OBS in header — cannot do TLE check")
        return
    p = tle_predict(line1, line2, station, when)
    print(f"  [{label}] {when.isoformat()}")
    print(f"      predicted: RA {p['ra_deg']:.5f}  Dec {p['dec_deg']:.5f}  "
          f"alt {p['altitude_km']:.1f} km  (sub {p['sub_lat']:.3f}, {p['sub_lon']:.3f})")
    if measured_radec is not None:
        mra, mdec = measured_radec
        sep = angsep_deg(mra, mdec, p['ra_deg'], p['dec_deg'])
        print(f"      measured : RA {mra:.5f}  Dec {mdec:.5f}")
        print(f"      offset measured-vs-predicted: {sep * 3600:.1f} arcsec ({sep:.4f} deg)")


# ======================================================================
# SETTINGS — edit these, no command-line arguments needed.
# ======================================================================
FITS_PATH = "/Users/none/internship/fit_files/MERIDIAN 8-001_1s.fit"
ANGLE_DEG = None
LENGTH_PX = 60
NSIGMA    = 4.0
MIN_ELONG = 3.0
CLOSE_FRAC = 0.5
MAX_ANGLE_DEV = 12.0
MIN_LEN_FRAC  = 0.5
BORDER_MARGIN = 10
SAT_BORDER_MARGIN = 60
PNG_DIR    = "/Users/none/internship/png_output"

# --- plate solving (offline: Gaia DR3 pattern match — no astrometry.net) ---
# Needs internet only for the Gaia cone query. The pointing centre and plate
# scale come from the FITS header (OBJCTRA/DEC + XPIXSZ + FOCALLEN). If your
# header lacks those, set the overrides below.
DO_PLATE_SOLVE = True
PLATE_CENTER       = None   # override: (ra_deg, dec_deg) approximate pointing centre
PLATE_SCALE_ARCSEC = None   # override: arcsec/pixel (else from XPIXSZ & FOCALLEN)

# --- triangulation (TWO stations, same satellite, same UTC time) ---
DO_TRIANGULATE = True        # True = process FITS_PATH + FITS_PATH_2 and triangulate
FITS_PATH_2 = "/Users/none/internship/fit_files/MERIDIAN 8-0001_bin4_3s.fit"
STATION_A = (18.5745,  98.4847, 2400.0)
STATION_B = (16.7607, 102.6309,  185.0)
TIME_UTC  = (2025, 3, 28, 14, 58, 29.795)

# --- TLE cross-check (predict RA/Dec + altitude at each frame's exact DATE-OBS) ---
DO_TLE_CHECK = False         # True = compare your measured RA/Dec to a candidate TLE
TLE_LINE1 = "1 44453U 19046A   25085.60272785  .00000220  00000+0  00000+0 0  9999"
TLE_LINE2 = "2 44453  62.7898 105.8254 6894526 273.7052  16.4822  2.00602667 41369"
# ======================================================================


def process_one(fits_path):
    stars, compact, ang, raw = detect_sources(
        fits_path, ANGLE_DEG, LENGTH_PX, NSIGMA,
        min_elong=MIN_ELONG, close_frac=CLOSE_FRAC,
        max_angle_dev=MAX_ANGLE_DEV, min_len_frac=MIN_LEN_FRAC,
        border_margin=BORDER_MARGIN, sat_border_margin=SAT_BORDER_MARGIN)

    top_n = math.ceil(len(stars) * 4 / 5)
    stars = stars[:top_n]

    name = os.path.basename(fits_path)
    print(f"[{name}] {len(stars)} stars, {len(compact)} compact, angle {ang:.1f} deg")
    if len(compact):
        sx, sy, _ = compact[0]
        print(f"    satellite (brightest compact) at (x, y) = ({sx:.2f}, {sy:.2f})")

    stem = os.path.basename(fits_path.rsplit(".", 1)[0])
    out_base = os.path.join(PNG_DIR, stem)
    if len(stars):
        np.savetxt(out_base + "_stars.txt", stars,
                   header="X Y FLUX (brightest first, 0-indexed px)", fmt="%.3f")
    if len(compact):
        np.savetxt(out_base + "_satellite.txt", compact,
                   header="X Y FLUX  (row 0 = brightest = likely satellite)", fmt="%.3f")

    png_path = out_base + "_stars.png"
    save_png(png_path, raw, stars, compact, ang)
    print(f"    -> {png_path}")

    sat_radec = None
    if DO_PLATE_SOLVE and len(stars) >= 4:
        H, W = raw.shape
        header = fits.getheader(fits_path)
        wcs = plate_solve(stars, W, H, header,
                          center=PLATE_CENTER, scale_arcsec=PLATE_SCALE_ARCSEC)
        if wcs is not None:
            cx, cy = np.median(stars[:, 0]), np.median(stars[:, 1])
            c = wcs.pixel_to_world(cx, cy)
            print(f"    SOLVED. star-field center ({cx:.1f},{cy:.1f}) "
                  f"-> RA {c.ra.deg:.5f}  Dec {c.dec.deg:.5f}")
            lines = [f"# field_center_px {cx:.3f} {cy:.3f}",
                     f"# field_center_radec {c.ra.deg:.6f} {c.dec.deg:.6f}"]
            if len(compact):
                sx, sy, _ = compact[0]
                s = wcs.pixel_to_world(sx, sy)
                sat_radec = (s.ra.deg, s.dec.deg)
                print(f"    satellite -> RA {s.ra.deg:.5f}  Dec {s.dec.deg:.5f}")
                lines.append(f"# satellite_px {sx:.3f} {sy:.3f}")
                lines.append(f"# satellite_radec {s.ra.deg:.6f} {s.dec.deg:.6f}")
            with open(out_base + "_wcs.txt", "w") as f:
                f.write("\n".join(lines) + "\n")
            print(f"    -> {out_base}_wcs.txt")
    return sat_radec


if __name__ == "__main__":
    os.makedirs(PNG_DIR, exist_ok=True)

    if DO_TRIANGULATE:
        if not os.path.isfile(FITS_PATH) or not os.path.isfile(FITS_PATH_2):
            raise SystemExit("Both FITS_PATH and FITS_PATH_2 must exist for triangulation.")
        if not DO_PLATE_SOLVE:
            raise SystemExit("Triangulation needs DO_PLATE_SOLVE = True (uses each frame's RA/Dec).")
        print("=== Station A ===")
        radec_a = process_one(FITS_PATH)
        print("=== Station B ===")
        radec_b = process_one(FITS_PATH_2)
        if radec_a is None or radec_b is None:
            raise SystemExit("Could not get a satellite RA/Dec from both frames.")
        r = triangulate(radec_a, STATION_A, radec_b, STATION_B, TIME_UTC)
        print("\n=== Triangulation ===")
        print(f"Satellite ECEF XYZ (km) : {r['xyz_ecef_km'].round(2)}")
        print(f"Altitude above ellipsoid: {r['altitude_km']:.2f} km")
        print(f"Sub-satellite point     : lat {r['lat_deg']:.4f}, lon {r['lon_deg']:.4f}")
        print(f"Geocentric distance     : {r['geocentric_km']:.2f} km")
        print(f"Slant range A / B       : {r['slant_km'][0]:.1f} / {r['slant_km'][1]:.1f} km")
        print(f"Ray miss distance       : {r['miss_km']:.3f} km  (smaller = more consistent)")

        if DO_TLE_CHECK:
            print("\n=== TLE cross-check (each frame at its own DATE-OBS) ===")
            tle_check_frame("A", FITS_PATH,   STATION_A, radec_a, TLE_LINE1, TLE_LINE2)
            tle_check_frame("B", FITS_PATH_2, STATION_B, radec_b, TLE_LINE1, TLE_LINE2)
        print("\nDone.")
    else:
        if not os.path.isfile(FITS_PATH):
            print(f"File not found: {FITS_PATH}")
        else:
            radec_a = process_one(FITS_PATH)
            if DO_TLE_CHECK:
                print("\n=== TLE cross-check (frame at its DATE-OBS) ===")
                tle_check_frame("A", FITS_PATH, STATION_A, radec_a, TLE_LINE1, TLE_LINE2)
            print("\nDone.")
