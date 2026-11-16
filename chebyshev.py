"""
chebyshev.py — 2D-Chebyshev-Fit + DIN-Kennwert-Ableitung
==========================================================
Generisches Kern-Modul für die Flanken-Charakterisierung. Wird von
gear_pipeline_phase4_uv.py (Ist-Fit + Soll-Vergleich pro Cluster) und
gear_pipeline_phase5_6_din.py (DIN-1328-Kennwerte) importiert.

Koeffizienten-Konvention (identisch zu numpy.polynomial.chebyshev):
  coeffs[i, j]  multipliziert  T_i(u) · T_j(v)
  u = normierte Abwälzlänge  ∈ [-1, +1]  (Profilrichtung)
  v = normierte z-Position   ∈ [-1, +1]  (Flankenlinienrichtung)

Diese Konvention ist konsistent mit gear_pipeline_phase4_uv.py's
soll_coeffs(): c[2,0] = Profilballigkeit-Term, c[0,2] = Flankenballigkeit-
Term, c[0,1] = Flankenlinien-Winkel-Term.

Keine MY_GEAR-Abhängigkeit — reine Mathematik, unabhängig von Soll-Werten.
"""

import numpy as np
from numpy.polynomial import chebyshev as _cheb
from numpy.polynomial.chebyshev import Chebyshev as _ChebSeries


def remap_u_domain(coeffs, l_range_from, l_range_to):
    """
    Rechnet eine 2D-Chebyshev-Koeffizientenmatrix, deren u-Achse (Zeilen-
    Index i) aktuell relativ zu l_range_from=(l_min,l_max) normiert ist
    (u=-1 bei l_min, u=+1 bei l_max), EXAKT in die äquivalente Darstellung
    um, deren u-Achse stattdessen relativ zu l_range_to normiert ist.
    v-Achse (Spalten-Index j) bleibt unverändert.

    Das ist eine EXAKTE affine Basisänderung (l_y ist in beiden Fällen
    dieselbe physikalische Größe, nur anders auf [-1,1] normiert) — KEIN
    neuer Fit, kein Extrapolationsproblem. Nutzt numpy's eigene
    Chebyshev-Domain-Konvertierung (Chebyshev(coef, domain=...).convert()),
    die für die hier verwendeten niedrigen Ordnungen numerisch unempfindlich
    ist.

    Wichtig für die Konditionierung: l_range_to sollte innerhalb oder nahe
    l_range_from liegen ("hineinzoomen", Interpolation) — das ist der sichere
    Fall. Der umgekehrte Fall (l_range_to viel breiter als l_range_from,
    "herauszoomen"/Extrapolation) verstärkt jedes Rauschen quadratisch mit
    dem Verhältnis der Fensterbreiten — siehe Diskussion in
    gear_pipeline_phase4_uv.py.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    n = coeffs.shape[0]
    out = np.zeros_like(coeffs)
    for j in range(coeffs.shape[1]):
        c_from = _ChebSeries(coeffs[:, j], domain=list(l_range_from))
        c_to   = c_from.convert(domain=list(l_range_to))
        c_arr  = c_to.coef
        m = min(len(c_arr), n)
        out[:m, j] = c_arr[:m]
    return out


def remap_v_domain(coeffs, z_range_from, z_range_to):
    """
    Identisch zu remap_u_domain(), aber für die v-Achse (Spalten-Index j,
    Flankenlinienrichtung/z) statt der u-Achse (Zeilen-Index i). Nützlich
    wenn die tatsächlich gescannte z-Abdeckung nicht die volle Zahnbreite
    (MY_GEAR.b) erreicht — dasselbe Domain-Kompressions-Problem wie bei
    l_y/u, nur auf der anderen Achse.

    Implementiert per Transposition + remap_u_domain(), da beide Achsen
    dieselbe 1D-Chebyshev-Basiswechsel-Mathematik verwenden — nur je nachdem
    ob man über Zeilen (u) oder Spalten (v) iteriert.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    return remap_u_domain(coeffs.T, z_range_from, z_range_to).T


def fit(u, v, f_y, order, mask=None):
    """
    Least-Squares-Fit eines 2D-Chebyshev-Polynoms (Ordnung `order` je Achse)
    an gestreute (u, v, f_y)-Daten.

    Parameter
    ---------
    u, v, f_y : (N,) Arrays — u,v sollten ≈ [-1,+1] normiert sein
    order     : int — max. Ordnung je Achse → (order+1)×(order+1) Koeffizienten
    mask      : optionale bool-Maske (N,) — nur diese Punkte gehen in den
                Fit ein (z.B. um Kopf-/Fuß-/Rand-Punkte auszuschließen).
                Residuen werden trotzdem für ALLE Eingabepunkte berechnet,
                damit der Aufrufer dieselbe (oder eine andere) Maske später
                erneut anwenden kann.

    Rückgabe
    --------
    coeffs    : (order+1, order+1) ndarray — coeffs[i,j] ↔ T_i(u)·T_j(v)
    rms       : float — RMS der Fit-Residuen auf der Maske
    residuals : (N,) ndarray — f_y − Fit-Wert, für ALLE Eingabepunkte
    """
    u   = np.asarray(u,   dtype=np.float64)
    v   = np.asarray(v,   dtype=np.float64)
    f_y = np.asarray(f_y, dtype=np.float64)

    if mask is None:
        mask = np.ones(len(u), dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)

    n_fit = int(mask.sum())
    n_coeffs = (order + 1) * (order + 1)
    if n_fit < n_coeffs:
        # Zu wenige Punkte für stabilen Fit dieser Ordnung — Nullkoeffizienten,
        # Residuen = f_y selbst (kein Modell), statt hartem Crash.
        coeffs = np.zeros((order + 1, order + 1))
        residuals = f_y.copy()
        rms = float(np.sqrt(np.mean(residuals[mask] ** 2))) if n_fit > 0 else float('nan')
        return coeffs, rms, residuals

    # Design-Matrix: Spalte (i,j) = T_i(u)·T_j(v), passend zu chebval2d-Konvention
    A = _cheb.chebvander2d(u[mask], v[mask], [order, order])
    coeffs_flat, _, _, _ = np.linalg.lstsq(A, f_y[mask], rcond=None)
    coeffs = coeffs_flat.reshape(order + 1, order + 1)

    fitted_all = evaluate(coeffs, u, v)
    residuals  = f_y - fitted_all
    rms = float(np.sqrt(np.mean(residuals[mask] ** 2)))

    return coeffs, rms, residuals


def evaluate(coeffs, u, v):
    """
    Wertet ein 2D-Chebyshev-Polynom mit gegebenen Koeffizienten an (u,v) aus.

    coeffs : (n, n) — coeffs[i,j] ↔ T_i(u)·T_j(v)
    u, v   : gleich-geformte Arrays (1D Punktliste ODER 2D Gitter via meshgrid)

    Rückgabe: Array derselben Form wie u/v.
    """
    coeffs = np.asarray(coeffs, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    return _cheb.chebval2d(u, v, coeffs)


def din_from_coeffs(dc, eval_grid_n=30):
    """
    Leitet vereinfachte DIN-1328-Kennwerte aus einer Δc-Koeffizientenmatrix
    (Ist − Soll, oder direkt Ist wenn kein Soll bekannt) ab.

    Formeln (siehe Docstring gear_pipeline_phase5_6_din.py, Paper II Abschnitt 3):
      C_α  = -2·c[2,0]     Profilballigkeit
      C_β  = -2·c[0,2]     Flankenballigkeit
      f_Hβ = +2·c[0,1]     Flankenlinien-Winkelmodifikation
      f_Hα = +2·c[1,0]     Profilwinkel-Modifikation
      Δr   = -c[0,0]       konstanter Offset (Pitch/Zahndicke)
      Fα   = größte Spannweite von Δf_y über u, bei festem v (worst-case über alle v)
      Fβ   = größte Spannweite von Δf_y über v, bei festem u (worst-case über alle u)

    Parameter
    ---------
    dc          : (n, n) Koeffizientenmatrix [mm]
    eval_grid_n : Auflösung des Auswertungsgitters für Fα/Fβ

    Rückgabe
    --------
    kw : dict skalarer Kennwerte [mm] — C_alpha_mm, C_beta_mm, f_H_alpha_mm,
         f_H_beta_mm, delta_r_mm, F_alpha_mm, F_beta_mm
    dw : (eval_grid_n, eval_grid_n) ndarray — Δf_y-Gitter [mm], indiziert [u,v]
         (meshgrid mit indexing='ij', wie in gear_pipeline_phase4_uv.py)
    """
    dc = np.asarray(dc, dtype=np.float64)
    n_i, n_j = dc.shape

    u_g = np.linspace(-1.0, 1.0, eval_grid_n)
    v_g = np.linspace(-1.0, 1.0, eval_grid_n)
    U, V = np.meshgrid(u_g, v_g, indexing='ij')
    dw = evaluate(dc, U, V)

    C_alpha   = float(-2.0 * dc[2, 0]) if n_i > 2 and n_j > 0 else 0.0
    C_beta    = float(-2.0 * dc[0, 2]) if n_j > 2 and n_i > 0 else 0.0
    f_H_beta  = float( 2.0 * dc[0, 1]) if n_j > 1 and n_i > 0 else 0.0
    f_H_alpha = float( 2.0 * dc[1, 0]) if n_i > 1 and n_j > 0 else 0.0
    delta_r   = float(-dc[0, 0])

    # Spannweiten direkt aus dem ausgewerteten Gitter (robuster als reine
    # Koeffizienten-Extraktion — erfasst auch höhere Ordnungen > 2).
    F_alpha = float((dw.max(axis=0) - dw.min(axis=0)).max())   # über u, je v=const
    F_beta  = float((dw.max(axis=1) - dw.min(axis=1)).max())   # über v, je u=const

    kw = dict(
        C_alpha_mm=C_alpha,
        C_beta_mm=C_beta,
        f_H_alpha_mm=f_H_alpha,
        f_H_beta_mm=f_H_beta,
        delta_r_mm=delta_r,
        F_alpha_mm=F_alpha,
        F_beta_mm=F_beta,
    )
    return kw, dw
