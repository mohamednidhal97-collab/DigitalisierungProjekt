"""
Zahnrad-Pipeline — Phase 4
===========================
Für jeden Flanken-Cluster:

  MAKRO (Soll-Ist-Vergleich in Evolventen-Koordinaten):
    u  = normierte Abwälzlänge  ∈ [-1, +1]   (Profil)
    v  = normierte z-Position   ∈ [-1, +1]   (Flankenlinie)
    f_y = r_b · flank · (φ_b,ist − φ_b,ref)  (Normalabweichung in mm)

    c_ist  = Chebyshev-Fit auf f_y(u,v)
    c_soll = analytisch aus GearParameters (C_α, C_β, f_Hβ)
    dc     = c_ist − c_soll  →  DIN-Kennwerte

  MIKRO (PCA für Residuen / Pitting):
    Residuen = f_y − f_y_fit  →  Heatmap, später Pitting-Erkennung

k_star wird datengetrieben verfeinert (ohne Sollparameter).
"""

import numpy as np
import h5py
import time
from pathlib import Path

from gear_soll import GearParameters, compute_derived_geometry, generate_soll_flank
from gear_soll import MY_GEAR
from chebyshev import fit as cheb_fit, evaluate as cheb_eval, din_from_coeffs
from chebyshev import remap_u_domain, remap_v_domain
from config import HDF5, FIGURES_DIR, CHEB_ORDER
from utils import log, robust_mad_outlier_mask

CFG = dict(
    input_hdf5     = str(HDF5),
    output_fig_dir = str(FIGURES_DIR),
    cheb_order     = CHEB_ORDER,
    uv_margin      = 0.05,    # Punkte bis ±1.05 für den Fit einbeziehen
    eval_grid_n    = 30,
    min_pts        = 50,
    ly_lo_pct      = 0.5,     # Perzentile pro Cluster für daten-getriebenen l_y-Bereich
    ly_hi_pct      = 99.5,
    z_lo_pct       = 0.5,     # Perzentile pro Cluster für daten-getriebenen z-Bereich (v-Achse)
    z_hi_pct       = 99.5,
)


# ─────────────────────────────────────────────────────────────────────────────
# l_y-Auswertebereich: daten-getrieben statt aus MY_GEAR.d_Ff/d_Nf geschätzt
# ─────────────────────────────────────────────────────────────────────────────

def compute_data_driven_ly_range(hdf5_file, r_b, lo_pct=0.5, hi_pct=99.5):
    """
    Bestimmt [l_y_min, l_y_max] aus der tatsächlich gescannten Punktwolke,
    statt aus MY_GEAR.d_Ff/d_Nf (die ohne bekanntes Gegenrad nur grob
    geschätzt werden können: d_Nf ≈ d_b + 1.0·m_n, s. gear_soll.py).

    Grund: die Soll-Schätzung nimmt oft einen Auswertebereich an, der viel
    näher am Grundkreis liegt als der Scan real abdeckt. Dann wird u beim
    Fit in ein schmales, off-center Teilintervall von [-1,+1] komprimiert
    (nie näher als u≈-0.08 an -1 heran), was die Tschebyscheff-Basis
    schlecht konditioniert: die Fit-RMS bleibt klein (das Polynom trifft
    die lokal beobachteten Daten weiterhin gut), aber die Koeffizienten
    selbst werden instabil und um Größenordnungen zu groß (das war die
    Ursache des Skalenfehlers in din_kennwerte.csv).

    Robust gegen einzelne verrauschte/unvollständige Cluster: pro Cluster
    werden die [lo_pct, hi_pct]-Perzentile von l_y bestimmt, anschließend
    wird der MEDIAN dieser Cluster-Perzentile als gemeinsamer, für alle
    166 Cluster identischer Bereich verwendet (ein einzelner besonders
    kurzer/langer Cluster darf den gemeinsamen Bereich nicht verzerren).
    """
    lo_vals, hi_vals = [], []
    for key in hdf5_file.keys():
        if not key.startswith('tooth_'):
            continue
        g = hdf5_file[key]
        if not bool(g.attrs.get('valid', False)):
            continue
        r = g['pts_cyl'][:, 0].astype(np.float64)
        if len(r) < 50:
            continue
        l_y = np.sqrt(np.abs(r**2 - r_b**2))
        lo_vals.append(np.percentile(l_y, lo_pct))
        hi_vals.append(np.percentile(l_y, hi_pct))

    if not lo_vals:
        return None, None

    return float(np.median(lo_vals)), float(np.median(hi_vals))


# ─────────────────────────────────────────────────────────────────────────────
# z-Auswertebereich (v-Achse): daten-getrieben statt aus MY_GEAR.b angenommen
# ─────────────────────────────────────────────────────────────────────────────

def compute_data_driven_z_range(hdf5_file, lo_pct=0.5, hi_pct=99.5):
    """
    Bestimmt [z_min, z_max] aus der tatsächlich gescannten Punktwolke, statt
    anzunehmen, dass z die volle Zahnbreite MY_GEAR.b von 0 bis b abdeckt.

    Exakt dasselbe Domain-Kompressions-Problem wie bei l_y/u
    (compute_data_driven_ly_range oben), nur auf der v-Achse: wenn der Scan
    nicht bis an beide Ränder der Zahnbreite heranreicht (oder b geringfügig
    größer als die tatsächlich erfasste Breite ist), wird v = 2·z/b − 1 nie
    ganz ±1 erreichen — sichtbar als Weißraum am Rand der (u,v)-Heatmaps.
    Da c_soll's C_β/f_Hβ-Terme (in soll_coeffs()) implizit annehmen, dass
    v die volle Zahnbreite abbildet, ist das dieselbe "Äpfel-mit-Birnen"-
    Domain-Inkonsistenz wie bei l_y — nur eben für die Flankenlinienrichtung
    statt die Profilrichtung.

    Robust wie l_y: Median der [lo_pct, hi_pct]-Perzentile pro Cluster.
    """
    lo_vals, hi_vals = [], []
    for key in hdf5_file.keys():
        if not key.startswith('tooth_'):
            continue
        g = hdf5_file[key]
        if not bool(g.attrs.get('valid', False)):
            continue
        z = g['pts_cyl'][:, 2].astype(np.float64)
        if len(z) < 50:
            continue
        lo_vals.append(np.percentile(z, lo_pct))
        hi_vals.append(np.percentile(z, hi_pct))

    if not lo_vals:
        return None, None

    return float(np.median(lo_vals)), float(np.median(hi_vals))


# ─────────────────────────────────────────────────────────────────────────────
# k_star datengetrieben verfeinern (ohne Sollparameter)
# ─────────────────────────────────────────────────────────────────────────────

def _refine_k_star(hdf5_file, k0, r_b, l_y_min, l_y_max,
                    n_grid=60, search_frac=0.05):
    """
    Minimiert std(phi_b_ist) über einen repräsentativen Cluster.
    Auf der idealen Evolventenflanke ist phi_b_ist = theta_corr + inv_at konstant.
    Der k der diese Streuung minimiert ist der korrekte Helix-Koeffizient.
    """
    keys = sorted(k for k in hdf5_file.keys()
                  if k.startswith('tooth_') and k.endswith('_R'))
    if not keys:
        return k0
    sample_key = keys[len(keys) // 2]
    if not bool(hdf5_file[sample_key].attrs.get('valid', False)):
        sample_key = next((k for k in keys
                           if hdf5_file[k].attrs.get('valid', False)), None)
    if sample_key is None:
        return k0

    cyl = hdf5_file[sample_key]['pts_cyl'][:]
    r, theta, z = (cyl[:, i].astype(np.float64) for i in range(3))
    l_y = np.sqrt(np.abs(r**2 - r_b**2))
    mask = (l_y >= l_y_min * 0.99) & (l_y <= l_y_max * 1.01)
    if mask.sum() < 100:
        return k0

    r_m, theta_m, z_m = r[mask], theta[mask], z[mask]
    alpha_t = np.arccos(np.clip(r_b / np.maximum(r_m, r_b * 1.001), -1, 1))
    inv_at  = np.tan(alpha_t) - alpha_t

    def phi_b_std(k):
        tc = (theta_m - k * z_m + np.pi) % (2 * np.pi) - np.pi
        return np.std(tc + inv_at)

    k_range = np.linspace(k0 * (1 - search_frac), k0 * (1 + search_frac), n_grid)
    k_best  = k_range[np.argmin([phi_b_std(k) for k in k_range])]

    fine = np.linspace(k_best * 0.995, k_best * 1.005, n_grid)
    return float(fine[np.argmin([phi_b_std(k) for k in fine])])


# ─────────────────────────────────────────────────────────────────────────────
# Kern: (u, v, f_y) für einen Cluster
# ─────────────────────────────────────────────────────────────────────────────

def cluster_to_uv_fy(pts_cyl, derived, soll_params, k_star, side, tc_ist):
    """
    Berechnet (u, v, f_y) für alle Punkte eines Flanken-Clusters.

    u   = normierte Abwälzlänge ∈ [-1,+1]  (Profilrichtung)
    v   = normierte z-Position  ∈ [-1,+1]  (Flankenlinienrichtung)
    f_y = Normalabweichung [mm] von der Soll-Evolvente am Ist-Zahnzentrum

    tc_ist: Ist-Zahnmittelwinkel (helix-korrigiert) aus tooth_angles[ti]
    """
    r      = pts_cyl[:, 0].astype(np.float64)
    theta  = pts_cyl[:, 1].astype(np.float64)
    z      = pts_cyl[:, 2].astype(np.float64)

    r_b     = derived['r_b']
    l_y_min = derived['l_y_min']
    l_y_max = derived['l_y_max']
    # z_min/z_max: daten-getrieben falls in `derived` gesetzt (siehe
    # compute_data_driven_z_range()), sonst Fallback auf die volle
    # theoretische Zahnbreite [0, b] (true domain, für derived_true).
    z_min   = derived.get('z_min', 0.0)
    z_max   = derived.get('z_max', soll_params.b)
    inv_at0 = derived['inv_alpha_t0']
    theta_t = derived['theta_t']

    flank = +1 if side == 'R' else -1

    # u aus Abwälzlänge
    l_y = np.sqrt(np.abs(r**2 - r_b**2))
    u   = 2.0 * (l_y - l_y_min) / (l_y_max - l_y_min) - 1.0

    # v aus z (daten-getriebener Bereich, analog zu u/l_y)
    v = 2.0 * (z - z_min) / (z_max - z_min) - 1.0

    # Helix-korrigierter Winkel
    theta_corr = (theta - k_star * z + np.pi) % (2 * np.pi) - np.pi

    # Evolventenwinkel
    alpha_t = np.arccos(np.clip(r_b / np.maximum(r, r_b * 1.001), -1.0, 1.0))
    inv_at  = np.tan(alpha_t) - alpha_t

    # phi_b,ist: Grundkreis-Ursprungswinkel des Ist-Punkts
    phi_b_ist = theta_corr + flank * inv_at

    # phi_b,ref: Median über Auswertungsbereich (zentriert f_y auf 0
    # für ideale Flanke, Pitch-Fehler geht in c00)
    mask_eval = (np.abs(u) <= 1.0) & (np.abs(v) <= 1.0)
    if mask_eval.sum() < 50:
        mask_eval = np.ones(len(u), dtype=bool)
    phi_b_ref = float(np.median(phi_b_ist[mask_eval]))

    # f_y = Normalabweichung
    delta_phi = (phi_b_ist - phi_b_ref + np.pi) % (2 * np.pi) - np.pi
    f_y = flank * r_b * delta_phi

    return dict(
        u=u.astype(np.float32),
        v=v.astype(np.float32),
        f_y=f_y.astype(np.float32),
        phi_b_ref=phi_b_ref,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Soll-Koeffizienten analytisch
# ─────────────────────────────────────────────────────────────────────────────

def soll_coeffs(soll_params, order):
    """
    Analytische Chebyshev-Koeffizienten der Soll-Modifikationen.

    Soll-Fläche: f_y_soll(u,v) = -C_α·u² - C_β·v² + (f_Hα/2)·u + (f_Hβ/2)·v
    In Chebyshev-Basis: T_2(u) = 2u²-1, T_1(u) = u, T_1(v) = v

    c_soll[0,0] = -(C_α + C_β)/2   (Offset aus parabolischen Termen)
    c_soll[2,0] = -C_α/2            (Profilballigkeit)
    c_soll[0,2] = -C_β/2            (Flankenballigkeit)
    c_soll[1,0] = +f_Hα/2           (Profilwinkel-Modifikation)
    c_soll[0,1] = +f_Hβ/2           (Flankenlinien-Winkel)
    """
    n = order + 1
    c = np.zeros((n, n))
    # Alle vier DIN-Parameter sind reguläre GearParameters-Felder (s.
    # gear_soll.py) -- einheitlich direkt zugegriffen statt teils per
    # getattr(..., 0.0) mit Fallback (frühere Inkonsistenz: nur f_H_alpha
    # hatte einen Default, C_alpha/C_beta/f_H_beta nicht, obwohl alle vier
    # gleich behandelt werden sollten).
    Ca  = soll_params.C_alpha   / 1000.0   # µm → mm
    Cb  = soll_params.C_beta    / 1000.0
    fHa = soll_params.f_H_alpha / 1000.0
    fHb = soll_params.f_H_beta  / 1000.0
    c[0, 0] = -(Ca + Cb) / 2.0
    if n > 2:
        c[2, 0] = -Ca / 2.0
        c[0, 2] = -Cb / 2.0
    if n > 1:
        c[1, 0] = +fHa / 2.0
        c[0, 1] = +fHb / 2.0
    return c


# ─────────────────────────────────────────────────────────────────────────────
# Teilungsabweichung f_p / F_p (DIN 3962 / ISO 1328)
# ─────────────────────────────────────────────────────────────────────────────

def compute_pitch_deviation(hdf5_file, N_teeth, tooth_angles, r_b):
    """
    Individuelle (f_p) und kumulative (F_p) Teilungsabweichung je Flankenseite.

    WICHTIG: rein SELBST-referenziert auf die tatsaechlich gemessenen Flanken
    dieses einen Zahnrads -- KEIN MY_GEAR, KEIN phi0-Rotations-Alignment
    noetig. Teilungsfehler misst die Gleichmaessigkeit der eigenen Zahnteilung
    (Fertigungsqualitaet), nicht die absolute Winkellage ggue. einem externen
    Soll-Modell -- phi0 (Phase 6) beantwortet eine andere Frage (Flanken-
    FORM-Abgleich gegen ein Soll-Modell) und ist dafuer weder noetig noch
    geeignet (haengt zusaetzlich von MY_GEAR und vom R/L-Konsistenz-Gate ab,
    das nicht immer >=90% erreicht).

    Nutzt `phase4_phi_b_ref` (je Cluster bereits als robuster Median-
    Grundkreis-Bezugswinkel berechnet, s. cluster_to_uv_fy) als Ist-Winkel je
    Zahn/Seite -- NICHT `tooth_angles` (das ist die idealisierte, exakt
    gleichmaessige Zuordnungs-Gitter aus Phase 5b, keine Einzelzahn-Messung;
    dient hier nur als Branch-Referenz zum Unwrap).

    EXZENTRIZITAETS-KORREKTUR (2026-07-15, per synthetischem Ground-Truth-Test
    gefunden und bestaetigt, s. [[project_gear_pipeline]]): `phase4_phi_b_ref`
    traegt -- unabhaengig von jeder echten Teilungsabweichung -- eine kleine,
    glatte 1x/Umdrehung-Komponente (~wenige µm, aus einer nicht perfekt
    zentrierten Rotationsachse; Phase 6 nennt xyz_shift selbst bereits
    "unverifiziert aus Phase 1"). Das ist Standard-Zahnradmesstechnik-Praxis:
    Rundlauf/Exzentrizitaet und echte Teilungsabweichung sind verschiedene
    Groessen und muessen getrennt werden, bevor f_p/F_p berechnet werden --
    sonst dominiert die (harmlose) Exzentrizitaet besonders F_p_total, das als
    kumulative Groesse ueberproportional auf glatte, niederfrequente Anteile
    reagiert (bestaetigt: bei rein zufaelligem, injiziertem Testfehler ohne
    jede Exzentrizitaet zeigte F_p_total trotzdem eine ~2x-Abweichung von der
    bekannten Wahrheit, waehrend f_p_max nur ~10% abwich -- Pro-Zahn-Vergleich
    zeigte eine klare, glatte 1x/rev-Restkurve in den Fehlern). Fix: 1x/rev-
    Sinuskomponente (kleinste Quadrate ueber den idealen Zahn-Index) wird VOR
    der Schritt-Berechnung herausgefittet und abgezogen.

    Rueckgabe je Seite (None falls <3 valide Zaehne):
      f_p_um       : Array, individuelle Teilungsabweichung je EINZEL-Schritt
                     (nur Schritte zwischen unmittelbar benachbarten validen
                     Zaehnen -- Luecken-ueberspannende Schritte fliessen in
                     F_p_cum_um mit ein, aber nicht in f_p_um/f_p_max_um)
      f_p_max_um   : max|f_p| ueber alle Einzel-Schritte
      F_p_cum_um   : kumulative Teilungsabweichung (Rundlauf-Kurve), startend
                     bei 0 am ersten validen Zahn
      F_p_total_um : max(F_p_cum) - min(F_p_cum) -- die DIN-Kennzahl
      n_gaps       : Anzahl Luecken-ueberspannender Schritte (invalide Zaehne
                     dazwischen)
      eccentricity_um : Amplitude der herausgefitteten 1x/rev-Komponente
                     (Diagnosewert, nicht Teil der DIN-Kennwerte)
    """
    pitch_soll = 2.0 * np.pi / N_teeth
    results = {}
    for side in ['R', 'L']:
        valid_ti, phi_meas = [], []
        for ti in range(N_teeth):
            key = f"tooth_{ti:03d}_{side}"
            if key not in hdf5_file:
                continue
            g = hdf5_file[key]
            if not bool(g.attrs.get('phase4_valid', False)):
                continue
            valid_ti.append(ti)
            phi_meas.append(float(g.attrs['phase4_phi_b_ref']))

        if len(valid_ti) < 3:
            results[side] = None
            continue

        valid_ti = np.array(valid_ti, dtype=int)
        phi_meas = np.array(phi_meas, dtype=np.float64)

        # Unwrap in die Naehe des idealen Gitters -- sicher, da ein echter
        # Teilungsfehler >> eine halbe Teilung ein nicht-funktionsfaehiges
        # Zahnrad waere (tooth_angles dient nur als Branch-Referenz).
        phi_unwrapped = tooth_angles[valid_ti] + (
            ((phi_meas - tooth_angles[valid_ti]) + np.pi) % (2 * np.pi) - np.pi)

        # Exzentrizitaets-/Rundlauf-Korrektur: 1x/Umdrehung-Sinuskomponente
        # per kleinste Quadrate herausfitten und abziehen (s. Docstring oben).
        # WICHTIG: der Fit laeuft auf dem RESIDUUM (phi_unwrapped minus die
        # ideale, gleichmaessige Teilung), NICHT auf phi_unwrapped selbst --
        # phi_unwrapped enthaelt noch die volle lineare Rampe ueber den
        # Zahn-Index (0..2π), die eine einzelne 1x/rev-Sinuskomponente nicht
        # darstellen kann; ein Fit direkt darauf liefert bedeutungslose
        # Radiant-Grosse-1-Koeffizienten (Sawtooth-Fourier-Artefakt), die als
        # "Exzentrizitaet in µm" um Groessenordnungen zu gross erscheinen.
        # theta_ideal nutzt den IDEALEN Zahn-Index (nicht die verrauschte
        # Messung selbst) als Bezug -- Standard-Konvention, da Exzentrizitaet
        # eine Eigenschaft der Rotationslage ist, nicht der (winzigen)
        # Teilungsabweichung, die sie ueberlagert.
        theta_ideal = 2.0 * np.pi * valid_ti.astype(np.float64) / N_teeth
        residual = phi_unwrapped - tooth_angles[valid_ti]   # klein (Bruchteil eines Grads)
        _A = np.column_stack([np.ones_like(theta_ideal), np.cos(theta_ideal), np.sin(theta_ideal)])
        _coef, *_ = np.linalg.lstsq(_A, residual, rcond=None)
        _a_fit, _b_fit = _coef[1], _coef[2]
        eccentricity_rad = _a_fit * np.cos(theta_ideal) + _b_fit * np.sin(theta_ideal)
        phi_unwrapped = phi_unwrapped - eccentricity_rad
        eccentricity_um = float(np.hypot(_a_fit, _b_fit) * r_b * 1000.0)

        n = len(valid_ti)
        step_dev = np.zeros(n)      # rad, Schritt zum jeweils naechsten validen Zahn
        n_slots  = np.zeros(n, dtype=int)
        for j in range(n):
            ti_a, ti_b = valid_ti[j], valid_ti[(j + 1) % n]
            ang_a, ang_b = phi_unwrapped[j], phi_unwrapped[(j + 1) % n]
            if j == n - 1:                          # Schluss-Schritt (Vollkreis)
                slots = (N_teeth - ti_a) + ti_b
                ang_b = ang_b + 2 * np.pi
            else:
                slots = ti_b - ti_a
            step_dev[j] = (ang_b - ang_a) - slots * pitch_soll
            n_slots[j]  = slots

        # Kumulative Teilungsabweichung: Rundlauf-Summe, verankert bei 0 am
        # ersten validen Zahn (Wahl des Ankers ist beliebig -- F_p_total ist
        # als max-min invariant dagegen).
        F_p_cum = np.concatenate([[0.0], np.cumsum(step_dev[:-1])])

        single_mask = n_slots == 1
        f_p_um     = step_dev[single_mask] * r_b * 1000.0
        F_p_cum_um = F_p_cum * r_b * 1000.0

        results[side] = dict(
            valid_ti=valid_ti,
            n_slots=n_slots,
            f_p_um=f_p_um,
            f_p_max_um=float(np.max(np.abs(f_p_um))) if len(f_p_um) else float('nan'),
            eccentricity_um=eccentricity_um,
            F_p_cum_um=F_p_cum_um,
            F_p_total_um=float(F_p_cum_um.max() - F_p_cum_um.min()),
            n_gaps=int(np.sum(n_slots > 1)),
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# HAUPTFUNKTION
# ─────────────────────────────────────────────────────────────────────────────

def process_all_clusters(cfg, soll_params):
    """
    EIN Chebyshev-Fit pro Cluster, über den daten-getriebenen (tatsächlich
    erreichten) l_y-Bereich (u-Achse) UND z-Bereich (v-Achse) — numerisch
    gut konditioniert, wie schon vor der kurzen "order_din"-Zwischenlösung.

    Beide Achsen leiden potenziell am selben Domain-Kompressions-Problem:
    u, wenn der Scan nicht bis an den theoretischen d_Nf/d_Ff-Rand heranreicht
    (behoben via compute_data_driven_ly_range()); v, wenn der Scan nicht die
    volle Zahnbreite MY_GEAR.b abdeckt (behoben via
    compute_data_driven_z_range()) — sichtbar z.B. als Weißraum am Rand der
    (u,v)-Heatmaps, wenn nur eine Achse korrigiert wäre.

    Der DIN-Vergleich (c_ist − c_soll) wird trotzdem korrekt (Äpfel mit
    Äpfeln), weil c_soll NICHT mehr direkt aus soll_coeffs() (das die echte/
    theoretische Eingriffsstrecke UND die volle Zahnbreite voraussetzt)
    übernommen wird, sondern per EXAKTER affiner Basisänderung
    (remap_u_domain() für u, remap_v_domain() für v, reine Koordinaten-
    umrechnung, kein neuer Fit) aus dem jeweiligen echten/theoretischen
    Bereich IN den daten-getriebenen Bereich hinein umgerechnet wird — nicht
    umgekehrt.

    Warum diese Richtung sicher ist, die andere nicht: c_ist über die echte
    (volle) Eingriffsstrecke zu fitten hieße, ein Krümmungsmaß aus einem
    schmalen, off-center Datenausschnitt auf ein ~4× breiteres Fenster zu
    EXTRAPOLIEREN — das verstärkt Rauschen um (W_true/W_data)² (hier ≈14.5×,
    empirisch bestätigt: siehe Chat-Diskussion vom selben Tag). c_soll
    dagegen aus dem vollen Fenster in das schmale, tatsächlich gemessene
    Fenster HINEIN zu zoomen ist der sichere Interpolations-Fall — dieselbe
    Rechnung, aber in die stabile Richtung, mit demselben Faktor als
    SCHRUMPFUNG statt Verstärkung. Die kleine (~1-9µm) reale Soll-Balligkeit
    wird dadurch, korrekt, auf einen entsprechend winzigen (~1/14.5) Anteil
    innerhalb des gemessenen Fensters herunterskaliert — meist unterhalb des
    Rauschbodens, aber das ist genau das erwartete, korrekte Ergebnis.

    Wichtige Annahme (im Bericht zu nennen): der Grundkreis-nahe Teil des
    Profils (Fuß) wird vom Scan nicht erfasst — nur die daten-getriebene
    Region wird ausgewertet; die Punktwolke wird als hinreichend sauber
    (repräsentativ für die tatsächlich erreichbare Flanke) angenommen.
    """
    t0 = time.time()
    print("\n" + "=" * 62)
    print("  PHASE 4: Evolventen-Koordinaten + Soll-Ist-Vergleich")
    print("=" * 62)

    derived_true = compute_derived_geometry(soll_params)
    # v-Achse: true domain ist die volle Zahnbreite [0, b] (siehe
    # generate_soll_flank(): Z = (V+1)/2·b läuft von 0 bis b). Wird unten,
    # analog zu l_y_min/max, ggf. durch die daten-getriebene Abdeckung
    # ersetzt (derived_data).
    derived_true['z_min'] = 0.0
    derived_true['z_max'] = soll_params.b
    order   = cfg["cheb_order"]
    margin  = cfg["uv_margin"]

    log(f"Soll-Geometrie (ECHTE Eingriffsstrecke, Quelle für c_soll-Remap): "
        f"r_b={derived_true['r_b']:.3f} mm  "
        f"l_y=[{derived_true['l_y_min']:.3f}, {derived_true['l_y_max']:.3f}] mm  "
        f"b={soll_params.b} mm")
    log(f"Chebyshev-Ordnung: {order}  ({order+1}×{order+1})")

    # Soll-Koeffizienten analytisch über der ECHTEN Eingriffsstrecke
    # (so wie C_alpha/C_beta/f_H_alpha/f_H_beta per DIN-Konvention definiert
    # sind) — wird weiter unten, sobald der daten-getriebene Bereich bekannt
    # ist, EXAKT in diesen Bereich umgerechnet.
    c_soll_true = soll_coeffs(soll_params, order)
    log(f"Soll (echte Eingriffsstrecke) c_α={c_soll_true[2,0]*1000:+.3f} µm  "
        f"c_β={c_soll_true[0,2]*1000:+.3f} µm  "
        f"c_Hα={c_soll_true[1,0]*1000:+.3f} µm  "
        f"c_Hβ={c_soll_true[0,1]*1000:+.3f} µm")

    with h5py.File(cfg["input_hdf5"], "r+") as f:
        meta = f['metadata']
        N_teeth     = int(meta.attrs['N_teeth'])
        tooth_angles = np.array(meta.attrs['tooth_angles'])
        k_star_fft  = float(meta.attrs['k_star'])
        meta.attrs['cheb_order'] = order

        # ── l_y-Bereich: daten-getrieben (tatsächlich erreichte Scan-Abdeckung) ──
        # Dies ist der EINZIGE Bereich, über den jetzt gefittet wird — sowohl
        # Ist (direkt) als auch Soll (per Remap hinein-gerechnet, siehe unten).
        derived_data = dict(derived_true)
        ly_min_data, ly_max_data = compute_data_driven_ly_range(
            f, derived_true['r_b'], cfg.get('ly_lo_pct', 0.5), cfg.get('ly_hi_pct', 99.5))
        if ly_min_data is not None:
            log(f"l_y-Bereich ECHT (theoretische Eingriffsstrecke, Quelle für Soll): "
                f"[{derived_true['l_y_min']:.3f}, {derived_true['l_y_max']:.3f}] mm")
            log(f"l_y-Bereich daten-getrieben (tatsächliche Scan-Abdeckung, "
                f"Perzentile {cfg.get('ly_lo_pct',0.5)}/{cfg.get('ly_hi_pct',99.5)}%, "
                f"Fit- und Vergleichsbereich für Ist UND Soll): "
                f"[{ly_min_data:.3f}, {ly_max_data:.3f}] mm")
            derived_data['l_y_min'] = ly_min_data
            derived_data['l_y_max'] = ly_max_data
            meta.attrs['l_y_min_data'] = ly_min_data
            meta.attrs['l_y_max_data'] = ly_max_data
        else:
            log("⚠ l_y-Bereich konnte nicht daten-getrieben bestimmt werden "
                "(keine gültigen Cluster gefunden) — nutze die echte "
                "Eingriffsstrecke (ggf. komprimiert)")

        # ── z-Bereich (v-Achse): daten-getrieben, analog zu l_y/u ────────────
        # Ohne diese Korrektur wird v = 2z/b−1 nie ±1 erreichen, sobald der
        # Scan nicht bis an beide Ränder der Zahnbreite b heranreicht — sicht-
        # bar als Weißraum am Rand der (u,v)-Heatmaps (Phase4-Figur).
        z_min_data, z_max_data = compute_data_driven_z_range(
            f, cfg.get('z_lo_pct', 0.5), cfg.get('z_hi_pct', 99.5))
        if z_min_data is not None:
            log(f"z-Bereich ECHT (volle Zahnbreite, Quelle für Soll): "
                f"[{derived_true['z_min']:.3f}, {derived_true['z_max']:.3f}] mm")
            log(f"z-Bereich daten-getrieben (tatsächliche Scan-Abdeckung, "
                f"Perzentile {cfg.get('z_lo_pct',0.5)}/{cfg.get('z_hi_pct',99.5)}%, "
                f"Fit- und Vergleichsbereich für Ist UND Soll): "
                f"[{z_min_data:.3f}, {z_max_data:.3f}] mm")
            derived_data['z_min'] = z_min_data
            derived_data['z_max'] = z_max_data
            meta.attrs['z_min_data'] = z_min_data
            meta.attrs['z_max_data'] = z_max_data
        else:
            log("⚠ z-Bereich konnte nicht daten-getrieben bestimmt werden "
                "(keine gültigen Cluster gefunden) — nutze die volle "
                "Zahnbreite [0, b] (ggf. komprimiert)")

        # ── c_soll in den daten-getriebenen Bereich hinein umrechnen ─────────
        # EXAKTE affine Basisänderung (kein neuer Fit) — sicherer "Zoom-rein"-
        # Fall, siehe Docstring oben. Falls kein daten-getriebener Bereich
        # gefunden wurde, bleibt derived_data == derived_true und der Remap
        # ist eine Identität (from == to). u- und v-Achse werden unabhängig
        # voneinander umgerechnet (remap_u_domain wirkt nur auf Zeilen/u,
        # remap_v_domain nur auf Spalten/v — Reihenfolge ist egal).
        c_soll = remap_u_domain(
            c_soll_true,
            (derived_true['l_y_min'], derived_true['l_y_max']),
            (derived_data['l_y_min'], derived_data['l_y_max']))
        c_soll = remap_v_domain(
            c_soll,
            (derived_true['z_min'], derived_true['z_max']),
            (derived_data['z_min'], derived_data['z_max']))
        log(f"Soll (in daten-getriebenen u,v-Bereich umgerechnet) "
            f"c_α={c_soll[2,0]*1000:+.3f} µm  c_β={c_soll[0,2]*1000:+.3f} µm  "
            f"c_Hα={c_soll[1,0]*1000:+.3f} µm  c_Hβ={c_soll[0,1]*1000:+.3f} µm")

        # k_star datengetrieben verfeinern (auf Basis der tatsächlichen Abdeckung)
        k_star = _refine_k_star(f, k_star_fft,
                                  derived_data['r_b'],
                                  derived_data['l_y_min'],
                                  derived_data['l_y_max'])
        log(f"k_star: FFT={k_star_fft:.7f}  verfeinert={k_star:.7f}  "
            f"Δ={(k_star - k_star_fft)*1e6:+.1f} ppm")

        # ── Diagnose: prüfe ob Soll-Parameter zum Zahnrad passen ──────────
        log("Diagnose: Soll-Parameter vs. erkannte Ist-Geometrie")
        meta_N   = int(meta.attrs['N_teeth'])
        meta_b   = float(meta.attrs.get('beta_deg', 0))
        meta_r   = float(meta.attrs['r_ref'])
        log(f"  Erkannt (Ist):  N={meta_N}  β={meta_b:.2f}°  r_ref={meta_r:.2f} mm")
        log(f"  Soll-Parameter: N={soll_params.n}  β={soll_params.beta_0_deg:.2f}°  "
            f"r_b={derived_true['r_b']:.2f} mm  b={soll_params.b:.1f} mm")
        if meta_N != soll_params.n:
            log(f"  ⚠ WARNUNG: Zähnezahl stimmt nicht! "
                f"Erkannt={meta_N}  MY_GEAR.n={soll_params.n}")
            log(f"     → Bitte MY_GEAR.n auf {meta_N} setzen!")
        else:
            log(f"  ✓ Zähnezahl stimmt ({meta_N})")

        # Schnell-Plausibilitätscheck auf einem Cluster
        sample_keys = [k for k in f.keys() if k.endswith('_R') and
                        bool(f[k].attrs.get('valid', False))]
        if sample_keys:
            g_s = f[sample_keys[len(sample_keys)//2]]
            cyl_s = g_s['pts_cyl'][:1000].astype(np.float64)
            r_s, theta_s, z_s = cyl_s[:,0], cyl_s[:,1], cyl_s[:,2]
            l_y_s = np.sqrt(np.abs(r_s**2 - derived_data['r_b']**2))
            u_s   = 2*(l_y_s - derived_data['l_y_min'])/(derived_data['l_y_max'] - derived_data['l_y_min']) - 1
            v_s   = 2*(z_s - derived_data['z_min'])/(derived_data['z_max'] - derived_data['z_min']) - 1
            alpha_t_s = np.arccos(np.clip(derived_data['r_b']/np.maximum(r_s, derived_data['r_b']*1.001),-1,1))
            inv_at_s  = np.tan(alpha_t_s) - alpha_t_s
            tc_s  = (theta_s - k_star*z_s + np.pi)%(2*np.pi) - np.pi
            phi_b_s = tc_s + inv_at_s
            phi_b_std_mm = np.std(phi_b_s[(np.abs(u_s)<=1)&(np.abs(v_s)<=1)]) * derived_data['r_b'] * 1000
            log(f"  phi_b Streuung auf Beispiel-Cluster: {phi_b_std_mm:.1f} µm "
                f"(ideal: < 10 µm, bei 5 µm Rauschen)")
            if phi_b_std_mm > 50:
                log(f"  ⚠ WARNUNG: phi_b-Streuung zu groß!")
                log(f"     → Mögliche Ursache: falsches d_b, falsches n, oder falsches b")
            # u/v sind jetzt beide relativ zum daten-getriebenen Bereich
            # normiert (l_y_min/max bzw. z_min/max aus derived_data) — sollten
            # also per Konstruktion nahe [-1,+1] liegen. Diese Ausgabe ist ein
            # reiner Plausibilitäts-Sanity-Check (z.B. falls k_star/r_b/n
            # grob falsch wären), keine MY_GEAR.b-Warnung mehr nötig, da b
            # hier gar nicht mehr direkt verwendet wird.
            log(f"  u-Bereich Beispiel: [{u_s.min():.2f}, {u_s.max():.2f}]  "
                f"(soll: nahe [-1, +1])")
            log(f"  v-Bereich Beispiel: [{v_s.min():.2f}, {v_s.max():.2f}]  "
                f"(soll: nahe [-1, +1])")
            if abs(u_s.max()) > 3 or abs(u_s.min()) > 3 or abs(v_s.max()) > 3 or abs(v_s.min()) > 3:
                log(f"  ⚠ WARNUNG: u/v weit außerhalb [-1,+1] trotz daten-"
                    f"getriebenem Bereich — deutet auf falsches r_b, n oder "
                    f"k_star hin, nicht mehr auf MY_GEAR.b/d_Ff/d_Nf.")

            # Abdeckung der ECHTEN Zahnbreite/Eingriffsstrecke durch den Scan
            # (informativ, wie stark der Soll-Remap "hineinzoomen" musste):
            z_span_true = derived_true['z_max'] - derived_true['z_min']
            z_span_data = derived_data['z_max'] - derived_data['z_min']
            ly_span_true = derived_true['l_y_max'] - derived_true['l_y_min']
            ly_span_data = derived_data['l_y_max'] - derived_data['l_y_min']
            log(f"  Abdeckung z (Flankenlinie): {100*z_span_data/z_span_true:.1f}% "
                f"der vollen Zahnbreite b={soll_params.b:.1f}mm")
            log(f"  Abdeckung l_y (Profil):     {100*ly_span_data/ly_span_true:.1f}% "
                f"der theoretischen Eingriffsstrecke")

        # Soll-Zahnzentren-Mapping
        pitch_s    = 2 * np.pi / soll_params.n
        soll_cents = np.arange(soll_params.n) * pitch_s
        i2s = np.array([
            int(np.argmin(
                np.abs(((tooth_angles[ti] - soll_cents) + np.pi) % (2*np.pi) - np.pi)
            ))
            for ti in range(N_teeth)
        ])

        # Soll-c für HDF5 speichern (einmalig in metadata)
        if 'cheb_coeffs_soll' in meta.attrs:
            del meta.attrs['cheb_coeffs_soll']
        meta.attrs['cheb_coeffs_soll'] = c_soll.flatten()

        n_ok = 0
        rms_all = []
        n_trim_total = 0

        for ti in range(N_teeth):
            for side in ['R', 'L']:
                key = f"tooth_{ti:03d}_{side}"
                if key not in f:
                    continue
                g = f[key]
                if not bool(g.attrs.get('valid', False)):
                    continue

                pts_cyl = g['pts_cyl'][:].astype(np.float64)
                if len(pts_cyl) < cfg['min_pts']:
                    g.attrs['phase4_valid'] = False
                    continue

                tc_ist = float(tooth_angles[ti])

                # ── (u, v, f_y) über den daten-getriebenen Bereich ───────
                uvfy_data = cluster_to_uv_fy(
                    pts_cyl, derived_data, soll_params, k_star, side, tc_ist)
                u_d   = uvfy_data['u'].astype(np.float64)
                v_d   = uvfy_data['v'].astype(np.float64)
                fy_d  = uvfy_data['f_y'].astype(np.float64)

                # ── EIN Fit (gut konditioniert, daten-getrieben) ─────────
                mask_fit = (np.abs(u_d) <= 1 + margin) & (np.abs(v_d) <= 1 + margin)
                # Robuster Ausreisser-Filter (dieselbe Regel/Validierung wie
                # in gear_pipeline_ist.py's phase7_build_and_save() fuer
                # F_Sigma/F_beta/F_alpha -- s. robust_mad_outlier_mask()
                # Docstring + project_gear_pipeline.md): ohne diesen Filter
                # ziehen einzelne Zahnkopf/Flankenscheitel-Mehrdeutigkeits-
                # Punkte den Least-Squares-Fit spuerbar von der tatsaechlichen
                # Flankenform weg (quadratische Gewichtung grosser Residuen).
                fit_outlier_mask = np.zeros_like(mask_fit)
                fit_outlier_mask[mask_fit] = robust_mad_outlier_mask(
                    (fy_d[mask_fit] * 1000.0))  # mm -> µm fuer denselben K_MAD-Massstab
                n_trim_fit = int(fit_outlier_mask.sum())
                mask_fit_clean = mask_fit & ~fit_outlier_mask
                c_ist, rms, residuals = cheb_fit(u_d, v_d, fy_d, order, mask=mask_fit_clean)
                dc = c_ist - c_soll
                kw, dw = din_from_coeffs(dc, cfg['eval_grid_n'])

                # ── Speichern ────────────────────────────────────────
                for ds in ['phase4_u', 'phase4_v', 'phase4_fy',
                           'phase4_residuals', 'phase4_c_ist',
                           'phase4_c_soll', 'phase4_dc', 'phase4_dw_grid']:
                    if ds in g:
                        del g[ds]

                g.create_dataset('phase4_u',   data=u_d[mask_fit].astype(np.float32),
                                  compression='gzip')
                g.create_dataset('phase4_v',   data=v_d[mask_fit].astype(np.float32),
                                  compression='gzip')
                g.create_dataset('phase4_fy',  data=fy_d[mask_fit].astype(np.float32),
                                  compression='gzip')
                g.create_dataset('phase4_residuals',
                                  data=residuals[mask_fit].astype(np.float32),
                                  compression='gzip')
                g.create_dataset('phase4_c_ist',  data=c_ist.astype(np.float32))
                g.create_dataset('phase4_c_soll', data=c_soll.astype(np.float32))
                g.create_dataset('phase4_dc',     data=dc.astype(np.float32))
                g.create_dataset('phase4_dw_grid',data=dw.astype(np.float32))

                g.attrs['phase4_valid']   = True
                g.attrs['phase4_rms_mm']  = rms
                g.attrs['phase4_n_pts']   = int(mask_fit_clean.sum())
                g.attrs['phase4_n_pts_trimmed'] = n_trim_fit   # s. robust_mad_outlier_mask()
                g.attrs['phase4_phi_b_ref'] = uvfy_data['phi_b_ref']
                for k_kw, val in kw.items():
                    if k_kw != 'dw_grid':
                        g.attrs[f'phase4_{k_kw}'] = val

                n_ok += 1
                rms_all.append(rms)
                n_trim_total += n_trim_fit

        log(f"Verarbeitet: {n_ok}/{N_teeth*2} Cluster", t0)
        if rms_all:
            log(f"  RMS median={np.median(rms_all)*1000:.2f} µm  "
                f"min={min(rms_all)*1000:.2f}  max={max(rms_all)*1000:.2f}")
        if n_trim_total:
            log(f"  Ausreisser-Filter (Chebyshev-Fit): {n_trim_total} Punkte insgesamt "
                f"vor RMS/c_ist ausgeschlossen (s. robust_mad_outlier_mask())")

        # ── Teilungsabweichung f_p / F_p (DIN 3962/ISO 1328) ──────────────
        # Selbst-referenziert (kein MY_GEAR, kein phi0 -- s. Docstring von
        # compute_pitch_deviation()). r_b: die tatsaechlich gemessene
        # (Phase-4-verfeinerte) Ist-Geometrie, nicht Soll.
        pitch_res = compute_pitch_deviation(f, N_teeth, tooth_angles, derived_data['r_b'])
        log("Teilungsabweichung f_p / F_p (DIN 3962, selbst-referenziert):")
        for side in ['R', 'L']:
            res = pitch_res[side]
            for ds in [f'pitch_{side}_valid_ti', f'pitch_{side}_f_p_um', f'pitch_{side}_F_p_cum_um']:
                if ds in meta:
                    del meta[ds]
            for attr in [f'pitch_{side}_f_p_max_um', f'pitch_{side}_F_p_total_um',
                         f'pitch_{side}_n_gaps', f'pitch_{side}_eccentricity_um']:
                if attr in meta.attrs:
                    del meta.attrs[attr]

            if res is None:
                log(f"  {side}: zu wenige valide Zähne (<3) — übersprungen")
                continue

            meta.create_dataset(f'pitch_{side}_valid_ti', data=res['valid_ti'])
            meta.create_dataset(f'pitch_{side}_f_p_um', data=res['f_p_um'].astype(np.float32))
            meta.create_dataset(f'pitch_{side}_F_p_cum_um', data=res['F_p_cum_um'].astype(np.float32))
            meta.attrs[f'pitch_{side}_f_p_max_um']   = res['f_p_max_um']
            meta.attrs[f'pitch_{side}_F_p_total_um'] = res['F_p_total_um']
            meta.attrs[f'pitch_{side}_n_gaps']       = res['n_gaps']
            meta.attrs[f'pitch_{side}_eccentricity_um'] = res['eccentricity_um']

            gap_note = f"  ({res['n_gaps']} Lücken-Schritte übersprungen für f_p)" if res['n_gaps'] else ""
            log(f"  {side}: f_p_max={res['f_p_max_um']:+.2f} µm  "
                f"F_p_total={res['F_p_total_um']:.2f} µm  "
                f"(1x/rev-Exzentrizität entfernt: {res['eccentricity_um']:.2f} µm Amplitude)  "
                f"({len(res['valid_ti'])}/{N_teeth} Zähne valide){gap_note}")

    return n_ok


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISIERUNG
# ─────────────────────────────────────────────────────────────────────────────

def visualize(cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t0 = time.time()
    print("\nVisualisiere Phase 4 ...")

    with h5py.File(cfg["input_hdf5"], 'r') as f:
        cd = []
        for key in sorted(f.keys()):
            if not key.startswith('tooth_'):
                continue
            g = f[key]
            if not bool(g.attrs.get('phase4_valid', False)):
                continue
            cd.append(dict(
                key=key,
                side='R' if key.endswith('_R') else 'L',
                rms=float(g.attrs['phase4_rms_mm']) * 1000,
                u=g['phase4_u'][:],
                v=g['phase4_v'][:],
                fy=g['phase4_fy'][:] * 1000,
                res=g['phase4_residuals'][:] * 1000,
                dw=g['phase4_dw_grid'][:] * 1000,
            ))
    if not cd:
        print("  Keine Daten")
        return

    rms_med = np.median([c['rms'] for c in cd])
    fig = plt.figure(figsize=(18, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.38)
    fig.suptitle(
        f"Phase 4 — Evolventen-Koordinaten + Soll-Ist-Vergleich\n"
        f"{len(cd)} Cluster  |  f_y RMS median = {rms_med:.2f} µm",
        fontsize=12, fontweight='bold')

    # Zeile 0: f_y Scatter für 3 Beispiel-Cluster
    for i, idx in enumerate([0, len(cd)//2, len(cd)-1]):
        c = cd[idx]
        ax = fig.add_subplot(gs[0, i])
        lim = np.percentile(np.abs(c['fy']), 99)
        sc = ax.scatter(c['u'], c['v'], c=c['fy'], s=2,
                        cmap='RdBu_r', vmin=-lim, vmax=lim)
        ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
        ax.add_patch(plt.Rectangle((-1, -1), 2, 2,
                                    fill=False, edgecolor='k', lw=1))
        ax.set_xlabel("u (Profil)"); ax.set_ylabel("v (Flankenlinie)")
        ax.set_title(f"{c['key']}\nf_y  RMS={c['rms']:.1f}µm", fontsize=9)
        plt.colorbar(sc, ax=ax, label="f_y [µm]")

    # Zeile 0 rechts: Δw-Gitter
    ax = fig.add_subplot(gs[0, 3])
    dw = cd[0]['dw']
    lim_dw = max(np.abs(dw).max(), 1)
    im = ax.imshow(dw.T, origin='lower', extent=[-1, 1, -1, 1],
                   cmap='RdBu_r', vmin=-lim_dw, vmax=lim_dw, aspect='auto')
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax.set_title(f"Δw = Ist−Soll\n{cd[0]['key']}", fontsize=9)
    plt.colorbar(im, ax=ax, label="Δw [µm]")

    # Zeile 1 links: RMS je Cluster
    ax = fig.add_subplot(gs[1, :2])
    rms_R = [c['rms'] for c in cd if c['side'] == 'R']
    rms_L = [c['rms'] for c in cd if c['side'] == 'L']
    x = np.arange(len(rms_R))
    ax.bar(x - 0.2, rms_R, 0.4, label='R', color='#185FA5', alpha=0.85)
    ax.bar(x + 0.2, rms_L, 0.4, label='L', color='#993C1D', alpha=0.85)
    ax.set_ylabel("f_y RMS [µm]")
    ax.set_title("Chebyshev-Fit RMS je Zahn")
    ax.legend(); ax.grid(alpha=0.3, axis='y')

    # Zeile 1 rechts: Residuen-Std (Mikro)
    ax = fig.add_subplot(gs[1, 2:])
    res_stds = [np.std(c['res']) for c in cd]
    ax.plot(res_stds, 'o-', color='#534AB7', ms=3, alpha=0.8)
    ax.axhline(np.median(res_stds), color='k', ls='--', lw=1,
               label=f"Median={np.median(res_stds):.1f}µm")
    ax.set_xlabel("Cluster-Index"); ax.set_ylabel("Residuum-Std [µm]")
    ax.set_title("Mikro-Residuen (Ist − Fit)\nHinweis auf Pitting/Rattermarken")
    ax.legend(); ax.grid(alpha=0.3)

    # Zeile 2: Residuen-Histogramm und Heatmap
    ax = fig.add_subplot(gs[2, 0])
    all_res = np.concatenate([c['res'] for c in cd])
    # 'res' enthaelt Residuen fuer ALLE mask_fit-Punkte (siehe chebyshev.fit()-
    # Docstring), nicht nur den fuer den Fit verwendeten getrimmten Teilsatz --
    # eine kleine Zahl bereits erkannter Kopf-/Fuss-Ambiguitaets-Ausreisser
    # bleibt absichtlich drin (fuer die Pitting-Erkennung). Das zieht den
    # Auto-Achsenbereich weit auseinander, obwohl >99% der Punkte nahe Null
    # liegen. Fix hier: Anzeige auf ±6σ clippen, Ausreisser-Anzahl annotieren
    # statt den irrefuehrend breiten Vollbereich zu zeigen.
    std_res = float(np.std(all_res))
    lim = 6.0 * std_res if std_res > 1e-9 else max(float(np.max(np.abs(all_res))), 1e-3)
    n_out = int(np.sum(np.abs(all_res) > lim))
    ax.hist(all_res, bins=80, range=(-lim, lim), color='#1D9E75', alpha=0.8)
    ax.set_xlim(-lim, lim)
    ax.set_xlabel("Residuum [µm]"); ax.set_ylabel("Anzahl")
    title = f"Alle Residuen (±6σ)\nStd={std_res:.2f} µm  N={len(all_res):,}"
    if n_out:
        title += f"\n{n_out} Ausreisser außerhalb ±6σ nicht dargestellt"
    ax.set_title(title, fontsize=9)

    ax = fig.add_subplot(gs[2, 1])
    c0 = cd[0]
    lim_r = np.percentile(np.abs(c0['res']), 99)
    sc = ax.scatter(c0['u'], c0['v'], c=c0['res'], s=2,
                    cmap='RdBu_r', vmin=-lim_r, vmax=lim_r)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.add_patch(plt.Rectangle((-1, -1), 2, 2,
                                fill=False, edgecolor='k', lw=1))
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax.set_title(f"Mikro-Heatmap\n{c0['key']}", fontsize=9)
    plt.colorbar(sc, ax=ax, label="Residuum [µm]")

    ax = fig.add_subplot(gs[2, 2:]); ax.axis('off')
    ax.text(0.02, 0.98,
            "Phase 4 Architektur\n"
            "─────────────────────────────────────\n"
            "MAKRO (Soll-Ist-Vergleich):\n"
            "  u = normierte Abwälzlänge (Profil)\n"
            "  v = normierte z-Position  (Flankenlinie)\n"
            "  f_y = r_b·flank·(φ_b,ist − φ_b,ref)\n"
            "  c_ist = Chebyshev-Fit auf f_y(u,v)\n"
            "  c_soll = analytisch aus Parametern\n"
            "  dc = c_ist − c_soll → DIN-Kennwerte\n\n"
            "MIKRO (Pitting/Rattermarken):\n"
            "  res = f_y − f_y_fit\n"
            "  → Heatmap je Flanke",
            family='monospace', fontsize=9, va='top')

    Path(cfg["output_fig_dir"]).mkdir(exist_ok=True, parents=True)
    out = Path(cfg["output_fig_dir"]) / "phase4_uv.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"→ {out}", t0)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISIERUNG — Chebyshev-Detail (gefittete Fläche + Koeffizient-Amplitude)
# ─────────────────────────────────────────────────────────────────────────────

def visualize_chebyshev_detail(cfg):
    """
    Ergänzende Diagnose-Abbildung zur bestehenden visualize():
      (A) Heatmap der GEFITTETEN Ist-Fläche auf dichtem (u,v)-Gitter —
          nicht das Streudiagramm der Rohdaten (das zeigt visualize()
          bereits), sondern das ausgewertete Chebyshev-Polynom selbst,
          glatt über [-1,1]×[-1,1].
      (B) Koeffizient-vs-Amplitude: Median |c_ij| über alle Cluster,
          absteigend sortiert — zeigt welche Terme die Flanke dominieren.
          Rot markiert = die 6 Terme mit direkter DIN/Topologie-Bedeutung
          (i+j ≤ 2): c00 (Offset), c10/c01 (Winkel fHα/fHβ),
          c20/c02 (Balligkeit Cα/Cβ), c11 (Twist).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t0 = time.time()
    print("\nVisualisiere Chebyshev-Detail ...")

    with h5py.File(cfg["input_hdf5"], 'r') as f:
        cd = []
        for key in sorted(f.keys()):
            if not key.startswith('tooth_'):
                continue
            g = f[key]
            if not bool(g.attrs.get('phase4_valid', False)):
                continue
            cd.append(dict(
                key=key,
                rms=float(g.attrs['phase4_rms_mm']) * 1000,
                c_ist=g['phase4_c_ist'][:].astype(np.float64),
            ))
    if not cd:
        print("  Keine Daten (Phase 4 zuerst ausführen)")
        return

    n = cd[0]['c_ist'].shape[0]
    grid_n = int(cfg.get("eval_grid_n", 30))
    u_g = np.linspace(-1, 1, grid_n)
    v_g = np.linspace(-1, 1, grid_n)
    U, V = np.meshgrid(u_g, v_g, indexing='ij')

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle("Chebyshev-Fit — Detailansicht (gefittete Fläche + Koeffizienten)",
                 fontsize=12, fontweight='bold')

    # (A) Heatmaps der gefitteten Ist-Fläche — 3 Beispiel-Cluster + Median
    example_idx = [0, len(cd) // 2, len(cd) - 1]
    for i, idx in enumerate(example_idx):
        c = cd[idx]
        surf = cheb_eval(c['c_ist'], U, V) * 1000  # µm
        ax = fig.add_subplot(gs[0, i])
        lim = max(np.percentile(np.abs(surf), 99), 1e-3)
        im = ax.imshow(surf.T, origin='lower', extent=[-1, 1, -1, 1],
                        cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        ax.set_xlabel("u (Profil)"); ax.set_ylabel("v (Flankenlinie)")
        ax.set_title(f"{c['key']}\nGefittete Fläche  RMS={c['rms']:.1f}µm", fontsize=9)
        plt.colorbar(im, ax=ax, label="f_y_fit [µm]")

    c_med = np.median([c['c_ist'] for c in cd], axis=0)
    surf_med = cheb_eval(c_med, U, V) * 1000
    ax = fig.add_subplot(gs[0, 3])
    lim = max(np.percentile(np.abs(surf_med), 99), 1e-3)
    im = ax.imshow(surf_med.T, origin='lower', extent=[-1, 1, -1, 1],
                    cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
    ax.set_xlabel("u"); ax.set_ylabel("v")
    ax.set_title(f"Median-Fläche ({len(cd)} Cluster)", fontsize=9)
    plt.colorbar(im, ax=ax, label="f_y_fit [µm]")

    # (B) Koeffizient-vs-Amplitude — Median |c_ij| über alle Cluster
    ax = fig.add_subplot(gs[1, :])
    c_stack = np.stack([c['c_ist'] for c in cd]) * 1000  # (n_cluster, n, n) µm
    c_med_abs = np.median(np.abs(c_stack), axis=0)
    ij_pairs = [(i, j) for i in range(n) for j in range(n)]
    named_set = {(0, 0), (1, 0), (0, 1), (2, 0), (0, 2), (1, 1)}
    values = c_med_abs.flatten()
    order_idx = np.argsort(-values)
    bar_colors = ['#993C1D' if ij_pairs[k] in named_set else '#185FA5' for k in order_idx]
    labels = [f"({ij_pairs[k][0]},{ij_pairs[k][1]})" for k in order_idx]

    ax.bar(range(len(values)), values[order_idx], color=bar_colors)
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("Median |c_ij| [µm]")
    ax.set_title("Koeffizient-Amplitude (absteigend)  —  rot = DIN/Twist-Terme (i+j≤2)")
    ax.grid(alpha=0.3, axis='y')

    Path(cfg["output_fig_dir"]).mkdir(exist_ok=True, parents=True)
    out = Path(cfg["output_fig_dir"]) / "phase4_chebyshev_detail.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"→ {out}", t0)


# ─────────────────────────────────────────────────────────────────────────────
# VISUALISIERUNG — Teilungsabweichung f_p / F_p
# ─────────────────────────────────────────────────────────────────────────────

def visualize_pitch_deviation(cfg):
    """
    Visualisiert die von compute_pitch_deviation() berechneten und in
    'metadata' gespeicherten f_p/F_p-Ergebnisse (DIN 3962, selbst-
    referenziert). Liest NUR bereits vorhandene HDF5-Daten, berechnet nichts
    neu -- process_all_clusters() muss vorher gelaufen sein.

      (A) F_p_cum vs Zahn-Index (R/L)  -- die "Rundlauf"-Kurve, Vollkreis-
          geschlossen dargestellt (letzter Punkt = erster Punkt wiederholt,
          rein zur Visualisierung, aendert die gespeicherten Daten nicht).
      (B) f_p je Schritt (R/L) als Balkendiagramm -- Schritt-Index, nicht
          Zahn-Index (f_p_um enthaelt nur Einzel-Schritte zwischen valide
          benachbarten Zaehnen, s. Docstring compute_pitch_deviation()).
      (C) Text-Zusammenfassung: f_p_max, F_p_total, Exzentrizitaet, n_gaps.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t0 = time.time()
    print("\nVisualisiere Teilungsabweichung f_p / F_p ...")

    with h5py.File(cfg["input_hdf5"], 'r') as f:
        meta = f['metadata']
        N_teeth = int(meta.attrs['N_teeth'])
        data = {}
        for side in ['R', 'L']:
            vt_key = f'pitch_{side}_valid_ti'
            if vt_key not in meta:
                data[side] = None
                continue
            data[side] = dict(
                valid_ti=np.asarray(meta[vt_key]),
                f_p_um=np.asarray(meta[f'pitch_{side}_f_p_um'], dtype=np.float64),
                F_p_cum_um=np.asarray(meta[f'pitch_{side}_F_p_cum_um'], dtype=np.float64),
                f_p_max_um=float(meta.attrs.get(f'pitch_{side}_f_p_max_um', np.nan)),
                F_p_total_um=float(meta.attrs.get(f'pitch_{side}_F_p_total_um', np.nan)),
                eccentricity_um=float(meta.attrs.get(f'pitch_{side}_eccentricity_um', np.nan)),
                n_gaps=int(meta.attrs.get(f'pitch_{side}_n_gaps', 0)),
            )

    if data['R'] is None and data['L'] is None:
        print("  Keine Teilungsabweichungs-Daten gefunden "
              "(process_all_clusters() zuerst ausführen)")
        return

    colors = dict(R='#185FA5', L='#993C1D')
    fig = plt.figure(figsize=(15, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.32,
                            width_ratios=[2.2, 1])
    fig.suptitle(f"Teilungsabweichung f_p / F_p (DIN 3962, selbst-referenziert)  "
                 f"—  N={N_teeth} Zähne", fontsize=12, fontweight='bold')

    # (A) F_p_cum "Rundlauf"-Kurve, Vollkreis-geschlossen
    ax = fig.add_subplot(gs[0, 0])
    for side in ['R', 'L']:
        d = data[side]
        if d is None:
            continue
        x = np.concatenate([d['valid_ti'], [N_teeth]])   # Vollkreis-Schluss nur fürs Plotten
        y = np.concatenate([d['F_p_cum_um'], [d['F_p_cum_um'][0]]])
        ax.plot(x, y, 'o-', color=colors[side], ms=3, lw=1.2,
                label=f"{side}  F_p_total={d['F_p_total_um']:.2f}µm")
    ax.axhline(0, color='k', lw=0.5, alpha=0.5)
    ax.set_xlabel("Zahn-Index"); ax.set_ylabel("F_p,cum [µm]")
    ax.set_title("Kumulative Teilungsabweichung (Rundlauf-Kurve)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # (B) f_p je Schritt, Balkendiagramm
    ax = fig.add_subplot(gs[1, 0])
    width = 0.4
    for i, side in enumerate(['R', 'L']):
        d = data[side]
        if d is None:
            continue
        x = np.arange(len(d['f_p_um'])) + (i - 0.5) * width
        ax.bar(x, d['f_p_um'], width, color=colors[side], alpha=0.85,
               label=f"{side}  f_p_max={d['f_p_max_um']:+.2f}µm")
    ax.axhline(0, color='k', lw=0.6)
    ax.set_xlabel("Schritt-Index (nicht Zahn-Index, s. Docstring)")
    ax.set_ylabel("f_p [µm]")
    ax.set_title("Individuelle Teilungsabweichung je Schritt")
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis='y')

    # (C) Text-Zusammenfassung
    ax = fig.add_subplot(gs[:, 1]); ax.axis('off')
    lines = [
        "═══════════════════════════════════",
        "  TEILUNGSABWEICHUNG — ÜBERSICHT",
        "═══════════════════════════════════", "",
    ]
    for side in ['R', 'L']:
        d = data[side]
        if d is None:
            lines.append(f"{side}: keine Daten (<3 valide Zähne)")
            continue
        lines += [
            f"{side}-Flanke ({len(d['valid_ti'])}/{N_teeth} Zähne valide):",
            f"  f_p_max      : {d['f_p_max_um']:+7.2f} µm",
            f"  F_p_total    : {d['F_p_total_um']:7.2f} µm",
            f"  Exzentrizität: {d['eccentricity_um']:7.2f} µm  (entfernt vor f_p/F_p;",
            f"                 Amplitude — unterschätzt ggf. echten Achs-",
            f"                 versatz, s. [[project_gear_pipeline]])",
            f"  Lücken-Schritte: {d['n_gaps']}",
            "",
        ]
    lines += [
        "───────────────────────────────────",
        "f_p  = Einzelschritt-Abweichung",
        "F_p  = max−min der kumulativen Kurve",
        "Beide selbst-referenziert (nur eigene",
        "Zahnteilung, kein MY_GEAR nötig).",
    ]
    ax.text(0.0, 0.98, "\n".join(lines), family='monospace', fontsize=9,
            va='top', ha='left')

    Path(cfg["output_fig_dir"]).mkdir(exist_ok=True, parents=True)
    out = Path(cfg["output_fig_dir"]) / "phase4_pitch.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"→ {out}", t0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_all = time.time()
    process_all_clusters(CFG, MY_GEAR)
    visualize(CFG)
    visualize_chebyshev_detail(CFG)
    visualize_pitch_deviation(CFG)
    print(f"\nFERTIG — {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()