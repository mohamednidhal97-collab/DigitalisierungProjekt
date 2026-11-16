"""
gear_pipeline_ist.py — Zahnrad-Pipeline v2
===========================================
Vollständig datengetriebene Ist-Geometrie-Erkennung und Segmentierung.
Soll-Parameter werden NICHT für Segmentierung oder Ist-Erkennung verwendet.

Phase 1  Laden + PCA-Achse                         (kein MY_GEAR)
Phase 2  Zähnezahl N + grober k_star via FFT        (kein MY_GEAR)
Phase 3  Normalen + L/R-Maske via n_tang            (kein MY_GEAR)
Phase 4  Präzises k_star + r_b_ist via phi_b-Opt    (kein MY_GEAR)  ← Kern
Phase 5  Zahn-Zuordnung via phi_b mod Teilung       (kein MY_GEAR)
Phase 6  Soll-Verifikation: Residual-Check          ← erster MY_GEAR-Einsatz
Phase 7  HDF5-Ausgabe (kompatibel mit phase4_uv.py)

Warum dieser Ansatz:
  Die klassische Pipeline (k_star → theta_corr → L/R → Zahn) hat Fehler-
  kaskaden: ein ~0.5° k_star-Fehler verursacht L/R-Asymmetrie, weil das
  Winkelfilter nach Helix-Korrektur zentriert wird.

  Hier:
    • L/R via n_tang: rein lokal, k-unabhängig, immer korrekt
    • k_star + r_b: gemeinsam aus phi_b-Streuung optimiert (2D-Problem)
    • Soll-Modell: nur zur Verifikation, nicht zur Segmentierung

Output HDF5 metadata enthält zusätzlich:
  r_b_ist, k_star_ist, beta_ist, alpha_t0_ist
→ gear_pipeline_phase4_uv.py kann r_b_ist statt MY_GEAR.d_b/2 verwenden.
"""

import numpy as np
import pandas as pd
import h5py
import time
import warnings
from pathlib import Path

from scipy.spatial import KDTree
from scipy.optimize import minimize, minimize_scalar
from scipy.signal import find_peaks
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from config import PUNKTWOLKE, HDF5, FIGURES_DIR
from utils import log, robust_mad_outlier_mask

warnings.filterwarnings("ignore", category=RuntimeWarning,
                         message="Mean of empty slice")
warnings.filterwarnings("ignore", category=RuntimeWarning,
                         message="invalid value in arccos")

CFG = dict(
    input_file      = str(PUNKTWOLKE),
    output_hdf5     = str(HDF5),
    output_fig_dir  = str(FIGURES_DIR),
    # Normalen
    knn_normals     = 20,
    helix_unwind_normals = True,  # KNN+PCA im "entwundenen" Frame (theta-k*z) statt
                                  # roh — vermeidet Scher-Bias in n_tang bei großem β.
                                  # False = altes Verhalten (Diagnose-Vergleich).
    # Flanken-Radiusfilter (grob, vor Optimierung)
    r_band_frac_lo  = 0.15,   # untere 15% des Radius-Range abschneiden
    r_band_frac_hi  = 0.98,   # obere 2% (Kopfkante) abschneiden
    # Optimierung
    ds_opt          = 0,       # Subsample für phi_b-Optimierung (0 = volle Punktwolke;
                                # >0 = feste Punktzahl, schneller aber ungenauer).
                                # k_star_ist/r_b_ist sind globale Referenzwerte — JEDER
                                # Punkt-Residual (d_lot) wird relativ dazu gemessen; ein
                                # Fehler hier ist kein lokales Rauschen, sondern ein
                                # systematischer Versatz im GESAMTEN Ist/Soll-Residualplot.
                                # Daher volle Punktzahl bevorzugt, trotz höherer Laufzeit.
    ds_fft          = 0,       # Subsample für Zähnezahl/k_star-Grobschätzung (0 = volle
                                # Punktwolke). Kosten skalieren hier NICHT mit Punktzahl
                                # (Histogramm + FFT über feste Bin-Anzahl) — volle Punkt-
                                # zahl ist praktisch kostenlos und erhöht das SNR gegen
                                # konkurrierende Periodizitäten (z.B. Leichtbaubohrungen).
    n_pt_slices     = 24,      # z-Scheiben für Phasenverfolgung (Phase 2)
    r_gear_lo_frac  = 0.88,    # Gear-Surface-Filter: untere r-Grenze (×r_ref)
    r_gear_hi_frac  = 1.04,    # Gear-Surface-Filter: obere r-Grenze (×r_ref)
    axis_mode        = "z",    # ZURUECKGESTELLT auf "z" (2026-07-15) -- der "pca"-Testlauf
                                # zum Pruefen der Achs-Tilt-Hypothese (s. [[project_gear_pipeline]])
                                # ergab einen Tilt von 0.1442°, aber die resultierende f_y-
                                # Verzerrung (±400µm linearer v-Gradient, Fβ~800µm flach ueber
                                # alle Zaehne) zeigte, dass die UNGEWICHTETE PCA ueber die rohe
                                # Punktwolke durch das ~5:1 R:L-Ungleichgewicht verzerrt wird
                                # (dominiert von der R-Seite) -- NICHT verlaesslich fuer diesen
                                # Halb-Scan. "z" bleibt die sicherere Wahl bis ein R/L-balancierter
                                # Achs-Schaetzer existiert.
                                # "z": Achse=[0,0,1] (Wolke bereits z-orientiert); "pca": automatisch
    n_teeth_override = 0,      # >0: N-Erkennung überspringen, N direkt vorgeben.
                               # Pflicht bei Halb-Scan (<70% Winkelabdeckung), da FFT
                               # nur gerade Harmonische sieht und falsche N liefert.
                               # 0 = auto-Erkennung (Vollscan ≥70% Abdeckung vorausgesetzt).
    n_teeth_min     = 6,       # Mindest-Zähnezahl (unter diesem Wert → Rayleigh-Scan)
    n_teeth_max     = 250,     # Maximale Zähnezahl für Rayleigh-Scan
    fft2d_n_theta   = 512,     # 2D-FFT Bild θ-Auflösung (nur Visualisierung)
    fft2d_n_z       = 128,     # 2D-FFT Bild z-Auflösung (nur Visualisierung)
    # L/R Filter
    n_rad_min       = 0.12,    # |n·r_hat| > 0.12 → sichere Normalen-Orientierung
    gm_eps          = 0.008,   # Gaußsche Karte: |n_r·n_tang| Unsicherheitsschwelle
                              # (0.012 schneidet Flankenpunkte nahe r=104mm ab → u_max≈0.63;
                              #  0.008 erweitert auf r≈105mm → u_max≈0.85)
    curv_max        = 0.08,    # Krümmungsfilter — jetzt auch in Phase 5 auf den
                               # finalen Zahn/Seite-Cluster angewendet (nicht mehr
                               # nur auf das Phase-4-Fit-Subsample), verwirft
                               # Fußrundung/Trochoide vor dem Chebyshev-Fit.
    # Zahn-Zuordnung — Stufenfilter via phi_b_tol_frac (Eingriffsfläche-Gate):
    #   0.00 = Stufe 0: Winner-Takes-All (kein absoluter Threshold, nur r>=r_b)
    #          → KEIN Qualitäts-Cutoff: Kopfkante/Fase/Rand landen im Cluster.
    #   0.10 = Stufe 1: schmales Evolventen-Band (~0.43°)  ← Default
    #          Fertigungsabweichung bei diesem Radmaßstab typ. <0.1° — Stufe 1
    #          lässt >4× Spielraum für echtes Messsignal, verwirft aber
    #          Chamfer/Fußrundungs-Punkte (kategorial größere Abweichung).
    #   0.25 = Stufe 2: Standard-Flankenbreite  (~1.08°)
    #   0.35 = Stufe 3: breites Band mit Randpunkten (~1.52°)
    phi_b_tol_frac  = 0.10,    # Stufe 1: Eingriffsfläche-Gate statt Winner-Takes-All
    rg_min_pts      = 30,       # Phase 7: Mindest-Punkte für validen Cluster
    # ── Region Growing (Phase 5 Alternativ) ────────────────────────────────────
    # use_region_growing = True  → phase5_region_growing() statt phi_b-Zuordnung.
    # Direkter Normal-basierter Ansatz — kein phi_b, keine Kreisstreuung.
    # Erzwingt exakt 2 × n_teeth = 166 Zahnflanken-Cluster.
    # ACHTUNG Dispatcher-Prioritaet (s. run_pipeline()): use_dbscan_flank
    # > use_region_growing_lr_split > use_region_growing > (Schwellwert-
    # Fallback). Nur EINE dieser Methoden laeuft pro Ausfuehrung. Aktuell
    # ist use_dbscan_flank=True Standard, daher bewusst False hier -- nicht
    # aus Versehen zwei "aktive" Methoden gleichzeitig in der Config stehen
    # lassen (frueher standen hier versehentlich beide auf True, was den
    # Eindruck erweckte, Region Growing liefe mit, obwohl DBSCAN es immer
    # verdeckt hat).
    use_region_growing  = False,
    rg_voxel_size       = 0.5,    # [mm] Voxel-Größe für Downsampling (~2% Seeds)
    rg_angle_deg        = 5.0,    # [°] Max. Winkel zwischen Normalen (Region Growing)
    rg_dist_mm          = 6.0,    # [mm] Max. Abstand für Nachbarn (Region Growing)
    rg_seed_min         = 30,     # Min. Seeds pro Cluster (kleinere verworfen)
    rg_seed_knn         = 30,     # k für PCA-Normalen auf dem Seed-Grid (wie Notebook)
    # use_region_growing_lr_split = True → phase5_region_growing_lr_split() statt
    # phase5_region_growing(). L/R wird via gm_product (Phase 3, rein lokal,
    # k-unabhängig) VOR dem BFS entschieden statt danach — ein R- und ein
    # L-Punkt können dadurch nie im selben BFS-Lauf verbunden werden, auch
    # nicht an der Zahnkopf-Kante, wo gm_product und die BFS-Winkel-Schwelle
    # gleichzeitig am schwächsten sind. Nutzt dieselben rg_*-Parameter.
    use_region_growing_lr_split = False,
    edge_trim_pct   = 0.0,     # optionaler r-Perzentil-Trim je Cluster (0=aus, s. Phase 7)
    # ── DBSCAN Flanken-Segmentierung (Phase 5 — empfohlener Ansatz) ────────────
    # use_dbscan_flank = True  → phase5_dbscan_flank() statt phi_b-Threshold oder RG.
    # Methode: tc_rel-Vorfilter (R/L trennen) → DBSCAN in (φ_b/Teilung, z/z_max).
    # Tip/Fuß/Fase-Punkte → Rauschen (Label -1) ohne manuellen Toleranzwert.
    # Voraussetzung: k_star + r_b aus Phase 4 (blind erkannt, kein MY_GEAR).
    use_dbscan_flank    = True,   # DBSCAN als Standard-Phase-5
    dbscan_eps          = 0.05,   # ε in norm. (φ_b/Teilung, z/z_max) Raum
    dbscan_min_pts      = 30,     # DBSCAN min_samples
    dbscan_tc_pre_frac  = 0.45,   # |tc_rel| < frac×Teilung → Kandidat (r≥r_b)
    # Phase 5c: räumliche Kohärenz-Bereinigung
    # min_gzk_frac: Mindest-Anteil der größten Zusammenhangskomponente (GZK).
    # Wenn GZK < min_gzk_frac × Cluster-Größe → kNN hat Cross-Tooth-Kanten
    # → Bereinigung wäre destruktiv → Cluster wird NICHT bereinigt.
    # 0.50 = Standard: nur bereinigen wenn GZK ≥ 50% des Clusters.
    # 0.00 = Phase 5c komplett deaktivieren (immer überspringen).
    min_gzk_frac    = 0.50,
    # Soll-Verifikation
    soll_dist_flag      = 0.30,   # [mm] Punkte weiter als das → verdächtig
    soll_dist_drop      = 0.80,   # [mm] Punkte weiter als das → verworfen
    soll_align_max_dist = 5.0,    # [mm] max. Alignment-Fehler; darüber → Phase 6 skip
    # Phase 4 z-Band-Filter: None = alle Punkte verwenden;
    # Tupel z.B. (10.0, 30.0) = nur Punkte mit z ∈ [10, 30] mm einbeziehen.
    # Nützlich um Rand-Artefakte (Scanner-Inzidenzwinkel-Bias) auszublenden.
    phase4_z_band       = None,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1 — Laden + PCA-Achse
# ═══════════════════════════════════════════════════════════════════════════════

def phase1_load_and_axis(cfg):
    t0 = time.time()
    log("Phase 1: Laden + PCA-Achse")

    pts = pd.read_csv(cfg["input_file"], header=None, dtype=np.float64).to_numpy()
    log(f"  {len(pts):,} Punkte geladen", t0)

    center = pts[:, :3].mean(axis=0)

    # Achsenbestimmung: entweder aus CFG (wenn Punktwolke bereits z-orientiert)
    # oder via PCA. WICHTIG: für flache Zahnräder (r >> Breite) liefert PCA den
    # größten Eigenwert entlang der radialen Richtung, NICHT entlang der Drehachse!
    # Daher: 'axis_mode = "z"' setzt Achse = [0,0,1] (Punktwolke bereits ausgerichtet).
    axis_mode = str(cfg.get("axis_mode", "pca")).lower()

    if axis_mode == "z":
        axis = np.array([0., 0., 1.])
        log("  Achse: [0,0,1] (CFG axis_mode='z' — Punktwolke bereits z-orientiert)")
    else:
        cov    = np.cov((pts[:, :3] - center).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        # Kleinster Eigenwert = minimale Varianz = Rotationsachse eines Zylinders/Scheibe
        # Größter Eigenwert wäre für lange Wellen korrekt, aber für Zahnrad-Scheiben falsch.
        # Heuristik: nehme den Eigenvektor, der am stärksten mit der z-Spalte des
        # Eingabe-Koordinatensystems übereinstimmt (robuster als Min/Max-Eigenwert).
        z_unit  = np.array([0., 0., 1.])
        dots    = np.abs(eigvecs.T @ z_unit)   # Ähnlichkeit jedes EV mit z
        axis    = eigvecs[:, np.argmax(dots)]   # EV der am meisten z-ähnlich ist
        if axis[2] < 0:
            axis = -axis
        tilt_deg = float(np.degrees(np.arccos(np.clip(axis[2], -1.0, 1.0))))
        log(f"  Achse (PCA, z-ähnlichster EV): {axis.round(4)}  "
            f"Tilt ggue. [0,0,1]: {tilt_deg:.4f}°")

    tmp = np.array([1., 0., 0.]) if abs(axis[0]) < 0.9 else np.array([0., 1., 0.])
    e1  = np.cross(axis, tmp); e1 /= np.linalg.norm(e1)
    e2  = np.cross(axis, e1)

    # Zylinderkoordinaten
    p_perp   = pts[:, :3] - center
    z_raw    = p_perp @ axis
    p_rad    = p_perp - np.outer(z_raw, axis)
    r        = np.linalg.norm(p_rad, axis=1)
    theta    = np.arctan2(p_rad @ e2, p_rad @ e1)
    z_off    = z_raw.min()
    z        = z_raw - z_off

    pts_cyl = np.column_stack([r, theta, z]).astype(np.float32)
    log(f"  r∈[{r.min():.1f}, {r.max():.1f}]mm  "
        f"z∈[0, {z.max():.1f}]mm  "
        f"θ∈[{np.degrees(theta.min()):.0f}°, {np.degrees(theta.max()):.0f}°]", t0)

    return dict(
        pts_xyz=pts.astype(np.float32),
        pts_cyl=pts_cyl,
        center=center, axis=axis, e1=e1, e2=e2, z_off=z_off,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2 — Zähnezahl + grober k_star (FFT, kein MY_GEAR)
# ═══════════════════════════════════════════════════════════════════════════════

def _detect_n_teeth(theta_corr, n_bins=1440, max_n=200):
    hist, _ = np.histogram(theta_corr, bins=n_bins, range=(0, 2*np.pi))
    fft = np.abs(np.fft.rfft(hist - hist.mean()))
    fft[:4] = 0
    peaks, props = find_peaks(fft[:max_n], height=0.15 * fft[:max_n].max(),
                               distance=3)
    if not len(peaks):
        return int(np.argmax(fft[:max_n]))
    order   = np.argsort(-props['peak_heights'])
    best    = int(peaks[order[0]])
    # Oberwellen-Check: ist die halbe Frequenz ein stärkerer Peak?
    for p in peaks[order]:
        half = p / 2
        if half >= 4 and any(abs(pk - round(half)) <= 2 for pk in peaks):
            cand = round(half)
            if cand < best:
                best = cand
    return best


def _rayleigh_score(k, theta, z, tooth_period):
    """Kreisschärfe von theta_corr mod tooth_period (höher = besser)."""
    tc  = (theta - k * z) % tooth_period
    ang = tc / tooth_period * 2 * np.pi
    return float(np.abs(np.mean(np.exp(1j * ang))))


def _kstar_phase_tracking(pts_cyl, n_teeth, n_slices=24, n_bins_theta=1440):
    """
    Schätzt k_star via linearer Phasenverfolgung über z-Scheiben.

    Für jede z-Scheibe: Phase des N-ten FFT-Harmonischen in theta.
    Die Phase verschiebt sich linear mit z: ψ(z) = N · k_star · z + ψ_0.
    Gewichteter linearer Fit (Gewicht ∝ FFT-Amplitude der Scheibe) → k_star.

    Vorteile vs. Rayleigh-Gitter:
      - Keine Startschätzung nötig, kein Gitter-Aliasing
      - Jede Scheibe unabhängig → linear-phase-Modell direkt testbar
      - Gibt Unsicherheitsmaß (Phasen-Residuen-Std)
      - Skaliert automatisch mit Punktzahl (mehr Punkte → besseres SNR pro Scheibe)
    """
    theta = pts_cyl[:, 1].astype(np.float64)
    z     = pts_cyl[:, 2].astype(np.float64)
    z_min, z_max = float(z.min()), float(z.max())
    z_span = z_max - z_min

    if z_span < 1e-6:
        return None, None

    dz        = z_span / n_slices
    z_centers = np.linspace(z_min + dz/2, z_max - dz/2, n_slices)

    z_vals, phases, weights = [], [], []
    for z_c in z_centers:
        mask_s = np.abs(z - z_c) < dz / 2
        if int(mask_s.sum()) < n_teeth * 5:
            continue
        hist, _ = np.histogram(theta[mask_s], bins=n_bins_theta,
                                range=(0, 2*np.pi))
        fft_h   = np.fft.rfft(hist.astype(float) - hist.mean())
        z_vals.append(z_c)
        phases.append(float(np.angle(fft_h[n_teeth])))
        weights.append(float(np.abs(fft_h[n_teeth])))

    if len(z_vals) < 4:
        return None, None, None, None, None   # nicht genug Scheiben mit ausreichend Punkten

    z_arr  = np.array(z_vals)
    ph_arr = np.unwrap(np.array(phases))
    w_arr  = np.array(weights)
    w_arr  = w_arr / w_arr.sum()

    # Gewichtetes LSQ:  phase = (N·k_star)·z + offset
    A    = np.column_stack([z_arr, np.ones(len(z_arr))])
    AW   = A * w_arr[:, None]
    coef = np.linalg.solve(AW.T @ A, AW.T @ ph_arr)

    k_star_pt = float(coef[0]) / n_teeth        # Steigung / N = k_star
    # Qualitätsmaß: Phasen-Residuen-Std → kleiner = besser
    res_std   = float(np.sqrt(np.average((ph_arr - A @ coef)**2, weights=w_arr)))
    k_std_est = res_std / (n_teeth * z_span) if z_span > 0 else np.inf

    return k_star_pt, k_std_est, z_arr, ph_arr, A @ coef


def _kstar_ring_xcorr(pts_cyl, n_teeth, r_ref, n_bins=4096):
    """Ring-zu-Ring-Kreuzspektrum (N-tes Harmonisches) — robustere β-Schätzung.

    Verwendet die Phase des N-ten Harmonischen im Kreuzspektrum benachbarter
    z-Ringe: Δφ = N × k × Δz  →  k = Δφ / (N × Δz).

    Vorteile gegenüber Phasenverfolgung (24 dicke Scheiben):
      • ~4000 Ring-Paare statt 24 → statistisch ~13× kleinerer σ_k
      • Nutzt das vollständige Histogramm, nicht nur einen Mittelwert pro Scheibe
      • Direkte Phasenmessung → kein Regressionsrauschen

    Bekannte Schwäche (weiterhin ungelöst, siehe [[project_gear_pipeline]]):
    liefert wiederholt einen falschen β (26-27° statt der von zwei
    unabhängigen Methoden übereinstimmend gefundenen ~21.8°). Ein Amplituden-
    Gate (nur die stärkere Hälfte der Ring-Paare nach n_teeth-Harmonischen-
    Stärke) wurde am 2026-07-13 versucht und auf echten Daten getestet — das
    Ergebnis wurde SCHLECHTER (26.46°→27.31°), also wieder zurückgesetzt
    (siehe Kommentar unten bei "close"). Ein sauberer Fix bräuchte denselben
    R/L-Split-und-Mittelung-Trick, der off_tc in phase5_dbscan_flank() und
    phase5b_scan_profile_crosscheck() erfolgreich korrigiert hat — das
    scheitert hier aber strukturell: diese Funktion läuft in Phase 2, VOR
    Phase 4, wo r_b erst bestimmt wird; ohne r_b kein phi_b, ohne phi_b keine
    R/L-Trennung. k_xcorr fließt ohnehin nirgends in k_init ein (nur
    Referenz-Log-Zeile) — Umbau lohnt sich nur, falls dieser Wert selbst
    gebraucht wird.

    Eingabe: pts_cyl  – (N, 3) Array [r, θ, z], bereits gear-gefiltert
             n_teeth  – korrekte Zähnezahl (N-tes Harmonisches)
             r_ref    – Referenzradius [mm] für β-Anzeige
    Ausgabe: (k_star [rad/mm], k_std [rad/mm])  oder (None, None) bei Fehler
    """
    theta = pts_cyl[:, 1] % (2.0 * np.pi)
    z     = pts_cyl[:, 2]

    # z-Ebenen via 0.01mm-Ganzzahl-Raster (vermeidet Fließkomma-Matching)
    z_um  = np.round(z * 100.0).astype(np.int32)
    levels, counts = np.unique(z_um, return_counts=True)
    valid_mask = counts >= max(20, n_teeth)
    z_valid = levels[valid_mask]

    if len(z_valid) < 10:
        log(f"  Ring-XKorr: zu wenige gültige z-Ebenen ({len(z_valid)}) — übersprungen")
        return None, None

    # θ-Histogramme vektorisiert aufbauen (kein Python-Loop über z-Ebenen)
    bins   = np.linspace(0.0, 2.0 * np.pi, n_bins + 1)
    t_bin  = np.clip(
        np.digitize(theta, bins) - 1, 0, n_bins - 1
    ).astype(np.int32)
    # z-Ebenen-Index pro Punkt via binärer Suche
    z_lev_idx = np.searchsorted(z_valid, z_um)
    z_lev_idx = np.clip(z_lev_idx, 0, len(z_valid) - 1)
    exact     = z_valid[z_lev_idx] == z_um          # nur exakt zugeordnete Punkte

    idx2d = (z_lev_idx * n_bins + t_bin).astype(np.int64)
    idx2d[~exact] = -1                               # markiere Nicht-Ebenen-Punkte
    valid_pts = idx2d >= 0

    rings_flat = np.bincount(idx2d[valid_pts],
                              minlength=len(z_valid) * n_bins).astype(np.float32)
    rings = rings_flat.reshape(len(z_valid), n_bins)

    # FFT aller Ringe (axis=1) → Kreuzspektrum N-tes Harmonisches
    F     = np.fft.rfft(rings, axis=1)                        # (n_lev, n_bins//2+1)
    cross = F[1:, n_teeth] * np.conj(F[:-1, n_teeth])         # (n_lev-1,)

    dz_mm   = np.diff(z_valid.astype(np.float64)) / 100.0    # Δz [mm] pro Paar
    # Vorzeichenkorrektur FFT-Konvention: rfft verwendet exp(-i·2π·k·n/N),
    # daher angle(F[i+1]·conj(F[i])) = -(N·k·Δz) für positives k.
    # Negieren liefert das physikalisch richtige Vorzeichen.
    dtheta  = -np.angle(cross) / float(n_teeth)               # Δθ [rad] pro Paar
    k_pairs = dtheta / np.maximum(dz_mm, 1e-6)               # k [rad/mm] pro Paar

    # Nur reguläre z-Schritte (Lücken im Scan verzerren k)
    dz_med   = float(np.median(dz_mm))
    close    = np.abs(dz_mm - dz_med) < dz_med * 0.5
    if close.sum() < 5:
        close = np.ones(len(k_pairs), dtype=bool)

    # HINWEIS (2026-07-13): ein Amplituden-Gate (nur die stärkere Hälfte der
    # Ring-Paare nach n_teeth-Harmonischen-Stärke) wurde hier versuchsweise
    # eingebaut und auf echten Daten getestet — das Ergebnis wurde SCHLECHTER
    # (β driftete von 26.46° auf 27.31°, weiter weg von den ~21.8°, auf die
    # sich Phasenverfolgung und Joint-Opt. unabhängig einigen). Das spricht
    # dagegen, dass "amplitudenstark = zuverlässig" hier gilt — plausibler:
    # ein starkes n_teeth-Harmonisches entsteht gerade dann, wenn ein Ring
    # von EINER Flankenseite dominiert wird (scharfer, einseitiger Peak),
    # während ein schwächeres/breiteres Harmonisches eher eine ausgewogenere
    # R+L-Mischung anzeigt (näher an der wahren Mitte). Amplituden-Filterung
    # bevorzugte damit unabsichtlich die R-dominierten (stärker verzerrten)
    # Ringe. Zurückgesetzt auf den einfachen, ungewichteten Median. Ein
    # sauberer Fix bräuchte eine R/L-Aufspaltung wie bei off_tc — die aber
    # hier (Phase 2, vor Phase 4) an r_b scheitert: phi_b braucht r_b, das
    # erst in Phase 4 aus dieser Methode mit-bestimmt wird. Da k_xcorr
    # ohnehin nirgends in k_init einfließt (siehe Log "Referenz, nicht in
    # k_init"), lohnt sich der Umbau nur, wenn dieser Wert selbst gebraucht
    # wird — aktuell nicht der Fall.
    k_good = k_pairs[close]
    k_star = float(np.median(k_good))
    q25, q75 = np.percentile(k_good, [25, 75])
    k_std  = float((q75 - q25) / 1.35 / np.sqrt(len(k_good)))

    beta = float(np.degrees(np.arctan(k_star * r_ref)))
    log(f"  Ring-XKorr ({n_teeth}. Harm., {len(k_good):,} Paare): "
        f"k={k_star:.7f}  β={beta:.4f}°  σ_k={k_std:.2e}")
    return k_star, k_std


def _dlot_analytical(phi_b_arr, r_b, beta_b, side, tooth_period):
    """
    Analytische Lot-Abstandsabweichung (d_lot) aus φ_b-Felddaten.

    Herleitung (exakt für kleine Abweichungen, da r·cos(α_t) = r_b):
        d_lot = +r_b · cos(β_b) · Δφ_b   (R-Flanke)
        d_lot = -r_b · cos(β_b) · Δφ_b   (L-Flanke)
    Vorzeichen: d_lot > 0 = Messpunkt liegt außerhalb der Idealflanke
                d_lot < 0 = Messpunkt liegt unterhalb (mehr Material).

    Parameter
    ---------
    phi_b_arr    : (N,) φ_b-Werte [rad] für diese Flanke × Zahn-Gruppe
    r_b          : Grundkreisradius [mm]  (aus Phase 4)
    beta_b       : Grundkreis-Schrägungswinkel [rad] = arctan(k · r_b)
    side         : 'R' oder 'L'
    tooth_period : 2π / N_teeth [rad]

    Rückgabe
    --------
    d_lot        : (N,) Lot-Abstandsabweichung [mm]
    phi_b_mean   : Kreismittelwert von φ_b [rad]
    """
    ang        = phi_b_arr / tooth_period * (2.0 * np.pi)
    phi_b_mean = float(
        np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())
        / (2.0 * np.pi) * tooth_period
    )
    delta  = (phi_b_arr - phi_b_mean + tooth_period / 2.0) % tooth_period - tooth_period / 2.0
    sign   = +1.0 if side == 'R' else -1.0
    d_lot  = sign * r_b * np.cos(beta_b) * delta
    return d_lot, phi_b_mean


def _build_2d_fft_image(pts_cyl, n_theta=512, n_z=128):
    """
    2D-Histogramm in (θ, z)-Raum: mittlerer Radius pro Zelle (DC-subtrahiert).
    Nur für Visualisierung. Schräge Streifen = Schrägungswinkel β.
    """
    r     = pts_cyl[:, 0].astype(np.float64)
    theta = pts_cyl[:, 1].astype(np.float64)
    z     = pts_cyl[:, 2].astype(np.float64)
    z_max = float(z.max())

    i_th = np.clip(np.digitize(theta, np.linspace(0, 2*np.pi, n_theta+1)) - 1,
                   0, n_theta - 1)
    i_z  = np.clip(np.digitize(z, np.linspace(0, z_max, n_z+1)) - 1,
                   0, n_z - 1)

    r_sum   = np.zeros((n_theta, n_z))
    r_count = np.zeros((n_theta, n_z), dtype=np.int32)
    np.add.at(r_sum,   (i_th, i_z), r)
    np.add.at(r_count, (i_th, i_z), 1)

    image = np.where(r_count > 0, r_sum / np.maximum(r_count, 1), r.mean())
    image -= image.mean(axis=0, keepdims=True)   # DC pro z-Spalte entfernen
    return image, z_max



def _find_gear_r_ref(r_ds, th_ds, n_tooth_min=10, n_tooth_max=250,
                      n_bands=80, dr_mm=3.0):
    """Finde Zahnrad-r_ref durch Scannen von r-Bändern.
    Gibt den r-Wert zurück, bei dem das FFT-Spektrum die stärkste N-fache
    Periodizität (N=n_tooth_min..n_tooth_max) zeigt → Zahnflanken-Region.
    Ignoriert Flansche/Gehäuse, die am gleichen r mehr Punkte haben können
    aber keine Zahn-Periodizität zeigen.
    """
    r_lo = float(np.percentile(r_ds, 2))
    r_hi = float(np.percentile(r_ds, 98))
    r_centers = np.linspace(r_lo + dr_mm, r_hi - dr_mm, n_bands)

    best_rel = 0.0
    best_r   = float(np.median(r_ds))   # Fallback

    for r_c in r_centers:
        mask = np.abs(r_ds - r_c) < dr_mm / 2
        cnt  = int(mask.sum())
        if cnt < 200:
            continue
        th_band = th_ds[mask] % (2*np.pi)
        hist, _ = np.histogram(th_band, bins=720, range=(0, 2*np.pi))
        amp = np.abs(np.fft.rfft(hist.astype(float) - hist.mean()))
        n_hi_idx  = min(n_tooth_max, len(amp) - 1)
        # Normiere durch Punktzahl → Rayleigh-ähnlicher Score
        peak_rel = float(amp[n_tooth_min : n_hi_idx + 1].max()) / cnt
        if peak_rel > best_rel:
            best_rel = peak_rel
            best_r   = r_c

    return best_r, best_rel

def phase2_rough_helix_and_teeth(s1, cfg):
    """
    Zähnezahl N via 1D-FFT, k_star via z-Scheiben-Phasenverfolgung.

    Phasenverfolgung (phase tracking) statt Rayleigh-Gitter:
      - Für jede z-Scheibe: Phase des N-ten Harmonischen in theta
      - Phase verschiebt sich linear: ψ(z) = N·k_star·z + ψ₀
      - Gewichtetes LSQ → k_star ohne Gitter-Aliasing
      - Keine Startschätzung, kein Gitter, kein Rayleigh-Score-Sweep
    """
    t0 = time.time()
    log("Phase 2: Zähnezahl + k_star (2D Phasenverfolgung)")

    pts_cyl = s1["pts_cyl"].astype(np.float64)
    n       = len(pts_cyl)
    # ds_fft=0 (Default) → volle Punktwolke. Histogramm+FFT skalieren nicht mit
    # der Punktzahl (feste Bin-Anzahl) — Subsampling bringt hier keinen echten
    # Geschwindigkeitsgewinn, kostet aber SNR gegen konkurrierende Periodizitäten.
    ds_fft = int(cfg.get("ds_fft", 0))
    if ds_fft and ds_fft < n:
        # Seeded RNG statt globalem np.random.choice() -- ds_fft ist per
        # Default 0 (deaktiviert, s. CFG-Kommentar), aber falls doch aktiv,
        # soll das Ergebnis reproduzierbar sein statt bei jedem Lauf anders
        # (genau die Nicht-Determinismus-Klasse, die 2026-07-13 fuer den
        # Default-Fall bereits gefunden+behoben wurde, s. project_gear_pipeline.md).
        idx_ds     = np.random.default_rng(42).choice(n, ds_fft, replace=False)
        th_ds      = pts_cyl[idx_ds, 1]
        r_ds       = pts_cyl[idx_ds, 0]
        z_ds       = pts_cyl[idx_ds, 2]
        pts_cyl_ds = pts_cyl[idx_ds]
    else:
        th_ds      = pts_cyl[:, 1]
        r_ds       = pts_cyl[:, 0]
        z_ds       = pts_cyl[:, 2]
        pts_cyl_ds = pts_cyl

    # r_ref: r-Band-Scan → findet Zahnflanken-Region via max. N-facher FFT-Amplitude.
    # Robuster als Mode/Median wenn Flansch/Gehäuse bei anderem r dominiert.
    r_ref, _r_ref_score = _find_gear_r_ref(r_ds, th_ds)
    log(f"  r_ref (Zahnflanken-Scan): {r_ref:.2f}mm  Score={_r_ref_score:.4f}  "
        f"r∈[{r_ds.min():.1f}, {r_ds.max():.1f}]mm")

    # ── Theta-Diagnostik (gesamte Stichprobe, kein r-Filter) ──────────────────
    _th_all = th_ds % (2*np.pi)
    _hist72, _ = np.histogram(_th_all, bins=72, range=(0, 2*np.pi))  # 5°-Bins
    _cov_all = float((_hist72 > 0).sum()) / 72.0
    log(f"  Theta (alle Pkte, 5°-Bins): Abdeckung={_cov_all*100:.0f}%  "
        f"θ∈[{np.degrees(_th_all.min()):.1f}°, {np.degrees(_th_all.max()):.1f}°]")

    # ── Gear-Surface-Filter ────────────────────────────────────────────────────
    # Nur Punkte auf der Zahnflanken-Oberfläche verwenden (r ≈ r_ref).
    # Wellen, Flansche, Grundkörper haben deutlich kleineren oder größeren r.
    # Ohne Filter dominieren Nieder-Frequenz-Artefakte (z.B. 4-fach-Symmetrie
    # durch ungleichmäßige Scan-Abdeckung) die 1D-FFT.
    r_lo_gear = r_ref * float(cfg.get("r_gear_lo_frac", 0.88))
    r_hi_gear = r_ref * float(cfg.get("r_gear_hi_frac", 1.04))
    mask_gear = (r_ds >= r_lo_gear) & (r_ds <= r_hi_gear)
    n_gear    = int(mask_gear.sum())

    if n_gear < 2000:
        log(f"  ⚠ Gear-Filter (r∈[{r_lo_gear:.1f},{r_hi_gear:.1f}]mm) "
            f"liefert nur {n_gear} Punkte — verwende gesamte Stichprobe")
        th_gear = th_ds % (2*np.pi)
        z_gear  = z_ds
        pts_cyl_gear = pts_cyl_ds
    else:
        th_gear = th_ds[mask_gear] % (2*np.pi)
        z_gear  = z_ds[mask_gear]
        pts_cyl_gear = pts_cyl_ds[mask_gear]
        log(f"  Gear-Filter: r∈[{r_lo_gear:.1f},{r_hi_gear:.1f}]mm → "
            f"{n_gear:,}/{len(th_ds):,} Punkte")

    # ── Schritt 1: N_teeth aus 1D-FFT — nur Zahnkopf-Band ───────────────────
    # Problem: Die Gear-Band [r_lo, r_hi] enthält auch den Grundkörper des
    # Zahnrads (z.B. Leichtbaubohrungen → 6-fache Symmetrie dominiert FFT).
    # Lösung: oberes r-Teilband (Zahnkopf-Region) — dort gibt es nur Zähne,
    # keine Bohrungen oder anderen Symmetrie-Störer.
    r_tip_frac = float(cfg.get("r_tip_frac", 0.65))   # oberste 35% des Gear-Bandes
    r_tip_lo   = r_lo_gear + r_tip_frac * (r_hi_gear - r_lo_gear)
    mask_tip   = mask_gear & (r_ds >= r_tip_lo)
    n_tip      = int(mask_tip.sum())

    if n_tip > 1000:
        th_tip = th_ds[mask_tip] % (2*np.pi)
        z_tip  = z_ds[mask_tip]
        log(f"  Zahnkopf-Band r>[{r_tip_lo:.1f}]mm: {n_tip:,} Punkte für N-Erkennung")
    else:
        log(f"  ⚠ Zahnkopf-Band zu leer ({n_tip} Pkt) — nutze volles Gear-Band")
        th_tip = th_gear
        z_tip  = z_gear

    # ── N-Override: bei Halb-Scan oder bekanntem Zahnrad N direkt vorgeben ──────
    # Halb-Scan-Diagnose: wenn FFT-Top-10 nur gerade Zahlen enthält → Halb-Scan.
    # In diesem Fall ist automatische N-Erkennung nicht möglich; N muss aus CFG.
    n_teeth_override = int(cfg.get("n_teeth_override", 0))
    n_min_plausible  = int(cfg.get("n_teeth_min", 6))

    if n_teeth_override > 0:
        n_teeth_rough = n_teeth_override
        log(f"  N_teeth (CFG-Override): {n_teeth_rough}  "
            f"[Halb-Scan — automatische Erkennung nicht möglich]")
    else:
        n_teeth_rough = _detect_n_teeth(th_tip)
        log(f"  N_teeth (1D-FFT, Zahnkopf-Band): {n_teeth_rough}")

        # Halb-Scan-Prüfung: <70% Winkelabdeckung → FFT sieht nur gerade Harmonische
        # → N-Erkennung liefert Vielfache von N, nicht N selbst → k_star falsch
        # → alle nachgelagerten Phasen kaskadieren den Fehler.
        # Harte Abbruchbedingung statt stiller Warnung.
        _hist_d, _ = np.histogram(th_tip, bins=360, range=(0, 2*np.pi))
        _cov = float((_hist_d > 0).sum()) / 360.0
        if _cov < 0.7:
            raise RuntimeError(
                f"Halb-Scan erkannt: Winkelabdeckung nur {_cov*100:.0f}% (<70%). "
                f"Automatische N-Erkennung ist für Halb-Scans nicht zuverlässig — "
                f"ein falsches N macht k_star und alle nachgelagerten Ergebnisse "
                f"unbrauchbar (z.B. beta=falscher Wert, L/R-Asymmetrie). "
                f"Bitte 'n_teeth_override' in CFG auf die bekannte Zähnezahl setzen."
            )

        if n_teeth_rough < n_min_plausible:
            log(f"  ⚠ N={n_teeth_rough} unplausibel — Rayleigh-Scan über "
                f"N={n_min_plausible}..{cfg.get('n_teeth_max', 250)}")
            n_max = int(cfg.get("n_teeth_max", 250))
            n_candidates = np.arange(n_min_plausible, n_max + 1)
            r_scores = np.array([
                _rayleigh_score(0.0, th_tip, np.zeros_like(th_tip), 2*np.pi/n)
                for n in n_candidates
            ])
            n_teeth_rough = int(n_candidates[np.argmax(r_scores)])
            log(f"  → Rayleigh-Scan: N={n_teeth_rough}  (Score={r_scores.max():.4f})")

    # ── Schritt 2: k_star via z-Scheiben-Phasenverfolgung ─────────────────────
    pt_result = _kstar_phase_tracking(
        pts_cyl_gear, n_teeth_rough, n_slices=cfg.get("n_pt_slices", 24)
    )
    if pt_result[0] is not None:
        k_pt, k_std, z_slices, ph_measured, ph_fit = pt_result

        # ── Vorzeichenkorrektur: vergleiche Rayleigh-Score für +k und -k ────────
        # Die z-Richtung der Punktwolke kann zur Soll-Konvention invertiert sein,
        # was das Vorzeichen von k_star umkehrt. Rayleigh-Score wählt das richtige.
        tp_r = 2 * np.pi / n_teeth_rough
        score_pos = _rayleigh_score(+abs(k_pt), th_gear, z_gear, tp_r)
        score_neg = _rayleigh_score(-abs(k_pt), th_gear, z_gear, tp_r)
        if score_neg > score_pos:
            k_pt = -abs(k_pt)
            log(f"  Vorzeichenkorrektur: k_star negiert (Score +{score_pos:.4f} < -{score_neg:.4f})")
        else:
            k_pt = +abs(k_pt)
            log(f"  Vorzeichen bestätigt: k_star positiv (Score +{score_pos:.4f} > -{score_neg:.4f})")

        beta_pt = np.degrees(np.arctan(k_pt * r_ref))
        log(f"  Phasenverfolgung: k_star={k_pt:.7f}  β={beta_pt:.4f}°  "
            f"Phasen-Std={np.degrees(k_std * n_teeth_rough):.3f}°/mm")
    else:
        log("  ⚠ Phasenverfolgung fehlgeschlagen — fallback auf Rayleigh-Gitter")
        tooth_period_r = 2 * np.pi / n_teeth_rough
        k_candidates   = np.tan(np.radians(np.linspace(0, 40, 120))) / r_ref
        scores         = [_rayleigh_score(k, th_gear, z_gear, tooth_period_r)
                          for k in k_candidates]
        k_pt = float(k_candidates[np.argmax(scores)])
        k_std, z_slices, ph_measured, ph_fit = None, None, None, None

    # ── Schritt 3: N_teeth mit k_pt bestätigen (bei Override: überspringen) ───
    n_teeth_step1 = n_teeth_rough          # N aus Schritt 1 merken
    th_corr_gear = (th_gear - k_pt * z_gear) % (2*np.pi)
    if n_teeth_override > 0:
        log(f"  N_teeth nach Helix-Korrektur: {n_teeth_rough}  (Override beibehalten)")
    else:
        n_teeth_rough = _detect_n_teeth(th_corr_gear)
        if n_teeth_rough < n_min_plausible:
            n_candidates = np.arange(n_min_plausible, int(cfg.get("n_teeth_max", 250))+1)
            r_scores = np.array([_rayleigh_score(0.0, th_corr_gear,
                                                  np.zeros_like(th_corr_gear),
                                                  2*np.pi/n) for n in n_candidates])
            n_teeth_rough = int(n_candidates[np.argmax(r_scores)])
            log(f"  N_teeth (Rayleigh nach Helix-Korrektur): {n_teeth_rough}")
        else:
            log(f"  N_teeth (nach k_pt Helix-Korrektur): {n_teeth_rough}")
    tooth_period_r = 2 * np.pi / n_teeth_rough

    # ── Schritt 3a: Phasenverfolgung mit korrigiertem N (wenn N geändert) ────
    # Problem: Wenn Schritt 1 ein falsches N liefert (z.B. N=6 statt N=83 wegen
    # Leichtbaubohrungen), nutzt Phasenverfolgung in Schritt 2 das falsche N
    # → k_pt völlig falsch (z.B. β=46° statt β=22°).
    # Lösung: Nach N-Korrektur in Schritt 3 → erneuter Phasenverfolgung-Lauf
    # mit richtigem N → liefert korrektes k_pt.
    if n_teeth_rough != n_teeth_step1 and n_teeth_override == 0:
        log(f"  N geändert ({n_teeth_step1}→{n_teeth_rough}) — re-run Phasenverfolgung mit N={n_teeth_rough}")
        _pt2 = _kstar_phase_tracking(
            pts_cyl_gear, n_teeth_rough, n_slices=cfg.get("n_pt_slices", 24)
        )
        if _pt2[0] is not None:
            _k2, _ks2, _zs2, _phm2, _phf2 = _pt2
            # Vorzeichenkorrektur
            _tp2  = 2 * np.pi / n_teeth_rough
            _sp   = _rayleigh_score(+abs(_k2), th_gear, z_gear, _tp2)
            _sn   = _rayleigh_score(-abs(_k2), th_gear, z_gear, _tp2)
            _k2   = +abs(_k2) if _sp >= _sn else -abs(_k2)
            log(f"  Phasenverfolgung (N={n_teeth_rough}, re-run): k_star={_k2:.7f}  "
                f"β={np.degrees(np.arctan(_k2*r_ref)):.4f}°")
            # Altes (falsches) k_pt ersetzen
            k_pt, k_std         = _k2, _ks2
            z_slices, ph_measured, ph_fit = _zs2, _phm2, _phf2

    # ── Schritt 3b: Ring-zu-Ring-Kreuzspektrum (volle Punktwolke, kein Subsample) ──
    # WICHTIG: kNN-Subsample (pts_cyl_gear) hat ~42 Punkte/Ring → Histogramm zu dünn.
    # Daher: gear-Filter auf ALLE Punkte anwenden und an xcorr übergeben.
    _r_full = pts_cyl[:, 0]
    _mask_gear_full = (_r_full >= r_lo_gear) & (_r_full <= r_hi_gear)
    pts_cyl_xcorr   = pts_cyl[_mask_gear_full]
    log(f"  Ring-XKorr Datenbasis: {_mask_gear_full.sum():,} Punkte (voller Gear-Filter)")

    xcorr_result = _kstar_ring_xcorr(pts_cyl_xcorr, n_teeth_rough, r_ref)
    if xcorr_result[0] is not None:
        k_xcorr, k_xcorr_std = xcorr_result
        # Vorzeichenprüfung identisch zu Phasenverfolgung
        tp_xc   = 2 * np.pi / n_teeth_rough
        sc_xp   = _rayleigh_score(+abs(k_xcorr), th_gear, z_gear, tp_xc)
        sc_xn   = _rayleigh_score(-abs(k_xcorr), th_gear, z_gear, tp_xc)
        if sc_xn > sc_xp:
            k_xcorr = -abs(k_xcorr)
            log(f"  Ring-XKorr: Vorzeichen korrigiert (neg)")
        else:
            k_xcorr = +abs(k_xcorr)
    else:
        k_xcorr, k_xcorr_std = None, None

    # ── k_init: Phasenverfolgung-Ergebnis (xcorr nur zur Referenz) ───────────
    # WICHTIG: xcorr kann bei manchen Scan-Geometrien einen systematischen
    # Magnitudenfehler aufweisen (z.B. Leichtbaubohrungen beeinflussen den
    # N=83-Harmonischen indirekt). Daher: k_init = k_pt (aus Phasenverfolgung),
    # xcorr nur für Logging und Phase-8-Visualisierung.
    k_init = k_pt
    if k_xcorr is not None:
        beta_xc = float(np.degrees(np.arctan(k_xcorr * r_ref)))
        log(f"  Ring-XKorr (Referenz, nicht in k_init): k={k_xcorr:.7f}  β={beta_xc:.4f}°")
    log(f"  k_init (Phasenverfolgung): {k_init:.7f}  "
        f"β={np.degrees(np.arctan(k_init*r_ref)):.4f}°")

    # ── Schritt 4: Fein-Optimierung via Rayleigh ──────────────────────────────
    # Bounds ±8% um k_init: deckt typische Scan-Unsicherheiten ab ohne
    # in lokale Nebenminima zu fallen. Funktioniert auch für negatives k_init.
    k_lo = min(k_init * 0.92, k_init * 1.08)
    k_hi = max(k_init * 0.92, k_init * 1.08)
    if abs(k_hi - k_lo) < 1e-8:
        k_lo = k_init - 5e-5
        k_hi = k_init + 5e-5
    res = minimize_scalar(
        lambda k: -_rayleigh_score(k, th_gear, z_gear, tooth_period_r),
        bounds=(k_lo, k_hi),
        method='bounded', options={'xatol': 1e-9}
    )
    k_fft = float(res.x)

    # ── Schritt 5: N_teeth mit k_fft bestätigen + Zahnzentren ────────────────
    th_corr_all  = (pts_cyl[:, 1] - k_fft * pts_cyl[:, 2]) % (2*np.pi)
    if n_teeth_override > 0:
        n_teeth = n_teeth_override          # Override beibehalten — keine erneute Erkennung
    else:
        n_teeth = _detect_n_teeth(th_corr_all)
    tooth_period = 2 * np.pi / n_teeth

    n_bins     = n_teeth * 40
    hist, _    = np.histogram(th_corr_all, bins=n_bins, range=(0, 2*np.pi))
    fft_h      = np.fft.rfft(hist)
    phase      = np.angle(fft_h[n_teeth])
    offset     = (-phase / n_teeth) % tooth_period
    tooth_angles = np.sort((np.arange(n_teeth) * tooth_period + offset)
                            % (2*np.pi))

    beta_deg = np.degrees(np.arctan(k_fft * r_ref))
    log(f"  Ergebnis: N={n_teeth}  k_star={k_fft:.7f}  "
        f"β={beta_deg:.4f}°  r_ref={r_ref:.2f}mm", t0)

    return dict(
        n_teeth=n_teeth, tooth_period=tooth_period,
        k_fft=k_fft, r_ref=r_ref,
        tooth_angles=tooth_angles,
        # Phasenverfolgung-Details für Visualisierung
        pt_z_slices=z_slices,
        pt_ph_measured=ph_measured,
        pt_ph_fit=ph_fit,
        pt_k_std=k_std,
        # Ring-XKorr-Details (None wenn nicht verfügbar)
        k_xcorr=k_xcorr,
        k_xcorr_std=k_xcorr_std,
        k_init=k_init,
    )


def _rotate_about_axis(vecs, axis, angles):
    """
    Rotiert jede Zeile von vecs um ihren eigenen Winkel (rad) um `axis`
    (Rodrigues-Formel, axis muss Einheitsvektor sein).
    """
    angles = np.asarray(angles)
    c = np.cos(angles)[:, None]
    s = np.sin(angles)[:, None]
    dot = (vecs @ axis)[:, None]
    return (vecs * c
            + np.cross(axis[None, :], vecs) * s
            + axis[None, :] * dot * (1.0 - c))


def compute_tooth_index(theta, z, k_star, n_teeth, tooth_period, off_tc=None):
    """
    Gemeinsamer Kern-Baustein: Zahn-Index + Zahnzentren-Winkel + tc_rel aus
    (theta, z) -- dieselbe Berechnung, die vorher an mindestens 6 Stellen
    (Phase 4 Runde 1+2, Phase 5 DBSCAN Runde 1+2, Phase 5 Region Growing
    (beide Varianten)) einzeln copy-paste dupliziert war. Genau diese Art
    Duplikation hat bereits einen echten Bug verursacht (Phase 5b off-by-
    one, s. project_gear_pipeline.md) -- ein Tippfehler beim Kopieren einer
    dieser Stellen wäre sonst schwer zu finden gewesen.

    ABSICHTLICH NICHT verwendet für:
      - phase5_tooth_assignment() (Schwellwert-Methode, unter aktueller CFG
        unerreichbar): baut dort ein SORTIERTES tooth_angles (np.sort(...)),
        eine andere Semantik als hier -- nicht kompatibel, nicht angefasst,
        da toter Code.
      - phase8_visualize(): liest dort bewusst s5["tooth_angles"]/
        s5["tooth_all"] (die bereits von Phase 5b autoritativ gesetzten
        Werte) statt off_tc neu zu suchen -- ein frisch gesuchtes off_tc
        könnte vom in s5 gespeicherten leicht abweichen (andere Methode/
        Rundung) und wäre dort schlicht falsch.

    Parameter
    ---------
    theta, z     : (N,) Arrays [rad]/[mm] -- Rohkoordinaten
    k_star       : float -- Helix-Koeffizient [rad/mm]
    n_teeth      : int
    tooth_period : float -- 2π/n_teeth [rad]
    off_tc       : float oder None -- wenn None, wird der Zahnzentren-Offset
                   selbst per FFT-Phase des n_teeth-ten Harmonischen in
                   tc = (theta - k_star·z) mod 2π bestimmt (dichte-basiert,
                   wie an jeder ursprünglichen Fundstelle). Wenn gegeben
                   (z.B. ein bereits symmetrisierter Wert wie off_tc_sym),
                   wird der FFT-Schritt übersprungen und dieser Wert direkt
                   verwendet.

    Rückgabe
    --------
    tooth_all    : (N,) int-Array, Zahn-Index [0, n_teeth)
    tooth_angles : (n_teeth,) Array -- Zahnzentren-Winkel je Index (UN-
                   sortiert, natürliche Index-Reihenfolge -- tooth_angles[k]
                   ist der Winkel von Zahn k, wichtig für tooth_angles[tooth_all])
    tc_rel       : (N,) Array -- vorzeichenbehafteter Abstand zum eigenen
                   Zahnzentrum, auf [-π, +π] reduziert
    off_tc       : float -- der verwendete (ggf. neu bestimmte) Offset
    tc_mod       : (N,) Array -- (theta - k_star·z) mod 2π
    """
    tc_mod = (theta - k_star * z) % (2.0 * np.pi)
    if off_tc is None:
        n_bins  = n_teeth * 40
        hist, _ = np.histogram(tc_mod, bins=n_bins, range=(0.0, 2.0 * np.pi))
        fft_h   = np.fft.rfft(hist)
        off_tc  = float((-np.angle(fft_h[n_teeth]) / n_teeth) % tooth_period)

    tooth_all = (np.round(
        ((tc_mod - off_tc + 4.0 * np.pi) % (2.0 * np.pi)) / tooth_period
    ).astype(int) % n_teeth)
    tooth_angles = (np.arange(n_teeth) * tooth_period + off_tc) % (2.0 * np.pi)
    tc_rel = ((tc_mod - tooth_angles[tooth_all] + np.pi) % (2.0 * np.pi)) - np.pi

    return tooth_all, tooth_angles, tc_rel, off_tc, tc_mod


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3 — Normalen + L/R via n_tang  (kein r_b nötig)
# ═══════════════════════════════════════════════════════════════════════════════

def phase3_normals_and_lr(s1, s2, cfg):
    """
    Normalen via KDTree-PCA, dann L/R-Klassifikation via n_tang.

    KEIN r_b, KEIN MY_GEAR.
    n_tang = n · e_θ ist eine rein lokale Oberflächeneigenschaft.

    Helix-Entwindung (helix_unwind_normals): eine Schrägzahnflanke ist ein
    Helikoid — ein Flankenpunkt wandert bei fixem r um k·Δz in theta, wenn
    man in z verschiebt. Die reine z-Dichtekorrektur (z_scale_knn) macht die
    KNN-Nachbarschaft nur isotrop, sie entfernt NICHT diese systematische
    Scherung. Bei nicht-trivialem β wählt KDTree+PCA dadurch einen schräg
    verzerrten Nachbar-Patch → verzerrte (gescherte) Normalen, mit dem
    größten Fehler genau in n_tang — der Komponente, die die L/R-Klassi-
    fikation via gm_product = n_r·n_tang entscheidet.

    Fix: KNN + PCA im "entwundenen" Frame (r, theta - k_fft·z, z) durchführen
    — dort ist die Flanke näherungsweise z-invariant (spiralfrei, wie eine
    Geradverzahnung) — danach die resultierende Normale um (+k_fft·z) je
    Punkt zurückrotieren. k_fft kommt aus Phase 2 (rein FFT-basiert, unab-
    hängig von Phase 3/4) — Restfehler in der Entwindung skalieren mit dem
    FEHLER in k_fft, nicht mit k_fft selbst, d.h. auch ein leicht ungenauer
    Wert ist strikt besser als gar keine Korrektur (= Annahme k=0).
    """
    t0 = time.time()
    log("Phase 3: Normalen + L/R (k-unabhängig)")

    pts_xyz = s1["pts_xyz"].astype(np.float64)
    pts_cyl = s1["pts_cyl"].astype(np.float64)
    axis    = s1["axis"]
    center  = s1["center"]
    e1      = s1["e1"]
    e2      = s1["e2"]
    k_helix = float(s2["k_fft"])
    do_unwind = bool(cfg.get("helix_unwind_normals", True))

    # Radialer Vorfilter (grob: mittleres 70% des r-Range)
    r = pts_cyl[:, 0]
    r_lo = float(np.percentile(r, cfg["r_band_frac_lo"] * 100))
    r_hi = float(np.percentile(r, cfg["r_band_frac_hi"] * 100))
    mask_band = (r >= r_lo) & (r <= r_hi)
    pts_b = pts_xyz[mask_band]
    pts_c = pts_cyl[mask_band]
    log(f"  Radialfilter r∈[{r_lo:.1f}, {r_hi:.1f}]mm: "
        f"{mask_band.sum():,}/{len(pts_xyz):,} Punkte")

    # ── Scan-Struktur erkennen: z-Schritt vs. Arc-Schritt ────────────────────
    # Für strukturierte Zylinderscans (viele Punkte pro z-Ebene) liefert ein
    # isotroper 3D-KDTree nur gleich-z-Nachbarn → PCA gibt n ≈ z_hat → n_tang≈0.
    # Fix: z mit Faktor z_scale_knn skalieren, damit Cross-z-Nachbarn gleichgewichtig.
    z_vals_b  = pts_c[:, 2]
    r_med_b   = float(np.median(pts_c[:, 0]))
    z_range_b = float(np.ptp(z_vals_b))

    # Z-Schritt: aus einzigartigen gerundeten z-Werten
    z_levels_arr = np.unique(np.round(z_vals_b, 2))   # 10-µm-Raster
    if len(z_levels_arr) > 1:
        dz_lev   = np.diff(z_levels_arr)
        dz_lev   = dz_lev[dz_lev > 0.01]              # Nur echte Schritte
        z_step   = float(np.median(dz_lev)) if len(dz_lev) > 0 else z_range_b / 40
        n_lev    = max(1, int(round(z_range_b / z_step)))
    else:
        z_step   = z_range_b / 40
        n_lev    = 40

    # Arc-Schritt: Kreisumfang / Punkte pro z-Ebene
    n_per_lev  = max(1, len(pts_b) / n_lev)
    arc_step   = (2.0 * np.pi * r_med_b) / n_per_lev

    # Skalierungsfaktor: z_scaled = z × z_scale → gleich dicht wie Arc
    z_scale_knn = float(cfg.get("z_scale_knn",
                                 arc_step / max(z_step, 1e-9)))
    log(f"  Scan-Struktur: z_step≈{z_step:.3f}mm  arc_step≈{arc_step:.5f}mm  "
        f"z_scale_knn={z_scale_knn:.5f}  ({n_lev} Ebenen)")

    # ── Referenz-Koordinaten für KNN/PCA ──────────────────────────────────────
    # Aufgebaut aus (r, theta_ref, z) über die lokale (e1, e2, axis)-Basis statt
    # roher Welt-XYZ — bei axis_mode="z" identisch zu vorher (nur Rotations-
    # invarianz von KNN/PCA genutzt), aber korrekt verallgemeinert auf
    # axis_mode="pca" (dort ist Welt-Z NICHT zwingend die Rotationsachse, die
    # z-Skalierung muss entlang `axis` erfolgen, nicht entlang Welt-Index 2).
    #
    # theta_ref = theta - k_fft·z  (Helix-Entwindung, siehe Docstring oben):
    # macht die Flanke näherungsweise z-invariant, sodass KNN echte
    # Oberflächen-Nachbarn statt schräg-geschertem Patch findet.
    axis_u  = axis / np.linalg.norm(axis)
    theta_b = pts_c[:, 1]
    z_b     = pts_c[:, 2]
    r_b_arr = pts_c[:, 0]

    if do_unwind:
        theta_ref = theta_b - k_helix * z_b
        log(f"  Helix-Entwindung aktiv: k_fft={k_helix:.7f}  "
            f"(β={np.degrees(np.arctan(k_helix*s2['r_ref'])):.4f}° bei r_ref={s2['r_ref']:.2f}mm)")
    else:
        theta_ref = theta_b
        log("  Helix-Entwindung deaktiviert (helix_unwind_normals=False)")

    loc_knn = (r_b_arr[:, None] * (np.cos(theta_ref)[:, None] * e1[None, :] +
                                    np.sin(theta_ref)[:, None] * e2[None, :])
               + (z_b * z_scale_knn)[:, None] * axis_u[None, :])

    k_nn = cfg["knn_normals"]
    tree = KDTree(loc_knn)
    _, idx_nn = tree.query(loc_knn, k=k_nn, workers=-1)

    # PCA auf SKALIERTEN Referenz-Koordinaten (loc_knn):
    #   - In physikalischen Koordinaten: z_step≈0.01mm << arc_step≈0.65mm
    #     → Cross-z-Nachbarn haben z-Varianz≈0 → PCA gibt z_hat als Min-EV
    #   - In skalierten Koordinaten: z'_step = z_step × s ≈ arc_step
    #     → Gleichgewichtige 2D-Fläche → PCA gibt radialen Normal
    #
    # Rücktransformation (Gradienten-Theorem, Skalierung T entlang axis: z→z'=sz):
    #   n_phys = n' + (s-1)·(n'·axis)·axis   (nur Achsen-Komponente entskaliert)
    normals_unw = np.empty_like(pts_b)
    curvatures  = np.empty(len(pts_b), dtype=np.float32)
    batch = 25_000
    for s in range(0, len(pts_b), batch):
        e   = min(s + batch, len(pts_b))
        nb  = loc_knn[idx_nn[s:e]]          # (B, k, 3) — SKALIERTE Referenzkoords
        ce  = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', ce, ce)
        ev, evec = np.linalg.eigh(cov)
        curvatures[s:e] = (ev[:, 0] /
                            (ev.sum(axis=1) + 1e-12)).astype(np.float32)
        n_raw  = evec[:, :, 0]                                   # (B,3) skaliert
        n_dot  = n_raw @ axis_u
        n_sc   = n_raw + (z_scale_knn - 1.0) * n_dot[:, None] * axis_u[None, :]
        norms  = np.sqrt((n_sc ** 2).sum(axis=1, keepdims=True))
        normals_unw[s:e] = n_sc / np.maximum(norms, 1e-10)

    # Rückrotation: unwound-Frame → reale Orientierung, Rotation um `axis`
    # um den zum jeweiligen Punkt gehörigen Winkel (+k_helix·z).
    if do_unwind:
        normals = _rotate_about_axis(normals_unw, axis_u, k_helix * z_b)
        norms   = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = normals / np.maximum(norms, 1e-10)
    else:
        normals = normals_unw

    # Lokal-zylindrische Basisvektoren
    p_perp   = pts_b - center
    p_rad_2d = p_perp - np.outer(p_perp @ axis, axis)
    r_hat    = p_rad_2d / np.linalg.norm(p_rad_2d, axis=1, keepdims=True)
    e_theta  = np.cross(axis[None, :], r_hat)

    # ── Gaußsche Karte im lokalen Zylinderrahmen ──────────────────────────────
    #
    # Rohe (ungeflippte) Komponenten:
    #   n_r_raw   = n · r_hat    (radial)
    #   n_tang_raw = n · e_theta (tangential)
    #
    # Schlüssel-Eigenschaft: das PRODUKT n_r * n_tang ist Flip-invariant.
    # Beweis (für Evolventen-Flanken):
    #   R-Flanke, korrekte Orient.:  n_r > 0, n_tang > 0  → Produkt > 0 ✓
    #   R-Flanke, gekippte Orient.:  n_r < 0, n_tang < 0  → Produkt > 0 ✓
    #   L-Flanke, korrekte Orient.:  n_r > 0, n_tang < 0  → Produkt < 0 ✓
    #   L-Flanke, gekippte Orient.:  n_r < 0, n_tang > 0  → Produkt < 0 ✓
    #
    # → sign(n_r · n_tang) = L/R-Klassifikator ohne separaten Flip-Schritt.
    # → Unsicherheit: |n_r · n_tang| < eps_gm, erfasst BEIDE Instabilitäten:
    #     nahe Grundkreis (n_r ≈ 0) UND nahe Kopf-/Fußrund (n_tang ≈ 0).
    n_r_raw    = np.einsum("ij,ij->i", normals, r_hat)
    n_tang_raw = np.einsum("ij,ij->i", normals, e_theta)
    gm_product = n_r_raw * n_tang_raw               # Gaußsche Karte

    # Flip für HDF5-konsistente Ausgabe (nach außen weisende Normalen)
    flip    = np.where(n_r_raw >= 0, 1.0, -1.0)
    normals *= flip[:, None]
    n_rad    = np.abs(n_r_raw)
    n_tang   = n_tang_raw * flip  # nach Flip: sign(n_tang) == sign(gm_product)

    # Gaußsche Karte Unsicherheitsmaske (|n_r · n_tang| < eps_gm)
    eps_gm    = float(cfg.get("gm_eps", 0.012))
    uncertain = np.abs(gm_product) < eps_gm

    # Nachbarvote für unsichere Punkte — vektorisiert via nanmedian
    n_corr = 0
    if uncertain.sum() > 0:
        unc_idx  = np.where(uncertain)[0]
        nb_all   = idx_nn[unc_idx, 1:]                              # (n_unc, k-1)
        nb_rel   = ~uncertain[nb_all]                               # (n_unc, k-1)
        has_rel  = nb_rel.any(axis=1)                              # (n_unc,)
        gm_nb    = np.where(nb_rel, gm_product[nb_all],
                            np.nan).astype(np.float64)             # (n_unc, k-1)
        with np.errstate(all='ignore'):                                # unterdrückt All-NaN-Warnung
            votes = np.sign(np.nanmedian(gm_nb, axis=1))           # (n_unc,); All-NaN → 0
        cur_sg   = np.sign(gm_product[unc_idx])                    # (n_unc,)
        to_flip  = unc_idx[has_rel & (votes != 0) & (cur_sg != votes)]
        normals[to_flip]    *= -1.0
        n_tang[to_flip]     *= -1.0
        gm_product[to_flip] *= -1.0
        n_corr = len(to_flip)
        log(f"  Gaußsche Karte Nachbarvote: {uncertain.sum():,} unsichere Pkt, "
            f"{n_corr} korrigiert")

    # ── DIAGNOSE-BLOCK ──────────────────────────────────────────────────────────
    # Krümmungsfilter
    n_curv_pass = int((curvatures < cfg["curv_max"]).sum())
    log(f"  [Diag] Krümmungsfilter curv<{cfg['curv_max']}: "
        f"{n_curv_pass:,}/{len(curvatures):,} ({100*n_curv_pass/len(curvatures):.1f}%)")

    # n_z-Komponente — Diagnose für Stirnflächen-Punkte (n_z≈1 → gm≈0)
    # Erwartung Flanke:   |n_z| ≈ sin(β_b) ≈ 0.35
    # Erwartung Stirn:    |n_z| ≈ 1.0
    # Erwartung Kopfkreis: |n_z| ≈ 0
    n_z_abs = np.abs(normals[:, 2])
    log(f"  [Diag] n_z-Verteilung: "
        f"|n_z|>0.7: {(n_z_abs>0.7).mean()*100:.1f}%  "
        f"|n_z|>0.5: {(n_z_abs>0.5).mean()*100:.1f}%  "
        f"|n_z|<0.1: {(n_z_abs<0.1).mean()*100:.1f}%  "
        f"|n_z|∈[0.2,0.5]: {((n_z_abs>0.2)&(n_z_abs<0.5)).mean()*100:.1f}%")

    # n_tang-Verteilung
    n_ta = np.abs(n_tang_raw)
    log(f"  [Diag] |n_tang|: "
        f"<0.05: {(n_ta<0.05).mean()*100:.1f}%  "
        f"0.05–0.15: {((n_ta>0.05)&(n_ta<0.15)).mean()*100:.1f}%  "
        f"0.15–0.25: {((n_ta>0.15)&(n_ta<0.25)).mean()*100:.1f}%  "
        f">0.25: {(n_ta>0.25).mean()*100:.1f}%  "
        f"median={np.median(n_ta):.3f}")

    # gm_product Histogramm
    _gm_edges = [-1., -.3, -.1, -.05, -.012, -.008, 0., .008, .012, .05, .1, .3, 1.]
    _gm_cnt, _ = np.histogram(gm_product, bins=_gm_edges)
    _gm_str = "  ".join(
        f"[{lo:.3f},{hi:.3f}): {c/len(gm_product)*100:.1f}%"
        for lo, hi, c in zip(_gm_edges[:-1], _gm_edges[1:], _gm_cnt)
        if c > 0
    )
    log(f"  [Diag] gm_product Hist: {_gm_str}")

    # r-Verteilung der gm-selektierten Flankenpunkte
    _r_all  = pts_c[:, 0]
    _mask_gm = np.abs(gm_product) >= eps_gm
    if _mask_gm.any():
        _r_gm = _r_all[_mask_gm]
        log(f"  [Diag] Flanken-r (|gm|≥{eps_gm}): "
            f"[{_r_gm.min():.1f}, {_r_gm.max():.1f}]mm  "
            f"Q25={np.percentile(_r_gm,25):.1f}  "
            f"Median={np.median(_r_gm):.1f}  "
            f"Q75={np.percentile(_r_gm,75):.1f}")
    # ── Ende DIAGNOSE ────────────────────────────────────────────────────────────

    # Gaußsche Karte Statistik (Cluster-Trennung)
    mask_cand = curvatures < cfg["curv_max"]
    gm_R_cand = gm_product[mask_cand & (gm_product > eps_gm)]
    gm_L_cand = gm_product[mask_cand & (gm_product < -eps_gm)]
    if len(gm_R_cand) > 0 and len(gm_L_cand) > 0:
        log(f"  Gaußsche Karte: R-Cluster μ={gm_R_cand.mean():.3f} "
            f"σ={gm_R_cand.std():.3f}  |  "
            f"L-Cluster μ={gm_L_cand.mean():.3f} "
            f"σ={gm_L_cand.std():.3f}")

    # L/R-Masken via Gaußscher Karte (eps_gm = kombinierte Unsicherheitszone)
    mask_clean = (curvatures < cfg["curv_max"])
    mask_R     = mask_clean & (gm_product > eps_gm)
    mask_L     = mask_clean & (gm_product < -eps_gm)

    ratio = mask_R.sum() / max(mask_L.sum(), 1)
    # Für Halb-Scans eines Schrägzahnrads ist Ratio ≠ 1 geometrisch erwartet:
    # der Scanner sieht je nach Schrägungswinkel eine Flankenart bevorzugt.
    # Warnschwelle daher 0.5–2.0 statt 0.85–1.15 (Vollscan-Norm).
    if ratio < 0.1 or ratio > 10.0:
        sym_str = "⚠ STARK asymmetrisch — L/R-Klassifikation prüfen"
    elif ratio < 0.5 or ratio > 2.0:
        sym_str = "⚠ asymmetrisch (Halb-Scan Geometrie erwartet)"
    else:
        sym_str = "✓ akzeptabel"
    log(f"  R (gm>0): {mask_R.sum():7,}   L (gm<0): {mask_L.sum():7,}   "
        f"Ratio: {ratio:.3f}  ({sym_str})", t0)

    return dict(
        band_mask=mask_band,
        pts_b_xyz=pts_b.astype(np.float32),
        pts_b_cyl=pts_c.astype(np.float32),
        normals=normals.astype(np.float32),
        n_tang=n_tang.astype(np.float32),
        n_rad=n_rad.astype(np.float32),
        gm_product=gm_product.astype(np.float32),
        curvatures=curvatures,
        nn_idx=idx_nn,
        mask_R=mask_R,
        mask_L=mask_L,
        # r_hat / e_theta: nicht von nachgelagerten Phasen gelesen — weggelassen
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4 — Präzises k_star + r_b_ist via phi_b-Optimierung
# ═══════════════════════════════════════════════════════════════════════════════

def phase4_optimize_k_rb(s1, s2, s3, cfg):
    """
    Minimiert std(phi_b_R) über (k_star, r_b) auf R-Flanken-Kandidaten.

    phi_b_R = (θ - k·z) + inv(arccos(r_b/r))  soll für alle R-Punkte konstant sein.

    Warum das funktioniert:
      Auf einer Evolventenflanke ist phi_b geometrisch konstant.
      Wenn k oder r_b falsch sind, streut phi_b ∝ Fehler.
      → 2D-Minimum gibt gleichzeitig k_star_ist UND r_b_ist.

    Wichtig: R-Kandidaten (mask_R) kommen NICHT mehr aus Phase 3 (gm_product).
    Stattdessen: phi_b Winner-Takes-All mit k0 und r_b_init=0.925*r_ref.
    Grund: Phase 3 Normalen sind bei Stirnflächen-Punkten fast axial
    (n_z≈1 → gm≈0 → Rauschen-Zuweisung), was Phase 4 korrumpiert hätte.
    """
    t0 = time.time()
    log("Phase 4: Joint-Optimierung k_star + r_b_ist via phi_b")

    pts_c        = s3["pts_b_cyl"].astype(np.float64)
    _curv_raw    = s3["curvatures"]
    tooth_period = s2["tooth_period"]
    k0           = s2["k_fft"]
    r_ref        = s2["r_ref"]
    n_teeth      = s2["n_teeth"]

    # ── z-Band-Filter (optional) ──────────────────────────────────────────────
    # Blendet Randabschnitte aus, wo Scanner-Inzidenzwinkel-Bias r_b verfälscht.
    _z_band = cfg.get("phase4_z_band", None)
    if _z_band is not None:
        _z_lo, _z_hi = float(_z_band[0]), float(_z_band[1])
        _z_mask = (pts_c[:, 2] >= _z_lo) & (pts_c[:, 2] <= _z_hi)
        _n_before = len(pts_c)
        pts_c    = pts_c[_z_mask]
        _curv    = _curv_raw[_z_mask]
        log(f"  Phase 4 z-Band-Filter: z∈[{_z_lo:.1f},{_z_hi:.1f}]mm → "
            f"{len(pts_c):,}/{_n_before:,} Punkte ({len(pts_c)/_n_before*100:.1f}%)")
        if len(pts_c) < 1000:
            log("  ⚠ z-Band liefert < 1000 Punkte — Ergebnis unsicher")
    else:
        _curv = _curv_raw

    r_all     = pts_c[:, 0]
    theta_all = pts_c[:, 1]
    z_all     = pts_c[:, 2]
    curv_all  = _curv

    # ds=0 (Default) → volle Punktzahl. k_star_ist/r_b_ist sind globale
    # Referenzwerte, gegen die JEDES Punkt-Residual (d_lot) gemessen wird —
    # Subsampling-Rauschen hier wäre kein lokaler Fehler, sondern ein
    # systematischer Versatz im gesamten Ist/Soll-Residualplot.
    ds = int(cfg.get("ds_opt", 0))
    _rng_sample = np.random.default_rng(42)   # seeded -- s. ds_fft-Kommentar Phase 2

    def _sample(mask):
        idx = np.where(mask)[0]
        if ds and len(idx) > ds:
            idx = _rng_sample.choice(idx, ds, replace=False)
        return (pts_c[idx, 0],   # r
                pts_c[idx, 1],   # theta
                pts_c[idx, 2])   # z

    # ── tc_rel-basierte Flanken-Separation (r_b-unabhängig) ─────────────────
    # WARUM: phi_b Vorab-Klassifikation mit r_b_init erzeugt einen Zirkelschluss:
    #   falscher r_b_init → falsche L/R-Maske → falsches r_b in 1D-Scan → Fehler-Kaskade.
    # LÖSUNG: tc_rel = (theta-k0*z) - Zahnzentrum, zentriert auf [-T/2, +T/2].
    #   sign(tc_rel) trennt die beiden Flanken geometrisch OHNE Kenntnis von r_b.
    #   Vollständig r_b-unabhängig → keine Fehler-Kaskade möglich.
    tooth_idx_tc, tooth_ang_tc, tc_rel_p, off_tc, tc_p = compute_tooth_index(
        theta_all, z_all, k0, n_teeth, tooth_period)

    curv_ok   = curv_all < cfg["curv_max"]
    # ±40% der Teilung = Flankenpunkte; Lücke/Kopf/Fase ausschließen
    in_flank  = np.abs(tc_rel_p) < (tooth_period * 0.40)
    mask_plus  = in_flank & curv_ok & (tc_rel_p > 0)   # eine Flanke
    mask_minus = in_flank & curv_ok & (tc_rel_p < 0)   # andere Flanke

    log(f"  tc_rel-Split (k0={k0:.6f}): "
        f"plus={mask_plus.sum():,}  minus={mask_minus.sum():,}  "
        f"Ratio={mask_plus.sum()/max(mask_minus.sum(),1):.2f}")

    r_plus,  th_plus,  z_plus  = _sample(mask_plus)
    r_minus, th_minus, z_minus = _sample(mask_minus)

    def phi_b_circ_var(r, theta, z, k, r_b, sign):
        """Kreisstreuung von phi_b mod tooth_period. Kleiner = besser."""
        r_safe   = np.maximum(r, r_b * 1.0005)
        alpha_t  = np.arccos(np.clip(r_b / r_safe, 0.0, 1.0))
        inv_at   = np.tan(alpha_t) - alpha_t
        phi_b    = (theta - k * z + sign * inv_at) % tooth_period
        ang      = phi_b / tooth_period * 2 * np.pi
        return 1.0 - float(np.abs(np.mean(np.exp(1j * ang))))

    # 1D r_b-Scan: kombinierte Kosten über beide Seiten (beide Vorzeichen getestet)
    # Bei korrektem r_b erreicht ein Vorzeichen pro Seite Kreisstreuung ≈ 0.
    # Kombiniertes Minimum ist deutlich schärfer/tiefer als Einzel-Seite.
    r_b_range = np.linspace(0.86 * r_ref, 0.99 * r_ref, 60)
    cv_scan   = []
    for rb in r_b_range:
        cv_pp = phi_b_circ_var(r_plus,  th_plus,  z_plus,  k0, rb, +1)
        cv_pn = phi_b_circ_var(r_plus,  th_plus,  z_plus,  k0, rb, -1)
        cv_mp = phi_b_circ_var(r_minus, th_minus, z_minus, k0, rb, +1)
        cv_mn = phi_b_circ_var(r_minus, th_minus, z_minus, k0, rb, -1)
        cv_scan.append(min(cv_pp, cv_pn) + min(cv_mp, cv_mn))

    r_b_start = float(r_b_range[np.argmin(cv_scan)])
    log(f"  r_b Startwert (1D-Scan, tc_rel): {r_b_start:.3f} mm  "
        f"→ α_t0 ≈ {np.degrees(np.arccos(r_b_start/r_ref)):.2f}°")

    # Vorzeichen-Bestimmung: welche Kombination (plus→R, minus→L) oder (plus→L, minus→R)?
    rb_s = r_b_start
    cv_pp = phi_b_circ_var(r_plus,  th_plus,  z_plus,  k0, rb_s, +1)
    cv_pn = phi_b_circ_var(r_plus,  th_plus,  z_plus,  k0, rb_s, -1)
    cv_mp = phi_b_circ_var(r_minus, th_minus, z_minus, k0, rb_s, +1)
    cv_mn = phi_b_circ_var(r_minus, th_minus, z_minus, k0, rb_s, -1)

    # sign_plus: welches Vorzeichen passt besser zur plus-Seite?
    sign_plus  = +1 if cv_pp <= cv_pn else -1
    sign_minus = -sign_plus
    log(f"  Flanken-Konvention: "
        f"{'plus(tc_rel>0)=R, minus(tc_rel<0)=L' if sign_plus==+1 else 'plus(tc_rel>0)=L, minus(tc_rel<0)=R'}"
        f"  [cv_pp={1-cv_pp:.4f} cv_pn={1-cv_pn:.4f} | "
        f"cv_mp={1-cv_mp:.4f} cv_mn={1-cv_mn:.4f}]")

    # R/L-Masken und Samples (R=phi_b mit +1, L=phi_b mit -1)
    if sign_plus == +1:
        mask_R = mask_plus
        mask_L = mask_minus
        r_R, th_R, z_R = r_plus,  th_plus,  z_plus
        r_L, th_L, z_L = r_minus, th_minus, z_minus
    else:
        mask_R = mask_minus
        mask_L = mask_plus
        r_R, th_R, z_R = r_minus, th_minus, z_minus
        r_L, th_L, z_L = r_plus,  th_plus,  z_plus

    ratio_round1 = mask_R.sum() / max(mask_L.sum(), 1)
    log(f"  R={mask_R.sum():,}  L={mask_L.sum():,}  "
        f"Ratio={ratio_round1:.2f}")

    # ── [Diag] Kontaminations-Check (2026-07-14), KEINE Verhaltensaenderung ──
    # Phase 5's Symmetrisierung deckte auf, dass der ROHE (R/L-Punktzahl-
    # gewichtete) off_tc das Vorfilter-Fenster Richtung R verschiebt — R's
    # Fenster effektiv enger (schneidet echte Spitzen-Punkte ab), L's Fenster
    # effektiv breiter (nimmt Fase-/Fuss-Rauschen mit auf). Phase 4 baut HIER
    # (Zeile ~1167-1179) seinen eigenen R/L-Split aus DERSELBEN unsymmetrischen
    # off_tc-Berechnung — moeglicher Erklaerungsansatz fuer Δr_b(R-L)=5.4mm
    # und die schlechtere L-Guete (cv_L) oben. Rein diagnostisch: loggt die
    # Kandidaten-Abdeckung (r, tc_rel) je Seite und einen unabhaengigen φ_b-
    # Peak-Vergleich (analog zur Phase-5-Symmetrisierungs-Diagnose), aendert
    # aber nichts an mask_R/mask_L/r_b_start/off_tc selbst.
    r_lo_R, r_hi_R = float(np.percentile(r_all[mask_R], 1)), float(np.percentile(r_all[mask_R], 99))
    r_lo_L, r_hi_L = float(np.percentile(r_all[mask_L], 1)), float(np.percentile(r_all[mask_L], 99))
    tc_lo_R, tc_hi_R = float(tc_rel_p[mask_R].min()), float(tc_rel_p[mask_R].max())
    tc_lo_L, tc_hi_L = float(tc_rel_p[mask_L].min()), float(tc_rel_p[mask_L].max())
    log(f"  [Diag] Kandidaten-Abdeckung R: r∈[{r_lo_R:.2f},{r_hi_R:.2f}]mm "
        f"(1.-99. Perz.)  tc_rel∈[{np.degrees(tc_lo_R):.2f}°,{np.degrees(tc_hi_R):.2f}°]")
    log(f"  [Diag] Kandidaten-Abdeckung L: r∈[{r_lo_L:.2f},{r_hi_L:.2f}]mm "
        f"(1.-99. Perz.)  tc_rel∈[{np.degrees(tc_lo_L):.2f}°,{np.degrees(tc_hi_L):.2f}°]")

    def _phi_b_peak_deg(r, theta, z, rb, sign):
        r_safe  = np.maximum(r, rb * 1.0005)
        alpha_t = np.arccos(np.clip(rb / r_safe, 0.0, 1.0))
        inv_at  = np.tan(alpha_t) - alpha_t
        phi_b   = (theta - k0 * z + sign * inv_at) % tooth_period
        ang     = phi_b / tooth_period * 2.0 * np.pi
        peak    = np.arctan2(np.mean(np.sin(ang)), np.mean(np.cos(ang))) % (2.0 * np.pi)
        return float(peak / (2.0 * np.pi) * tooth_period)

    peak_R_diag = _phi_b_peak_deg(r_all[mask_R], theta_all[mask_R], z_all[mask_R], r_b_start, +1)
    peak_L_diag = _phi_b_peak_deg(r_all[mask_L], theta_all[mask_L], z_all[mask_L], r_b_start, -1)
    delta_diag  = ((peak_R_diag - peak_L_diag + tooth_period / 2.0) % tooth_period) - tooth_period / 2.0
    log(f"  [Diag] φ_b-Peak (bei r_b_start={r_b_start:.2f}mm): R={np.degrees(peak_R_diag):.3f}°  "
        f"L={np.degrees(peak_L_diag):.3f}°  Δ(R-L)={np.degrees(delta_diag):.3f}°  "
        f"(roh off_tc={np.degrees(off_tc):.3f}° zum Vergleich — eine grosse "
        f"Abweichung hier deutet auf dieselbe R-Dichte-Verzerrung wie in "
        f"Phase 5 hin, siehe [[project_gear_pipeline]])")

    # ── Runde 2: Kandidaten mit symmetrisiertem off_tc neu waehlen (2026-07-14) ──
    # Bestaetigt durch obigen Diagnose-Block (Δ(R-L)=grosse Abweichung, R/L
    # radial fast disjunkt) -- dasselbe Muster, das in Phase 5 gefunden und
    # behoben wurde. off_tc_sym (zirkulaerer Mittelwert der beiden unabhaengigen
    # φ_b-Peaks) ersetzt das rohe, R-Dichte-verzerrte off_tc als Vorfilter-
    # Zentrum. WICHTIG: die R/L-Konvention (welches Vorzeichen zu welcher
    # Seite gehoert) wird in der NEUEN Referenz FRISCH bestimmt, nicht das
    # alte sign_plus wiederverwendet -- genau das war der Bug im ersten
    # Phase-5-Fixversuch (Referenzrahmen-Wechsel kann das Vorzeichen kippen).
    ang_R_diag   = peak_R_diag / tooth_period * 2.0 * np.pi
    ang_L_diag   = peak_L_diag / tooth_period * 2.0 * np.pi
    ang_avg_diag = np.arctan2(np.sin(ang_R_diag) + np.sin(ang_L_diag),
                               np.cos(ang_R_diag) + np.cos(ang_L_diag))
    off_tc_sym   = float((ang_avg_diag / (2.0 * np.pi)) * tooth_period % tooth_period)
    log(f"  Symmetrisierung: off_tc roh={np.degrees(off_tc):.3f}° → "
        f"symmetrisiert={np.degrees(off_tc_sym):.3f}°")

    tooth_idx_tc_sym, tooth_ang_tc_sym, tc_rel_sym, _, _ = compute_tooth_index(
        theta_all, z_all, k0, n_teeth, tooth_period, off_tc=off_tc_sym)

    in_flank_sym   = np.abs(tc_rel_sym) < (tooth_period * 0.40)
    mask_plus_sym  = in_flank_sym & curv_ok & (tc_rel_sym > 0)
    mask_minus_sym = in_flank_sym & curv_ok & (tc_rel_sym < 0)

    r_plus_s,  th_plus_s,  z_plus_s  = _sample(mask_plus_sym)
    r_minus_s, th_minus_s, z_minus_s = _sample(mask_minus_sym)

    cv_pp_s = phi_b_circ_var(r_plus_s,  th_plus_s,  z_plus_s,  k0, r_b_start, +1)
    cv_pn_s = phi_b_circ_var(r_plus_s,  th_plus_s,  z_plus_s,  k0, r_b_start, -1)
    cv_mp_s = phi_b_circ_var(r_minus_s, th_minus_s, z_minus_s, k0, r_b_start, +1)
    cv_mn_s = phi_b_circ_var(r_minus_s, th_minus_s, z_minus_s, k0, r_b_start, -1)
    sign_plus_s = +1 if cv_pp_s <= cv_pn_s else -1
    log(f"  Flanken-Konvention Runde 2: "
        f"{'plus(tc_rel_sym>0)=R' if sign_plus_s == +1 else 'plus(tc_rel_sym>0)=L'}  "
        f"[cv_pp={1 - cv_pp_s:.4f} cv_pn={1 - cv_pn_s:.4f} | "
        f"cv_mp={1 - cv_mp_s:.4f} cv_mn={1 - cv_mn_s:.4f}]")

    if sign_plus_s == +1:
        mask_R = mask_plus_sym
        mask_L = mask_minus_sym
        r_R, th_R, z_R = r_plus_s,  th_plus_s,  z_plus_s
        r_L, th_L, z_L = r_minus_s, th_minus_s, z_minus_s
    else:
        mask_R = mask_minus_sym
        mask_L = mask_plus_sym
        r_R, th_R, z_R = r_minus_s, th_minus_s, z_minus_s
        r_L, th_L, z_L = r_plus_s,  th_plus_s,  z_plus_s

    ratio_round2 = mask_R.sum() / max(mask_L.sum(), 1)
    log(f"  R (Runde 2, final)={mask_R.sum():,}  L (Runde 2, final)={mask_L.sum():,}  "
        f"Ratio={ratio_round2:.2f}")
    r_lo_R2, r_hi_R2 = float(np.percentile(r_all[mask_R], 1)), float(np.percentile(r_all[mask_R], 99))
    r_lo_L2, r_hi_L2 = float(np.percentile(r_all[mask_L], 1)), float(np.percentile(r_all[mask_L], 99))
    log(f"  [Diag] Kandidaten-Abdeckung Runde 2 R: r∈[{r_lo_R2:.2f},{r_hi_R2:.2f}]mm  "
        f"L: r∈[{r_lo_L2:.2f},{r_hi_L2:.2f}]mm")

    # Sanity-Check (gleiches Muster wie in Phase 5 verwendet): R:L-Verhaeltnis
    # sollte durch die Symmetrisierung nicht kippen (>1 vs <1) oder um >2x
    # springen — beides waere ein Anzeichen fuer denselben Referenzrahmen-
    # Fehler, der beim ersten Phase-5-Fixversuch auftrat.
    if ((ratio_round1 > 1.0) != (ratio_round2 > 1.0)
            or ratio_round2 > ratio_round1 * 2.0
            or ratio_round2 < ratio_round1 / 2.0):
        log(f"  ⚠⚠ VERDACHT auf Referenzrahmen-Fehler: Runde-1-Ratio="
            f"{ratio_round1:.2f} vs Runde-2-Ratio={ratio_round2:.2f} weichen "
            f"zu stark ab — Ergebnis vor Verwendung pruefen!")

    # 2D Joint-Optimierung auf R-Flanken
    # L-BFGS-B mit expliziten Bounds — kein Penalty-Cliff mehr.
    # k-Bounds: ±15% um k0 (deckt beide Vorzeichen ab).
    # r_b-Bounds: [0.80·r_ref, 0.999·r_ref].
    k_lo = min(k0 * 0.85, k0 * 1.15)
    k_hi = max(k0 * 0.85, k0 * 1.15)
    if abs(k_hi - k_lo) < 1e-9:           # Spur-Schrägungswinkel ≈ 0
        k_lo, k_hi = k0 - 5e-5, k0 + 5e-5

    res_R = minimize(
        lambda p: phi_b_circ_var(r_R, th_R, z_R, p[0], p[1], +1),
        x0=[k0, r_b_start],
        method='L-BFGS-B',
        bounds=[(k_lo, k_hi), (0.80 * r_ref, 0.999 * r_ref)],
        options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 5000},
    )
    k_opt_R, r_b_opt_R = res_R.x
    cv_R = float(res_R.fun)

    # Verifikation auf L-Flanken (unabhängige Bestätigung)
    res_L = minimize(
        lambda p: phi_b_circ_var(r_L, th_L, z_L, p[0], p[1], -1),
        x0=[k_opt_R, r_b_opt_R],   # Startwert aus R-Ergebnis
        method='L-BFGS-B',
        bounds=[(k_lo, k_hi), (0.80 * r_ref, 0.999 * r_ref)],
        options={'ftol': 1e-15, 'gtol': 1e-10, 'maxiter': 5000},
    )
    k_opt_L, r_b_opt_L = res_L.x
    cv_L = float(res_L.fun)

    # Mitteln (R und L sollten dasselbe r_b liefern)
    dk    = abs(k_opt_R - k_opt_L)
    dr_b  = abs(r_b_opt_R - r_b_opt_L)
    if dr_b > 0.5:
        log(f"  ⚠ r_b Diskrepanz R vs L: {dr_b:.3f} mm — nutze R-Ergebnis")
        k_star = k_opt_R
        r_b    = r_b_opt_R
    else:
        k_star = (k_opt_R + k_opt_L) / 2
        r_b    = (r_b_opt_R + r_b_opt_L) / 2

    beta_ist    = float(np.degrees(np.arctan(k_star * r_ref)))
    alpha_t0    = float(np.arccos(np.clip(r_b / r_ref, 0, 1)))
    alpha_t0_deg = np.degrees(alpha_t0)

    log(f"  k_star_ist = {k_star:.7f}   β_ist = {beta_ist:.4f}°  "
        f"(FFT war {np.degrees(np.arctan(s2['k_fft']*r_ref)):.4f}°)")
    log(f"  r_b_ist    = {r_b:.4f} mm   α_t0_ist = {alpha_t0_deg:.3f}°")
    log(f"  phi_b Kreisschärfe: R={1-cv_R:.4f}  L={1-cv_L:.4f}  "
        f"(1.0 = perfekt)")
    log(f"  Δk(R-L)={dk*1e6:.1f} ppm   Δr_b(R-L)={dr_b*1000:.1f} µm", t0)

    return dict(
        k_star=k_star, r_b=r_b,
        beta_ist=beta_ist, alpha_t0_ist=alpha_t0,
        r_b_R=r_b_opt_R, r_b_L=r_b_opt_L,
        k_R=k_opt_R, k_L=k_opt_L,
        r_b_scan_range=r_b_range,
        r_b_scan_cv=cv_scan,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 — Zahn-Zuordnung via phi_b mod Teilung
# ═══════════════════════════════════════════════════════════════════════════════

def _phi_b_phase_offset(phi_b_vals, n, tooth_period):
    """
    FFT-Phasenerkennung: wo liegt Zahn 0 im phi_b-Histogramm?
    offset = (-phase / n) % tooth_period  [korrekte Formel, kein 2π-Bug]
    """
    if len(phi_b_vals) < n * 5:
        return 0.0
    n_bins  = n * 40
    hist, _ = np.histogram(phi_b_vals, bins=n_bins, range=(0, 2*np.pi))
    fft_h   = np.fft.rfft(hist)
    phase   = np.angle(fft_h[n])
    return float((-phase / n) % tooth_period)


def phase5_tooth_assignment(s1, s2, s3, s4, cfg):
    """
    Zahn-Zuordnung via phi_b_ist mod Teilung.
    Verwendet k_star_ist + r_b_ist aus Phase 4 (datengetrieben).

    Eingriffsfläche-Restriktion (datengetrieben, kein MY_GEAR):
    Zwei Filter grenzen die Punkte auf die aktive/nutzbare Evolventenflanke
    ein, bevor sie den finalen Zahn/Seite-Clustern zugeordnet werden:
      1. Krümmungsfilter (curvatures < curv_max, aus Phase 3): verwirft
         Fußrundung/Trochoide (konkav, hohe Krümmung) und Ausreißer/Rauschen.
         War bisher NUR im Phase-4-Fit-Subsample aktiv, nicht im finalen
         Cluster — Fußrundungspunkte mit r>=r_b konnten also bis in die
         Chebyshev-Fit-Cluster durchrutschen.
      2. Absoluter phi_b-Residuen-Schwellwert (phi_b_tol_frac, Stufe ≥1):
         eine Kopfkante/Fase (Chamfer) ist zwar oft LOKAL FLACH (Krümmungs-
         filter greift dort nicht), aber sie liegt nicht auf der Ideal-
         Evolvente — ihr phi_b_ist weicht daher systematisch vom Zahn-
         Zentrum ab. Echte Fertigungsabweichungen (das Messsignal, das
         Phase 4-UV eigentlich sehen soll) liegen bei diesem Rad-Maßstab
         typischerweise <0.1° (~100µm/r_b) — Stufe 1 (~0.43°) lässt dafür
         >4× Spielraum, verwirft aber Chamfer/Fußrundungs-Ausreißer, deren
         Abweichung kategorial größer ist.
    """
    t0 = time.time()
    log("Phase 5: Zahn-Zuordnung (phi_b_ist)")

    pts_c   = s3["pts_b_cyl"].astype(np.float64)
    r       = pts_c[:, 0]
    theta   = pts_c[:, 1]
    z       = pts_c[:, 2]
    curv    = s3["curvatures"]

    k_star       = s4["k_star"]
    r_b          = s4["r_b"]
    n_teeth      = s2["n_teeth"]
    tooth_period = s2["tooth_period"]

    # phi_b für ALLE Bandpunkte
    r_safe  = np.maximum(r, r_b * 1.0005)
    alpha_t = np.arccos(np.clip(r_b / r_safe, 0.0, 1.0))
    inv_at  = np.tan(alpha_t) - alpha_t
    tc      = theta - k_star * z

    phi_b_R = (tc + inv_at) % (2 * np.pi)
    phi_b_L = (tc - inv_at) % (2 * np.pi)

    # ── Offset-Schätzung via tc_rel-gefilterter FFT ──────────────────────────
    # PROBLEM: bei starker Flanken-Asymmetrie (R:L >> 1 durch Scan-Geometrie)
    # findet _phi_b_phase_offset in phi_b_L den SCHATTEN der R-Flanke
    # (phi_b_L_R-Punkte = tc - inv_at = C_R - 2×inv_at ≠ C_L), nicht den
    # echten L-Peak → off_L falsch → WTA weist L-Punkte der R-Seite zu.
    #
    # FIX: tc_rel trennt Flanken r_b-unabhängig via sign(theta_corr - Zahnzentrum).
    # off_R wird nur aus tc_rel>0-Seite (echte R-Flanke) geschätzt,
    # off_L nur aus tc_rel<0-Seite (echte L-Flanke). Keine Kontamination mehr.
    _tc5    = (theta - k_star * z) % (2.0 * np.pi)
    _nb5    = n_teeth * 40
    _ht5, _ = np.histogram(_tc5, bins=_nb5, range=(0.0, 2.0 * np.pi))
    _ft5    = np.fft.rfft(_ht5)
    _ot5    = float((-np.angle(_ft5[n_teeth]) / n_teeth) % tooth_period)
    _ti5    = (np.round(((_tc5 - _ot5 + 4.0*np.pi) % (2.0*np.pi)) / tooth_period)
               .astype(int) % n_teeth)
    _ta5    = (np.arange(n_teeth) * tooth_period + _ot5) % (2.0 * np.pi)
    _tr5    = ((_tc5 - _ta5[_ti5] + np.pi) % (2.0 * np.pi)) - np.pi   # tc_rel

    # Welche tc_rel-Seite ist R? Kleinere phi_b_R-Streuung gewinnt.
    _safe5  = np.maximum(r, r_b * 1.0005)
    _iat5   = np.tan(np.arccos(np.clip(r_b / _safe5, 0.0, 1.0))) \
              - np.arccos(np.clip(r_b / _safe5, 0.0, 1.0))
    _mp  = ((_tc5 + _iat5)[_tr5 > 0]) % (2.0 * np.pi)   # phi_b_R auf plus-Seite
    _mm  = ((_tc5 + _iat5)[_tr5 < 0]) % (2.0 * np.pi)   # phi_b_R auf minus-Seite
    def _frac_cv(v, tp):
        if len(v) < 20: return 1.0
        return 1.0 - float(np.abs(np.mean(np.exp(1j * v / tp * 2 * np.pi))))
    _cvp = _frac_cv(_mp, tooth_period)
    _cvm = _frac_cv(_mm, tooth_period)
    _plus_is_R = _cvp < _cvm   # True wenn tc_rel>0 = R-Flanke

    _mask_R5 = _tr5 > 0 if _plus_is_R else _tr5 < 0
    _mask_L5 = _tr5 < 0 if _plus_is_R else _tr5 > 0

    off_R = _phi_b_phase_offset(phi_b_R[_mask_R5], n_teeth, tooth_period)
    off_L = _phi_b_phase_offset(phi_b_L[_mask_L5], n_teeth, tooth_period)
    log(f"  phi_b Offsets (tc_rel-gefiltert): "
        f"R={np.degrees(off_R):.3f}°  L={np.degrees(off_L):.3f}°  "
        f"[plus_is_R={_plus_is_R}  cv+={1-_cvp:.4f}  cv-={1-_cvm:.4f}]")

    def assign(phi_b, offset):
        shifted  = (phi_b - offset + 4*np.pi) % (2*np.pi)
        tooth    = np.round(shifted / tooth_period).astype(int) % n_teeth
        phi_peak = (offset + tooth * tooth_period) % (2*np.pi)
        residual = np.abs(((phi_b - phi_peak + np.pi) % (2*np.pi)) - np.pi)
        return tooth, residual

    tooth_R, res_R = assign(phi_b_R, off_R)
    tooth_L, res_L = assign(phi_b_L, off_L)

    # ── Eingriffsfläche-Vorfilter: r>=r_b UND Krümmungsfilter ────────────────
    # curv_max schließt Fußrundung/Trochoide (hohe Krümmung) sowie Rauschen
    # aus — bisher nur für den Phase-4-Fit genutzt, jetzt auch hier, damit
    # diese Punkte erst gar nicht in die finalen Zahn/Seite-Cluster gelangen.
    curv_ok  = curv < cfg["curv_max"]
    above_rb = (r >= r_b) & curv_ok
    n_excl_curv = int(((r >= r_b) & ~curv_ok).sum())
    log(f"  [Diag] Krümmungsfilter verwirft {n_excl_curv:,} Punkte mit r≥r_b "
        f"(vermutlich Fußrundung/Rauschen)")

    _pR = np.degrees(np.percentile(res_R[above_rb], [10, 25, 50, 75, 90]))
    _pL = np.degrees(np.percentile(res_L[above_rb], [10, 25, 50, 75, 90]))
    log(f"  [Diag] res_R Perzentile (P10/25/50/75/90): "
        f"{_pR[0]:.3f}° / {_pR[1]:.3f}° / {_pR[2]:.3f}° / {_pR[3]:.3f}° / {_pR[4]:.3f}°")
    log(f"  [Diag] res_L Perzentile (P10/25/50/75/90): "
        f"{_pL[0]:.3f}° / {_pL[1]:.3f}° / {_pL[2]:.3f}° / {_pL[3]:.3f}° / {_pL[4]:.3f}°")

    # ── L/R-Zuweisung ────────────────────────────────────────────────────────
    # Stufenweise Filterung via CFG["phi_b_tol_frac"]:
    #   0.0 (Stufe 0): Winner-Takes-All — kein absoluter Schwellwert
    #   0.10 (Stufe 1): schmales Evolventen-Band  ← Default: Eingriffsfläche-Gate
    #   0.25 (Stufe 2): Standard-Flankenbandbreite
    #   0.35 (Stufe 3): breites Band (Randpunkte)
    # Stufe 0 hat KEINEN absoluten Qualitäts-Cutoff — jeder r>=r_b-Punkt (auch
    # Kopfkante/Fase/Rand) geht an die näher liegende Seite, selbst wenn beide
    # Residuen groß sind. Für die finalen Chebyshev-Fit-Cluster mindestens
    # Stufe 1 (siehe Docstring oben: Fertigungsabweichung <0.1° << 0.43°-Band).
    tol_frac = float(cfg.get("phi_b_tol_frac", 0.10))

    if tol_frac <= 0.0:
        # STUFE 0 — Winner-Takes-All: kein Schwellwert, r >= r_b genügt
        mask_R_ok = above_rb & (res_R <= res_L)
        mask_L_ok = above_rb & (res_L <  res_R)
        log(f"  phi_b Filter: Stufe 0 — Winner-Takes-All (kein Threshold)")
    else:
        # STUFE 1–3 — phi_b-Band
        tol_all   = tooth_period * tol_frac
        mask_R_ok = above_rb & (res_R < tol_all)
        mask_L_ok = above_rb & (res_L < tol_all)
        # Doppelzuweisungen: kleineres Residuum gewinnt
        both = mask_R_ok & mask_L_ok
        mask_R_ok[both] = res_R[both] < res_L[both]
        mask_L_ok[both] = ~mask_R_ok[both]
        log(f"  phi_b Filter: Stufe {tol_frac:.2f} — {np.degrees(tol_all):.2f}° = {tol_frac:.0%} × Teilung")

    tol_val = tooth_period * max(tol_frac, 0.35)   # für Return-Dict
    tol_R = tol_val
    tol_L = tol_val

    log(f"  Nach Filter: R={mask_R_ok.sum():,}  L={mask_L_ok.sum():,}  "
        f"Ratio={mask_R_ok.sum()/max(mask_L_ok.sum(),1):.2f}", t0)

    # Symmetrie-Diagnose pro Zahn
    asym_count = 0
    for ti in range(n_teeth):
        n_R = int((mask_R_ok & (tooth_R == ti)).sum())
        n_L = int((mask_L_ok & (tooth_L == ti)).sum())
        ratio = n_R / max(n_L, 1)
        if ratio > 2.5 or ratio < 0.4:
            log(f"  [⚠] Zahn {ti:2d}: R={n_R:5d} L={n_L:5d} ratio={ratio:.2f}")
            asym_count += 1
    if asym_count == 0:
        log(f"  ✓ Alle Zähne symmetrisch (R≈L)")

    # Zahnzentren-Winkel + einheitlicher Zahn-Index für alle Bandpunkte
    th_corr_all = pts_c[:, 1] - k_star * pts_c[:, 2]
    n_bins  = n_teeth * 40
    hist, _ = np.histogram(th_corr_all % (2*np.pi),
                             bins=n_bins, range=(0, 2*np.pi))
    fft_h    = np.fft.rfft(hist)
    phase_tc = np.angle(fft_h[n_teeth])
    off_tc   = (-phase_tc / n_teeth) % tooth_period
    tooth_angles = np.sort((np.arange(n_teeth) * tooth_period + off_tc)
                            % (2*np.pi))

    tc_all   = th_corr_all % (2*np.pi)
    tooth_all = (np.round(
        ((tc_all - off_tc + 4*np.pi) % (2*np.pi)) / tooth_period
    ).astype(int) % n_teeth)

    return dict(
        tooth_R=tooth_R, tooth_L=tooth_L,
        tooth_all=tooth_all,
        res_R=res_R,     res_L=res_L,
        mask_R_ok=mask_R_ok, mask_L_ok=mask_L_ok,
        tol_R=tol_R,     tol_L=tol_L,
        phi_b_R=phi_b_R, phi_b_L=phi_b_L,
        tooth_angles=tooth_angles,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 (DBSCAN) — Flanken-Segmentierung via Dichte in (φ_b, z)
# ═══════════════════════════════════════════════════════════════════════════════

def phase5_dbscan_flank(s1, s2, s3, s4, cfg):
    """
    Phase 5 (DBSCAN): Zahnflanken-Segmentierung in (φ_b, z)-Raum.

    Ersetzt den phi_b_tol_frac-Schwellwert durch dichtebasierte Ausreißer-
    Erkennung. Tip/Fuß/Fase-Punkte landen automatisch im Rauschen (Label -1) —
    kein manueller Toleranzwert nötig.

    Ablauf:
      1. tc_rel-Vorfilter (±45% Teilung, r≥r_b): trennt R/L-Kandidaten ohne
         phi_b — 0% Seiten-Inkonsistenz by construction.
      2. φ_b = (tc ± inv(α_t)) berechnen (k_star, r_b aus Phase 4 — kein MY_GEAR).
      3. DBSCAN in normiertem Raum (φ_b mod Teilung / Teilung, z / z_max):
           - Involuten-Punkte: φ_b ≈ const → dichter Streifen → Cluster
           - Tip/Fuß/Fase:    φ_b streut weit → Rauschen (Label -1)
         Jede valide Flanke (alle 83 Zähne einer Seite) bildet in diesem Raum
         EINEN dichten Streifen, da φ_b mod Teilung für alle Zähne gleich ist.
         DBSCAN findet diesen Streifen + weist Ausreißer automatisch als Noise aus.
      4. Zahnindex: round(tc / tooth_period) — exakt, kein Schwellwert.
      5. Zahnzentrum-Symmetrisierung: die finale tc_rel=0-Referenz
         (tooth_angles/off_tc_sym) wird NICHT aus der rohen, R/L-Punktzahl-
         gewichteten Dichte aller Bandpunkte bestimmt (das zieht die Mitte
         Richtung R, da R:L ≈ 4:1 durch Scanner-Geometrie), sondern aus dem
         zirkulären Mittelwert der beiden unabhängigen R-/L-DBSCAN-Peaks
         (peak_R, peak_L) — symmetrisch per Konstruktion. Betrifft nur die
         tc_rel-Diagnostik/Visualisierung (fig2/fig3); R/L-Punktmasken und
         DIN-Kennwerte sind unabhängig davon bereits korrekt.

    Ausgabe kompatibel mit phase5_tooth_assignment() für Phase 5b/6/7.
    Phase 5c ist in DBSCAN-Modus nicht nötig (Ausreißer bereits entfernt).
    """
    # sklearn DBSCAN entfernt — 1D φ_b-Peak-Finder (O(N), kein RAM-Problem)

    t0 = time.time()
    log("Phase 5 (DBSCAN): Zahnflanken-Segmentierung in (φ_b, z)")

    pts_c  = s3["pts_b_cyl"].astype(np.float64)
    r      = pts_c[:, 0]
    theta  = pts_c[:, 1]
    z      = pts_c[:, 2]

    k_star       = s4["k_star"]
    r_b          = s4["r_b"]
    n_teeth      = s2["n_teeth"]
    tooth_period = s2["tooth_period"]

    # ── 1. φ_b berechnen ─────────────────────────────────────────────────────
    r_safe  = np.maximum(r, r_b * 1.0005)
    alpha_t = np.arccos(np.clip(r_b / r_safe, 0.0, 1.0))
    inv_at  = np.tan(alpha_t) - alpha_t
    tc      = theta - k_star * z

    phi_b_R = (tc + inv_at) % (2.0 * np.pi)
    phi_b_L = (tc - inv_at) % (2.0 * np.pi)

    # ── 2. Zahnzentren + tc_rel ───────────────────────────────────────────────
    tooth_all, tooth_angles, tc_rel, off_tc, tc_mod = compute_tooth_index(
        theta, z, k_star, n_teeth, tooth_period)

    # ── 3. plus_is_R: welche tc_rel-Seite ist R? (wie Phase 5 phi_b) ─────────
    above_rb = r >= r_b
    def _cv(mask_sub, phi_b_arr):
        v = (phi_b_arr[mask_sub]) % tooth_period
        if len(v) < 20:
            return 1.0
        return 1.0 - float(np.abs(np.mean(np.exp(1j * v / tooth_period * 2.0 * np.pi))))

    _mask_plus  = above_rb & (tc_rel > 0)
    _mask_minus = above_rb & (tc_rel < 0)
    cv_plus  = _cv(_mask_plus,  phi_b_R)
    cv_minus = _cv(_mask_minus, phi_b_R)
    plus_is_R = cv_plus < cv_minus
    log(f"  Flanken-Konvention: tc_rel>0 ist {'R' if plus_is_R else 'L'}  "
        f"[φ_b-Schärfe: plus={1-cv_plus:.4f}  minus={1-cv_minus:.4f}]")

    # ── 4. tc_rel-Vorfilter: breite R/L-Kandidaten-Selektion ─────────────────
    pre_frac = float(cfg.get("dbscan_tc_pre_frac", 0.45))
    in_band  = above_rb & (np.abs(tc_rel) < tooth_period * pre_frac)

    if plus_is_R:
        r_cand = in_band & (tc_rel > 0)
        l_cand = in_band & (tc_rel < 0)
    else:
        r_cand = in_band & (tc_rel < 0)
        l_cand = in_band & (tc_rel > 0)

    log(f"  Vorfilter (±{pre_frac:.0%} Teilung, r≥r_b): "
        f"R={r_cand.sum():,}  L={l_cand.sum():,}")

    # ── 5. DBSCAN in (φ_b mod Teilung / Teilung,  z / z_max) ────────────────
    # Normierung: beide Achsen → [0, 1]-artig.
    # φ_b mod Teilung / Teilung: Involuten-Punkte → enger Streifen nahe 0 (oder 1).
    # z / z_max: Punkte über die gesamte Zahnbreite → 0..1.
    # Damit ist ε = 0.05 = 5% Teilung in φ_b / 5% Zahnbreite in z — skalenunabhängig.
    eps     = float(cfg.get("dbscan_eps",     0.05))
    min_pts = int(  cfg.get("dbscan_min_pts", 30))
    z_max   = float(z.max()) if z.max() > 0 else 1.0

    def _run_dbscan(cand_mask, phi_b_arr, side_label):
        """1D φ_b-Peak-Finder — ersetzt DBSCAN in (φ_b, z).

        Alle Flanken-Punkte derselben Seite landen (unabhängig von z) beim
        gleichen φ_b = tc ± inv(α_t) mod Teilung.  Die z-Dimension hat keine
        Diskriminierungskraft → Histogramm-Peak in 1D reicht aus.  O(N),
        kein sklearn, kein Speicherproblem.

        Gibt zusätzlich zur Inlier-Maske den gefundenen Peak IM ORIGINAL-
        (nicht rezentrierten) φ_b-mod-Teilung-Raum zurück — wird von der
        Zahnzentrum-Symmetrisierung unten benötigt (peak_absolute=None wenn
        keine Kandidaten vorhanden).
        """
        idx = np.where(cand_mask)[0]
        if len(idx) == 0:
            log(f"  φ_b-Peak {side_label}: keine Kandidaten")
            return np.zeros(len(theta), dtype=bool), None

        phi_b_mod = phi_b_arr[idx] % tooth_period

        # Zirkuläres Zentrieren: Cluster-Schwerpunkt auf 0.5 schieben, damit
        # der Modulo-Rand (bei 0/1) nicht mitten durch den Cluster läuft.
        _a    = phi_b_mod / tooth_period * (2.0 * np.pi)
        _ctr  = np.arctan2(np.sin(_a).mean(), np.cos(_a).mean())
        _shift = (0.5 - _ctr / (2.0 * np.pi)) * tooth_period
        phi_b_ctr  = (phi_b_mod + _shift) % tooth_period
        phi_b_norm = phi_b_ctr / tooth_period   # [0, 1], Involuten-Cluster ≈ 0.5

        # Histogramm-Peak: ~10 Bins pro ε-Breite, geglättet über ε-Fenster.
        n_hist   = max(200, int(round(1.0 / eps * 10)))
        hist, edges = np.histogram(phi_b_norm, bins=n_hist, range=(0.0, 1.0))
        kw       = max(1, int(round(eps * n_hist)))
        hist_sm  = np.convolve(hist.astype(float), np.ones(kw) / kw, mode='same')
        peak_bin = int(np.argmax(hist_sm))
        peak_phi = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0

        inlier_mask = np.abs(phi_b_norm - peak_phi) < eps
        n_in    = int(inlier_mask.sum())
        n_noise = len(idx) - n_in
        log(f"  φ_b-Peak {side_label}: {len(idx):,} Kandidaten  "
            f"Peak={peak_phi:.4f}  Inlier={n_in:,}  "
            f"Rauschen={n_noise:,} ({n_noise / len(idx) * 100:.1f}%)")
        if n_in < min_pts:
            log(f"  ⚠ {side_label}: Peak-Dichte zu gering (<{min_pts}) — eps anpassen")

        inlier = np.zeros(len(theta), dtype=bool)
        inlier[idx[inlier_mask]] = True

        # Peak zurück in den ORIGINALEN (nicht rezentrierten) φ_b-mod-Teilung-
        # Raum umrechnen — das ist dieselbe physikalische Größe wie off_tc.
        peak_absolute = float((peak_phi * tooth_period - _shift) % tooth_period)
        return inlier, peak_absolute

    mask_R_ok, peak_R = _run_dbscan(r_cand, phi_b_R, "R")
    mask_L_ok, peak_L = _run_dbscan(l_cand, phi_b_L, "L")

    ratio = mask_R_ok.sum() / max(mask_L_ok.sum(), 1)
    log(f"  Ergebnis Runde 1 (vorlaeufig, unsymmetrisches off_tc — nur zur "
        f"Peak-Bestimmung): R={mask_R_ok.sum():,}  L={mask_L_ok.sum():,}  "
        f"Ratio={ratio:.2f}", t0)

    # ── Zahnzentrum symmetrisieren: R- und L-Flanke sollten am selben
    # physikalischen φ_b,ref (mod Teilung) liegen (beide sind tc ± inv(α_t),
    # was am wahren Zahnzentrum für BEIDE Flanken denselben Wert ergibt).
    # Das ursprüngliche off_tc oben wurde aus der ROHEN, R/L-Punktzahl-
    # gewichteten Dichte aller Bandpunkte (R+L+Rauschen) berechnet — bei
    # R:L ≈ 4:1 (Scanner-Geometrie) zieht das die "Zahnmitte" systematisch
    # Richtung R, was genau die beobachtete Asymmetrie (R nahe tc_rel=0,
    # L um ≈1° verschoben) erklärt. off_tc wird daher hier durch den
    # zirkulären Mittelwert der beiden UNABHÄNGIGEN, bereits pro Seite
    # gefundenen DBSCAN-Peaks (peak_R, peak_L) ersetzt — symmetrisch per
    # Konstruktion, unabhängig vom R/L-Punktzahlverhältnis. Nur für
    # tc_rel-Diagnostik/Visualisierung relevant (fig2/fig3) — die R/L-
    # Punktmasken oben und die DIN-Kennwerte (Phase 4-UV zentriert pro
    # Cluster selbst über den lokalen Median von phi_b_ist) sind davon
    # unberührt.
    if peak_R is not None and peak_L is not None:
        ang_R  = peak_R / tooth_period * 2.0 * np.pi
        ang_L  = peak_L / tooth_period * 2.0 * np.pi
        ang_avg = np.arctan2(np.sin(ang_R) + np.sin(ang_L),
                              np.cos(ang_R) + np.cos(ang_L))
        off_tc_sym = float((ang_avg / (2.0 * np.pi)) * tooth_period % tooth_period)
        delta = ((peak_R - peak_L + tooth_period / 2.0) % tooth_period) - tooth_period / 2.0
        log(f"  Zahnzentrum-Symmetrisierung: R-Peak={np.degrees(peak_R):.3f}°  "
            f"L-Peak={np.degrees(peak_L):.3f}°  Δ(R−L)={np.degrees(delta):.3f}°  "
            f"off_tc: dichte-basiert={np.degrees(off_tc):.3f}° → "
            f"symmetrisiert={np.degrees(off_tc_sym):.3f}°")
    else:
        off_tc_sym = off_tc
        log("  ⚠ Zahnzentrum-Symmetrisierung übersprungen (eine Seite ohne "
            "Kandidaten) — nutze dichte-basiertes off_tc")

    tooth_all, tooth_angles, _, _, _ = compute_tooth_index(
        theta, z, k_star, n_teeth, tooth_period, off_tc=off_tc_sym)

    # ── 6. Runde 2: Vorfilter + φ_b-Peak-Suche mit symmetrisiertem off_tc_sym ──
    # Runde 1 (Schritte 4-5 oben) baute den tc_rel-Vorfilter aus dem ROHEN,
    # R/L-Punktzahl-gewichteten off_tc (R:L≈4:1 durch Scanner-Geometrie zieht
    # das Zentrum Richtung R). Da tc_rel mit dem Radius waechst (grob prop.
    # zu inv(alpha_t(r))), verschiebt ein falsch zentriertes ±pre_frac-Fenster
    # NICHT nur eine Visualisierung, sondern die tatsaechliche Kandidaten-
    # Auswahl: R's Fenster wird effektiv zu ENG (schneidet echte Spitzen-nahe
    # R-Punkte ab), L's Fenster wird zu BREIT (nimmt zusaetzliches Fase-/Fuss-
    # Rauschen jenseits der echten Flanke mit auf). Bestaetigt per Region-
    # Growing-Vergleich (2026-07-14): unter einer Methode ohne jede k_star-
    # Abwicklung verschwindet der beobachtete 1-2mm R-vs-L-Radiusunterschied
    # fast vollstaendig. Runde 2 wiederholt Vorfilter + Peak-Suche daher ein
    # zweites Mal, diesmal zentriert auf das bereits symmetrisierte off_tc_sym
    # (aus den zwei unabhaengigen Runde-1-Seiten-Peaks) — das ist die
    # tatsaechliche, finale Segmentierung; Runde 1 diente nur dazu, off_tc_sym
    # ueberhaupt erst zu bestimmen (Henne-Ei-Problem: der korrekte Mittelpunkt
    # ist erst nach einer ersten, groben Kandidatenrunde bekannt).
    tc_rel_sym = ((tc_mod - tooth_angles[tooth_all] + np.pi) % (2.0 * np.pi)) - np.pi

    # WICHTIG (Bugfix nach fehlgeschlagenem ersten Versuch, siehe Log
    # 2026-07-14): plus_is_R wurde oben (Schritt 3) aus dem ROHEN tc_rel
    # (referenziert auf off_tc) bestimmt. off_tc_sym unterscheidet sich davon
    # typischerweise um einen betraechtlichen Teil der Vorfilter-Fensterbreite
    # (hier z.B. 1.386° von 1.952° Halbbreite = 71%!) -- ein Punkt mit
    # tc_rel≈+1.0° (unter altem off_tc klar R) kann unter dem verschobenen
    # tc_rel_sym negativ werden. Wuerde man das ALTE plus_is_R hier blind
    # weiterverwenden, kippt genau dieser Punkt faelschlich nach L -- das war
    # der Bug im ersten Versuch (R:L Verhaeltnis kippte von 4.28 auf 0.16,
    # Seiten-Konvention selbst drehte sich um). Fix: plus_is_R MUSS in JEDER
    # tc_rel-Referenz neu bestimmt werden (derselbe phi_b-Schaerfe-Test wie
    # oben, nur diesmal auf tc_rel_sym statt tc_rel) -- die Aussage "plus
    # bedeutet R" ist keine feste Tatsache, sondern haengt vom verwendeten
    # Nullpunkt ab.
    _mask_plus_sym  = above_rb & (tc_rel_sym > 0)
    _mask_minus_sym = above_rb & (tc_rel_sym < 0)
    cv_plus_sym  = _cv(_mask_plus_sym,  phi_b_R)
    cv_minus_sym = _cv(_mask_minus_sym, phi_b_R)
    plus_is_R_sym = cv_plus_sym < cv_minus_sym
    log(f"  Flanken-Konvention Runde 2: tc_rel_sym>0 ist "
        f"{'R' if plus_is_R_sym else 'L'}  "
        f"[φ_b-Schärfe: plus={1 - cv_plus_sym:.4f}  minus={1 - cv_minus_sym:.4f}]")

    in_band_sym = above_rb & (np.abs(tc_rel_sym) < tooth_period * pre_frac)
    if plus_is_R_sym:
        r_cand_sym = in_band_sym & (tc_rel_sym > 0)
        l_cand_sym = in_band_sym & (tc_rel_sym < 0)
    else:
        r_cand_sym = in_band_sym & (tc_rel_sym < 0)
        l_cand_sym = in_band_sym & (tc_rel_sym > 0)
    log(f"  Vorfilter Runde 2 (symmetrisiert, ±{pre_frac:.0%} Teilung, r≥r_b): "
        f"R={r_cand_sym.sum():,}  L={l_cand_sym.sum():,}")

    mask_R_ok, _peak_R2 = _run_dbscan(r_cand_sym, phi_b_R, "R (Runde 2)")
    mask_L_ok, _peak_L2 = _run_dbscan(l_cand_sym, phi_b_L, "L (Runde 2)")

    ratio2 = mask_R_ok.sum() / max(mask_L_ok.sum(), 1)
    log(f"  Ergebnis (final, Runde 2): R={mask_R_ok.sum():,}  L={mask_L_ok.sum():,}  "
        f"Ratio={ratio2:.2f}")

    # Sanity-Check: Runde 2 sollte die R:L-Punktzahlrelation NICHT umkehren
    # (welche Seite mehr Punkte hat ist Scanner-Geometrie, kein Artefakt der
    # Fenster-Zentrierung — per Region-Growing-Vergleich bestaetigt ~4:1,
    # 2026-07-14). Kippt das Verhaeltnis (>1 vs <1) zwischen Runde 1 und
    # Runde 2, oder aendert es sich um mehr als 2x, deutet das auf denselben
    # Referenzrahmen-Bug hin, der im ersten Versuch dieses Fixes auftrat
    # (plus_is_R faelschlich wiederverwendet) -- dann NICHT weiterverwenden,
    # sondern melden.
    if (ratio > 1.0) != (ratio2 > 1.0) or ratio2 > ratio * 2.0 or ratio2 < ratio / 2.0:
        log(f"  ⚠⚠ VERDACHT auf Referenzrahmen-Fehler: Runde-1-Ratio={ratio:.2f} vs "
            f"Runde-2-Ratio={ratio2:.2f} weichen zu stark ab (R:L-Verhaeltnis sollte "
            f"durch die Symmetrisierung nicht kippen) — Ergebnis vor Verwendung pruefen!")

    # Symmetrie-Diagnose (vektorisiert via bincount statt Python-Schleife)
    r_cnt    = np.bincount(tooth_all[mask_R_ok], minlength=n_teeth)
    l_cnt    = np.bincount(tooth_all[mask_L_ok], minlength=n_teeth)
    ratios   = r_cnt / np.maximum(l_cnt, 1).astype(float)
    asym_mask = (ratios > 2.5) | (ratios < 0.4)
    for ti in np.where(asym_mask)[0]:
        log(f"  [⚠] Zahn {ti:2d}: R={r_cnt[ti]:5d} L={l_cnt[ti]:5d} "
            f"ratio={ratios[ti]:.2f}")
    if not asym_mask.any():
        log("  ✓ Alle Zähne symmetrisch (R≈L)")

    tol = tooth_period * float(cfg.get("phi_b_tol_frac", 0.10))

    return dict(
        tooth_R=tooth_all.copy(),
        tooth_L=tooth_all.copy(),
        tooth_all=tooth_all,
        res_R=np.zeros(0, dtype=np.float32),
        res_L=np.zeros(0, dtype=np.float32),
        mask_R_ok=mask_R_ok,
        mask_L_ok=mask_L_ok,
        tol_R=tol, tol_L=tol,
        phi_b_R=phi_b_R % tooth_period,
        phi_b_L=phi_b_L % tooth_period,
        tooth_angles=tooth_angles,
        plus_is_R=plus_is_R,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 (Alternativ) — Region Growing Segmentierung
# ═══════════════════════════════════════════════════════════════════════════════

def _voxel_downsample_fast(pts_xyz, voxel_size):
    """Ein Punkt pro Voxel (erster im Array). Gibt Indizes zurück."""
    vk = np.floor(pts_xyz / voxel_size).astype(np.int64)
    _, first_idx = np.unique(vk, axis=0, return_index=True)
    return np.sort(first_idx)


def _seed_normals_camera_oriented(seed_pts, center, k_nn=30):
    """
    PCA-Normalen, zur virtuellen Kamera [cx, cy, cz+1000mm] orientiert.

    Entspricht open3d.estimate_normals() + orient_normals_towards_camera_location().
    Die Kamera weit über dem Zahnrad (view_dir ≈ +z) stellt sicher, dass alle
    Normalen auf DERSELBEN Seite zeigen — kein zufälliger ±-Flip mehr.
    Das behebt das 337-statt-167-Cluster-Problem: PCA ohne Orientierung erzeugt
    zufällige Vorzeichenwechsel, die den BFS an Vorzeichen-Grenzen aufspalten.
    """
    from scipy.spatial import cKDTree
    n = len(seed_pts)
    tree = cKDTree(seed_pts)
    _, idx_nn = tree.query(seed_pts, k=k_nn, workers=-1)

    normals = np.zeros((n, 3), dtype=np.float64)
    batch = 10_000
    for s in range(0, n, batch):
        e = min(s + batch, n)
        nb  = seed_pts[idx_nn[s:e]]
        ce  = nb - nb.mean(axis=1, keepdims=True)
        cov = np.einsum('nki,nkj->nij', ce, ce)
        _, evec = np.linalg.eigh(cov)
        normals[s:e] = evec[:, :, 0]   # kleinster Eigenwert → Oberflächen-Normale

    # Vorzeichen konsistent zur Kamera ausrichten (wie open3d-Notebook)
    camera   = np.array([center[0], center[1], center[2] + 1000.0])
    view_dir = camera[np.newaxis, :] - seed_pts   # (N, 3), stark +z-dominiert
    flip     = np.einsum('ij,ij->i', normals, view_dir) < 0
    normals[flip] *= -1
    return normals


def _region_growing_bfs(seed_pts, seed_nrm, dist_thresh, angle_thresh_deg, min_size):
    """
    Region Growing BFS mit vorberechneten Nachbarlisten.

    Vorteil gegenüber Query-pro-Punkt: KDTree wird nur einmal gebaut,
    alle Nachbarn einmalig abgefragt → schneller als Notebook-Variante.
    Winkel-Check ist vektorisiert pro Batch.

    Gibt (labels, n_clusters) zurück.
    """
    from scipy.spatial import cKDTree
    from collections import deque

    cos_thresh = np.cos(np.radians(angle_thresh_deg))
    n = len(seed_pts)

    log(f"    KDTree-Build ({n:,} Seeds)...")
    tree = cKDTree(seed_pts)

    log(f"    Nachbar-Vorberechnung (r≤{dist_thresh}mm, parallel)...")
    nbrs = tree.query_ball_point(seed_pts, r=dist_thresh, workers=-1)

    visited = np.zeros(n, dtype=bool)
    labels  = np.full(n, -1, dtype=int)
    cluster_id = 0

    log(f"    BFS Segmentierung...")
    for seed in range(n):
        if visited[seed]:
            continue

        region = [seed]
        queue  = deque([seed])
        visited[seed] = True

        while queue:
            curr = queue.popleft()
            nb_list = nbrs[curr]
            if not nb_list:
                continue

            nb_arr = np.array(nb_list, dtype=np.int32)
            unvisited_mask = ~visited[nb_arr]
            if not unvisited_mask.any():
                continue

            cands = nb_arr[unvisited_mask]
            # Vektorisierter Winkel-Check: |n_curr · n_cand|
            cos_a = np.abs(seed_nrm[cands] @ seed_nrm[curr])
            passing = cands[cos_a >= cos_thresh]

            visited[passing] = True
            region.extend(passing.tolist())
            queue.extend(passing.tolist())

        if len(region) >= min_size:
            for idx in region:
                labels[idx] = cluster_id
            cluster_id += 1

    return labels, cluster_id


def _merge_to_n_clusters(seed_pts, seed_labels, n_found, n_target):
    """
    Mergt iterativ den kleinsten Cluster in seinen nächsten Nachbar-Cluster,
    bis genau n_target Cluster verbleiben. Anschließend Umnummerierung 0..n_target-1.
    """
    lbl  = seed_labels.copy()
    live = list(range(n_found))

    for _ in range(n_found - n_target):
        # Schwerpunkte und Größen der aktiven Cluster
        centroids = {}
        sizes = {}
        for c in live:
            m = lbl == c
            if m.sum() == 0:
                live.remove(c)
                break
            centroids[c] = seed_pts[m].mean(axis=0)
            sizes[c] = int(m.sum())
        else:
            # Kleinsten Cluster finden
            smallest = min(sizes, key=lambda c: sizes[c])

            # Nächsten Nachbar-Cluster (Schwerpunkt-Abstand)
            best_d = np.inf
            merge_into = None
            sc = centroids[smallest]
            for c in live:
                if c == smallest:
                    continue
                d = np.linalg.norm(sc - centroids[c])
                if d < best_d:
                    best_d = d
                    merge_into = c

            if merge_into is None:
                break

            lbl[lbl == smallest] = merge_into
            live.remove(smallest)
            continue
        break

    # Umnummerieren 0..n_verbleibend-1
    live_sorted = sorted(set(live) & set(np.unique(lbl[lbl >= 0])))
    remap = np.full(max(live_sorted) + 1 if live_sorted else 1, -1, dtype=int)
    for new_id, old_id in enumerate(live_sorted):
        remap[old_id] = new_id

    final = np.full_like(lbl, -1)
    valid = lbl >= 0
    final[valid] = remap[lbl[valid]]
    return final


def phase5_region_growing(s1, s2, s3, s4, cfg):
    """
    Phase 5 (Region Growing): geometrische Zahnflanken-Segmentierung.

    Ersetzt phi_b-Zuordnung durch Normal-basiertes Region Growing.
    Erzwingt exakt 2 × n_teeth = 166 Cluster (für N=83).

    Ablauf:
      1. Voxel-Downsampling → ~2% Seeds
      2. Region Growing BFS auf Seeds (Winkel + Abstand)
      3. Merge auf genau n_target Cluster
      4. Alle Bandpunkte → k-nächste Seeds, Score = d×(2-|cos|)
      5. L/R-Klassifikation via tc_rel-Median + gm_product-Konvention
      6. Zahnindex via tc (identisch zu Phase 5b)

    Ausgabe kompatibel mit phase5_tooth_assignment() für Phase 5b/5c/6/7.
    """
    from scipy.spatial import cKDTree
    t0 = time.time()
    log("Phase 5 (Region Growing): Zahnflanken-Segmentierung")

    # ── Eingabedaten ─────────────────────────────────────────────────────────
    # Seed-Cloud: ALLE Scan-Punkte (wie Notebook), nicht nur Band.
    # Vorteil: keine Lücken an Scan-Rändern, BFS findet ~167 statt 337 Cluster.
    all_pts_xyz  = s1["pts_xyz"].astype(np.float64)   # volle Punktwolke (~4.75M)
    pts_xyz  = s3["pts_b_xyz"].astype(np.float64)     # Bandpunkte für Projektion
    pts_cyl  = s3["pts_b_cyl"].astype(np.float64)
    gm_prod  = s3["gm_product"].astype(np.float64)

    center       = s1["center"]
    axis_v       = s1["axis"]
    e1_v         = s1["e1"]
    e2_v         = s1["e2"]
    k_star       = s4["k_star"]
    n_teeth      = s2["n_teeth"]
    tooth_period = s2["tooth_period"]
    n_target     = 2 * n_teeth

    theta   = pts_cyl[:, 1]
    z_cyl   = pts_cyl[:, 2]
    r_band  = pts_cyl[:, 0]

    # ── 1. Voxel-Downsampling der GESAMTEN Punktwolke ───────────────────────
    voxel_size = float(cfg.get("rg_voxel_size", 0.5))
    seed_idx   = _voxel_downsample_fast(all_pts_xyz, voxel_size)
    seed_pts   = all_pts_xyz[seed_idx]
    log(f"  Downsampling: {len(all_pts_xyz):,} → {len(seed_pts):,} Seeds "
        f"({voxel_size}mm Voxel, gesamte Punktwolke)")

    # ── 1b. Kamera-orientierte PCA-Normalen (wie Notebook: knn=30) ───────────
    # orient_normals_towards_camera_location([cx, cy, cz+1000mm]):
    # Alle Normalen zeigen zur selben Seite → konsistentes Vorzeichen im BFS.
    # Ohne Orientierung: zufällige ±1-Flips → BFS zerhackt Flanken → 337 statt 167.
    k_seed_norm = int(cfg.get("rg_seed_knn", 30))
    log(f"  Kamera-orientierte Normalen (k={k_seed_norm})...")
    seed_nrm = _seed_normals_camera_oriented(seed_pts, center, k_nn=k_seed_norm)

    # ── 2. Region Growing BFS (identisch mit Notebook) ──────────────────────
    angle_deg   = float(cfg.get("rg_angle_deg",  5.0))
    dist_thresh = float(cfg.get("rg_dist_mm",    6.0))
    seed_min    = int(  cfg.get("rg_seed_min",   30))

    seed_labels, n_found = _region_growing_bfs(
        seed_pts, seed_nrm, dist_thresh, angle_deg, seed_min)

    log(f"  Region Growing: {n_found} Cluster gefunden  (Ziel: {n_target})")

    # ── 3. Cluster-Anzahl erzwingen ──────────────────────────────────────────
    if n_found > n_target:
        seed_labels = _merge_to_n_clusters(
            seed_pts, seed_labels, n_found, n_target)
        n_found = n_target
        log(f"  Merge → {n_found} Cluster")
    elif n_found < n_target:
        log(f"  ⚠ Nur {n_found} Cluster (< {n_target}) — "
            f"rg_angle_deg oder rg_seed_min reduzieren")

    # ── 4. Projektion (Notebook-Methode): Winkelsektor + Distanz-Score ───────
    # Für jede Flanke c: nur Bandpunkte innerhalb ±1.5×Teilung berücksichtigen,
    # dann Score = Abstand zum Flanken-KDTree. Beste Flanke gewinnt pro Punkt.
    # Vorteil: L/R-Trennung automatisch durch Sektorfilter, kein k=1-NN der ganzen Cloud.
    log(f"  Projektion {len(pts_xyz):,} Bandpunkte (Winkelsektor + Distanz)...")

    active_clusters = sorted(set(seed_labels[seed_labels >= 0]))

    # Mittlerer Winkel jeder Flanke (zirkulärer Mittelwert, wie Notebook)
    p_perp_s = seed_pts - center[np.newaxis, :]
    z_s      = p_perp_s @ axis_v
    p_rad_s  = p_perp_s - np.outer(z_s, axis_v)
    theta_s  = np.arctan2(p_rad_s @ e2_v, p_rad_s @ e1_v)

    cluster_theta = {}
    cluster_trees = {}
    cluster_mean_nrm = {}
    for c in active_clusters:
        m = seed_labels == c
        th = theta_s[m]
        cluster_theta[c]    = float(np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th))))
        cluster_trees[c]    = cKDTree(seed_pts[m])
        n_mn = seed_nrm[m].mean(axis=0)
        cluster_mean_nrm[c] = n_mn / (np.linalg.norm(n_mn) + 1e-12)

    th_pts      = theta                               # Winkel der Bandpunkte
    best_score  = np.full(len(pts_xyz), np.inf)
    band_labels = np.full(len(pts_xyz), -1, dtype=int)

    for c in active_clusters:
        cth  = cluster_theta[c]
        dth  = np.angle(np.exp(1j * (th_pts - cth)))   # Winkelabstand (−π..+π)
        cand = np.where(np.abs(dth) < 1.5 * tooth_period)[0]
        if len(cand) == 0:
            continue
        d, _ = cluster_trees[c].query(pts_xyz[cand], workers=-1)
        better = d < best_score[cand]
        idx = cand[better]
        best_score[idx] = d[better]
        band_labels[idx] = c

    assigned = int((band_labels >= 0).sum())
    log(f"  Zugeordnet: {assigned:,}/{len(pts_xyz):,} "
        f"({assigned/len(pts_xyz)*100:.1f}%)")

    # ── 5. tc-Basis: Zahnzentren + tc_rel ───────────────────────────────────
    tooth_all, tooth_angles, tc_rel, off_tc, tc = compute_tooth_index(
        theta, z_cyl, k_star, n_teeth, tooth_period)

    # ── 6. L/R-Klassifikation der Cluster ───────────────────────────────────
    n_active = n_found  # nach Merge = n_target (oder < n_target wenn zu wenig)
    cluster_tc_rel_med = np.zeros(n_active)
    cluster_gm_mean    = np.zeros(n_active)
    cluster_tooth_id   = np.zeros(n_active, dtype=int)

    for c in range(n_active):
        m = (band_labels == c)
        if m.sum() < 5:
            continue
        cluster_tc_rel_med[c] = float(np.median(tc_rel[m]))
        cluster_gm_mean[c]    = float(gm_prod[m].mean())
        # Zahnindex via Median-tc
        tc_med = float(np.median(tc[m]))
        cluster_tooth_id[c] = int(np.round(
            ((tc_med - off_tc + 4.0 * np.pi) % (2.0 * np.pi)) / tooth_period
        )) % n_teeth

    # Konvention: welches tc_rel-Vorzeichen ist R?
    # Gaußsche Karte: gm_product > 0 ↔ R-Flanke (aus Phase 3).
    # Stimme Mehrheit der Cluster ab.
    pos_tc_and_pos_gm = int(((cluster_tc_rel_med > 0) & (cluster_gm_mean > 0)).sum())
    pos_tc_and_neg_gm = int(((cluster_tc_rel_med > 0) & (cluster_gm_mean < 0)).sum())
    plus_is_R = pos_tc_and_pos_gm >= pos_tc_and_neg_gm

    cluster_is_R = (cluster_tc_rel_med > 0) if plus_is_R else (cluster_tc_rel_med < 0)

    n_R_cl = int(cluster_is_R.sum())
    n_L_cl = int((~cluster_is_R).sum())
    log(f"  L/R: {n_R_cl} R-Cluster, {n_L_cl} L-Cluster  "
        f"(tc_rel>0 ist {'R' if plus_is_R else 'L'})")

    # ── 7. Band-Masken ───────────────────────────────────────────────────────
    # L/R NICHT via Cluster-Median (anfällig für Merge-Artefakte bei
    # Zahnspitze/Zahnlücken-Subclustern), sondern direkt via tc_rel pro Punkt.
    # Region Growing liefert:
    #   • is_labeled  — räumliche Kohärenz; Ausreißer ohne Cluster → False
    #   • plus_is_R   — L/R-Konvention (Mehrheitsvote aller Cluster-Mediane)
    # tc_rel liefert:
    #   • exaktes L/R-Vorzeichen pro Punkt → 0% Seiten-Inkonsistenz by definition
    # in_flank:
    #   • |tc_rel| < 40% Teilung → Kopf/Fuß ausblenden (nur echte Flankenpunkte)
    is_labeled = band_labels >= 0
    in_flank   = np.abs(tc_rel) < tooth_period * 0.40

    if plus_is_R:
        mask_R_ok = is_labeled & in_flank & (tc_rel > 0)
        mask_L_ok = is_labeled & in_flank & (tc_rel < 0)
    else:
        mask_R_ok = is_labeled & in_flank & (tc_rel < 0)
        mask_L_ok = is_labeled & in_flank & (tc_rel > 0)

    ratio = mask_R_ok.sum() / max(mask_L_ok.sum(), 1)
    log(f"  Masken: R={mask_R_ok.sum():,}  L={mask_L_ok.sum():,}  "
        f"Ratio={ratio:.2f}  (tc_rel-direkt, kein Merge-Artefakt)", t0)

    # ── 8. Ausgabe-Arrays ────────────────────────────────────────────────────
    # tooth_R/tooth_L = tooth_all (tc-basiert, identisch zu Phase 5b)
    # → Phase 5b-Cross-Check zeigt 100% Agreement
    tooth_R = tooth_all.copy()
    tooth_L = tooth_all.copy()

    tol = tooth_period * float(cfg.get("phi_b_tol_frac", 0.10))

    # phi_b für Visualisierung
    r_safe  = np.maximum(r_band, s4["r_b"] * 1.0005)
    inv_at  = np.tan(np.arccos(np.clip(s4["r_b"] / r_safe, 0.0, 1.0))) \
              - np.arccos(np.clip(s4["r_b"] / r_safe, 0.0, 1.0))
    phi_b_R = (tc + inv_at) % tooth_period
    phi_b_L = (tc - inv_at) % tooth_period

    return dict(
        tooth_R=tooth_R,  tooth_L=tooth_L,
        tooth_all=tooth_all,
        res_R=np.zeros(0, dtype=np.float32),
        res_L=np.zeros(0, dtype=np.float32),
        mask_R_ok=mask_R_ok, mask_L_ok=mask_L_ok,
        tol_R=tol, tol_L=tol,
        phi_b_R=phi_b_R, phi_b_L=phi_b_L,
        tooth_angles=tooth_angles,
        # Region-Growing-spezifisch (für Diagnose)
        band_labels=band_labels,
        cluster_is_R=cluster_is_R,
        cluster_tooth_id=cluster_tooth_id,
        n_rg_clusters=n_active,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5 (Alternativ) — Region Growing mit L/R-Vortrennung (voll k-unabhängig)
# ═══════════════════════════════════════════════════════════════════════════════

def phase5_region_growing_lr_split(s1, s2, s3, s4, cfg):
    """
    Wie phase5_region_growing(), aber L/R wird VOR dem BFS entschieden statt
    danach — die Segmentierung selbst kommt vollständig ohne k_star aus.

    Problem bei phase5_region_growing():
      BFS läuft auf der gemischten Punktwolke (R+L zusammen). Die Trennung
      der beiden Flanken AN der Zahnkopf-Kante ist ein Nebeneffekt der
      Winkel-/Abstandsschwelle (rg_angle_deg, rg_dist_mm) — und genau dort
      (Kopf/Fuß) wird nach Phase-3-Diagnose auch gm_product ≈ 0 instabil.
      Beide Signale schwächen sich also an derselben Stelle gleichzeitig.

    Lösung hier:
      L/R kommt aus gm_product = n_r · n_tang (Phase 3, Zeile ~911) — einer
      rein lokalen, k-unabhängigen Eigenschaft jedes einzelnen Punktes
      (siehe Docstring phase3_normals_and_lr). Seeds UND Bandpunkte werden
      per gm_product-Vorzeichen in R/L getrennt, BEVOR irgendein Nachbar-
      schafts-Check läuft. Ein R- und ein L-Punkt können dadurch NIE im
      selben BFS-Lauf verbunden werden — unabhängig davon, wie nah sie an
      der Zahnkopf-Kante beieinander liegen oder wie ungenau k_star_ist ist.

      k_star_ist wird ausschließlich noch für die Zahn-INDEX-Zuordnung
      (welcher der n_teeth Zähne, 0..82) der bereits fertig getrennten
      Cluster verwendet — ein reiner Post-hoc-Labeling-Schritt. Ein Fehler
      in k_star_ist kann hier bestenfalls Zähne falsch nummerieren, aber
      keine Punkte zwischen Clustern verschieben.

    Ablauf:
      1. Voxel-Downsampling (alle Scan-Punkte) → Seeds
      2. Kamera-orientierte PCA-Normalen auf Seeds
      3. gm_product pro Seed (lokal, k-unabhängig) → R / L / unsicher
      4. BFS GETRENNT je Seite, je Ziel n_teeth Cluster
      5. Bandpunkte → nächster GLEICHSEITIGER Cluster-KDTree
         (gm_product der Bandpunkte kommt direkt aus Phase 3 — dieselben
         Punkte, s3["pts_b_xyz"]); unsichere Bandpunkte (|gm|<gm_eps)
         bleiben bewusst unzugeordnet statt zu einer Seite gezwungen.
      6. Zahnindex via tc = theta - k_star·z — NUR Labeling, keine Trennung

    Ausgabe kompatibel mit phase5_tooth_assignment() für Phase 5b/6/7.
    """
    from scipy.spatial import cKDTree
    t0 = time.time()
    log("Phase 5 (Region Growing, L/R-getrennt): Zahnflanken-Segmentierung")

    all_pts_xyz  = s1["pts_xyz"].astype(np.float64)
    pts_xyz      = s3["pts_b_xyz"].astype(np.float64)
    pts_cyl      = s3["pts_b_cyl"].astype(np.float64)
    gm_prod_band = s3["gm_product"].astype(np.float64)
    curv_band    = s3["curvatures"].astype(np.float64)

    center       = s1["center"]
    axis_v       = s1["axis"]
    e1_v         = s1["e1"]
    e2_v         = s1["e2"]
    k_star       = s4["k_star"]
    n_teeth      = s2["n_teeth"]
    tooth_period = s2["tooth_period"]
    eps_gm       = float(cfg.get("gm_eps", 0.012))

    theta   = pts_cyl[:, 1]
    z_cyl   = pts_cyl[:, 2]
    r_band  = pts_cyl[:, 0]

    # ── 1. Voxel-Downsampling der GESAMTEN Punktwolke ───────────────────────
    voxel_size = float(cfg.get("rg_voxel_size", 0.5))
    seed_idx   = _voxel_downsample_fast(all_pts_xyz, voxel_size)
    seed_pts   = all_pts_xyz[seed_idx]
    log(f"  Downsampling: {len(all_pts_xyz):,} -> {len(seed_pts):,} Seeds "
        f"({voxel_size}mm Voxel, gesamte Punktwolke)")

    # ── 2. Kamera-orientierte PCA-Normalen (wie phase5_region_growing) ──────
    k_seed_norm = int(cfg.get("rg_seed_knn", 30))
    log(f"  Kamera-orientierte Normalen (k={k_seed_norm})...")
    seed_nrm = _seed_normals_camera_oriented(seed_pts, center, k_nn=k_seed_norm)

    # ── 3. gm_product pro Seed — rein lokal, k-unabhängig ───────────────────
    p_perp_s  = seed_pts - center[np.newaxis, :]
    z_s       = p_perp_s @ axis_v
    p_rad_s   = p_perp_s - np.outer(z_s, axis_v)
    r_s       = np.linalg.norm(p_rad_s, axis=1)
    r_hat_s   = p_rad_s / np.maximum(r_s[:, None], 1e-9)
    e_theta_s = np.cross(axis_v[None, :], r_hat_s)

    n_r_seed    = np.einsum("ij,ij->i", seed_nrm, r_hat_s)
    n_tang_seed = np.einsum("ij,ij->i", seed_nrm, e_theta_s)
    gm_seed     = n_r_seed * n_tang_seed   # flip-invariant, siehe Phase 3

    seed_side = np.zeros(len(seed_pts), dtype=np.int8)   # +1=R, -1=L, 0=unsicher
    seed_side[gm_seed > eps_gm]  = 1
    seed_side[gm_seed < -eps_gm] = -1
    log(f"  Seed-Klassifikation (gm_product, k-unabhängig): "
        f"R={int((seed_side>0).sum()):,}  L={int((seed_side<0).sum()):,}  "
        f"unsicher={int((seed_side==0).sum()):,} (kein BFS-Start)")

    # ── 4. BFS GETRENNT je Seite ─────────────────────────────────────────────
    angle_deg   = float(cfg.get("rg_angle_deg",  5.0))
    dist_thresh = float(cfg.get("rg_dist_mm",    6.0))
    seed_min    = int(  cfg.get("rg_seed_min",   30))

    seed_labels      = np.full(len(seed_pts), -1, dtype=int)
    cluster_is_R_lst = []
    n_offset = 0

    for side_val, side_name in [(1, "R"), (-1, "L")]:
        side_idx = np.where(seed_side == side_val)[0]
        if len(side_idx) == 0:
            log(f"  [Warnung] Keine {side_name}-Seeds gefunden")
            continue

        log(f"  BFS {side_name}-Seite ({len(side_idx):,} Seeds)...")
        sub_labels, n_found = _region_growing_bfs(
            seed_pts[side_idx], seed_nrm[side_idx], dist_thresh, angle_deg, seed_min)
        log(f"  {side_name}: {n_found} Cluster gefunden (Ziel: {n_teeth})")

        if n_found > n_teeth:
            sub_labels = _merge_to_n_clusters(seed_pts[side_idx], sub_labels, n_found, n_teeth)
            n_found = n_teeth
            log(f"  {side_name}: Merge -> {n_found} Cluster")
        elif n_found < n_teeth:
            log(f"  [Warnung] {side_name}: nur {n_found} Cluster (< {n_teeth}) -- "
                f"rg_angle_deg oder rg_seed_min reduzieren")

        valid = sub_labels >= 0
        seed_labels[side_idx[valid]] = sub_labels[valid] + n_offset
        cluster_is_R_lst.extend([side_val > 0] * n_found)
        n_offset += n_found

    n_active         = n_offset
    cluster_is_R_arr = np.array(cluster_is_R_lst, dtype=bool)
    active_clusters  = list(range(n_active))
    log(f"  Gesamt: {n_active} Cluster "
        f"({int(cluster_is_R_arr.sum())} R + {int((~cluster_is_R_arr).sum())} L)")

    # ── 5. Projektion: Bandpunkte → nächster GLEICHSEITIGER Cluster ─────────
    p_perp_s_all = seed_pts - center[np.newaxis, :]
    z_s_all      = p_perp_s_all @ axis_v
    p_rad_s_all  = p_perp_s_all - np.outer(z_s_all, axis_v)
    theta_s_all  = np.arctan2(p_rad_s_all @ e2_v, p_rad_s_all @ e1_v)

    cluster_theta = {}
    cluster_trees = {}
    for c in active_clusters:
        m = seed_labels == c
        if not m.any():
            continue
        th = theta_s_all[m]
        cluster_theta[c] = float(np.arctan2(np.mean(np.sin(th)), np.mean(np.cos(th))))
        cluster_trees[c] = cKDTree(seed_pts[m])

    # Bandpunkt-Seite: identische gm_product-Klassifikation, direkt aus Phase 3
    # (dieselben Punkte wie s3["pts_b_xyz"]) — kein erneutes Normalen-Fitting nötig.
    band_side = np.zeros(len(pts_xyz), dtype=np.int8)
    band_side[gm_prod_band > eps_gm]  = 1
    band_side[gm_prod_band < -eps_gm] = -1

    th_pts      = theta
    best_score  = np.full(len(pts_xyz), np.inf)
    band_labels = np.full(len(pts_xyz), -1, dtype=int)

    for c in active_clusters:
        if c not in cluster_theta:
            continue
        want_side = 1 if cluster_is_R_arr[c] else -1
        cth  = cluster_theta[c]
        dth  = np.angle(np.exp(1j * (th_pts - cth)))
        cand = np.where((band_side == want_side) &
                         (np.abs(dth) < 1.5 * tooth_period))[0]
        if len(cand) == 0:
            continue
        d, _ = cluster_trees[c].query(pts_xyz[cand], workers=-1)
        better = d < best_score[cand]
        idx = cand[better]
        best_score[idx] = d[better]
        band_labels[idx] = c

    assigned = int((band_labels >= 0).sum())
    log(f"  Zugeordnet: {assigned:,}/{len(pts_xyz):,} "
        f"({assigned/len(pts_xyz)*100:.1f}%)  -- "
        f"unsicher (gm-Grauzone, ausgeschlossen): {int((band_side==0).sum()):,}")

    # ── 6. Zahnindex — NUR Post-hoc-Labeling über k_star, keine Trennung ────
    tooth_all, tooth_angles, _, off_tc, tc = compute_tooth_index(
        theta, z_cyl, k_star, n_teeth, tooth_period)

    cluster_tooth_id = np.full(n_active, -1, dtype=int)
    for c in active_clusters:
        m = band_labels == c
        if m.sum() < 5:
            continue
        tc_med = float(np.median(tc[m]))
        cluster_tooth_id[c] = int(np.round(
            ((tc_med - off_tc + 4.0 * np.pi) % (2.0 * np.pi)) / tooth_period
        )) % n_teeth

    # ── 7. Band-Masken — direkt aus gm_product, kein tc_rel, kein k_star ────
    curv_ok   = curv_band < float(cfg.get("curv_max", 0.08))
    mask_R_ok = (band_labels >= 0) & curv_ok & (band_side == 1)
    mask_L_ok = (band_labels >= 0) & curv_ok & (band_side == -1)

    ratio = mask_R_ok.sum() / max(mask_L_ok.sum(), 1)
    log(f"  Masken: R={mask_R_ok.sum():,}  L={mask_L_ok.sum():,}  "
        f"Ratio={ratio:.2f}  (gm_product-direkt, Segmentierung k-unabhängig)", t0)

    # ── 8. Ausgabe-Arrays ────────────────────────────────────────────────────
    tooth_R = tooth_all.copy()
    tooth_L = tooth_all.copy()

    tol = tooth_period * float(cfg.get("phi_b_tol_frac", 0.10))

    r_safe  = np.maximum(r_band, s4["r_b"] * 1.0005)
    inv_at  = np.tan(np.arccos(np.clip(s4["r_b"] / r_safe, 0.0, 1.0))) \
              - np.arccos(np.clip(s4["r_b"] / r_safe, 0.0, 1.0))
    phi_b_R = (tc + inv_at) % tooth_period
    phi_b_L = (tc - inv_at) % tooth_period

    return dict(
        tooth_R=tooth_R,  tooth_L=tooth_L,
        tooth_all=tooth_all,
        res_R=np.zeros(0, dtype=np.float32),
        res_L=np.zeros(0, dtype=np.float32),
        mask_R_ok=mask_R_ok, mask_L_ok=mask_L_ok,
        tol_R=tol, tol_L=tol,
        phi_b_R=phi_b_R, phi_b_L=phi_b_L,
        tooth_angles=tooth_angles,
        band_labels=band_labels,
        cluster_is_R=cluster_is_R_arr,
        cluster_tooth_id=cluster_tooth_id,
        n_rg_clusters=n_active,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5b — Scan-Profil Cross-Check  (phi_b-unabhängige Verifikation)
# ═══════════════════════════════════════════════════════════════════════════════

def phase5b_scan_profile_crosscheck(s1, s2, s3, s4, s5, cfg):
    """
    NICHT nur ein passiver Cross-Check trotz des Funktionsnamens: das hier
    neu berechnete (tooth_all, tooth_angles) wird vom Dispatcher in
    run_pipeline() UNBEDINGT als s5["tooth_all"]/s5["tooth_angles"]
    übernommen und ist ab dann die EINZIGE, für Phase 6/7/8 verwendete
    Zahn-Index-Referenz -- unabhängig davon, welche Phase-5-Methode
    (DBSCAN/Region-Growing/Schwellwert) zuvor lief und welches tooth_all
    SIE berechnet hatte. Diese Funktion ist also die letzte Instanz für die
    Zahn-Indizierung, nicht nur eine Diagnose-Zwischenstation -- wer die
    Zahn-Zuordnung debuggt, muss HIER anfangen, nicht bei Phase 5.
    Die phi_b-Masken (mask_R_ok/mask_L_ok) dagegen werden tatsächlich nur
    unverändert durchgereicht (siehe Ende der Funktion) -- NUR tooth_all/
    tooth_angles werden hier neu (und autoritativ) bestimmt.

    Ursprünglicher Zweck (weiterhin gültig, s. Diagnose-Logs):
    Zahn-Index via Scan-Profil (vollständig phi_b-unabhängig).

    Methode:
      In theta_corr = theta - k*z ist jeder Zahn ein Peak. Der relative
      Offset eines Punktes zum nächsten Zahnzentrum (tc_rel) bestimmt:
        |tc_rel| > 40% der Teilung  → Lücke/Rand
        sign(tc_rel)                → welche Flanken-Seite (R oder L)

    Cross-Check gegen phi_b (Phase 5):
      - Zahn-Index Übereinstimmung > 95%  → phi_b-Offset korrekt
      - Systematische Verschiebung Δ=±1   → Offset um einen Zahn falsch
      - Seiten-Inkonsistenz > 10%         → L/R-Konventionsproblem
    """
    t0 = time.time()
    log("Phase 5b: Scan-Profil Cross-Check (phi_b-unabhängig)")

    pts_cyl      = s3["pts_b_cyl"].astype(np.float64)
    k_star       = s4["k_star"]
    n_teeth      = s2["n_teeth"]
    tooth_period = s2["tooth_period"]

    theta = pts_cyl[:, 1]
    z     = pts_cyl[:, 2]

    mask_R_ok = s5["mask_R_ok"]
    mask_L_ok = s5["mask_L_ok"]

    # ── tc-Methode: Zahn-Index für ALLE Bandpunkte ────────────────────────────
    tc = (theta - k_star * z) % (2.0 * np.pi)

    # Zahnzentrum-Bestimmung (2026-07-13, ERSETZT 2026-07-15, KORRIGIERT
    # 2026-07-15): zwei vorherige Ansaetze scheiterten an derselben Ursache.
    # Ein synthetischer Ground-Truth-Test zeigte: off_R und off_L (je Seite
    # unabhaengig geschaetzte Peak-Lage) liegen typischerweise MEHR als eine
    # halbe Teilung auseinander (Zahndicke > halbe Teilung ist normal), UND
    # exakt in der Mitte zwischen R- und L-Peak liegt oft eine dichte-freie
    # Luecke (Zahnkopf-Land wird von den Ist-Flankenpunkten meist nicht mit
    # abgedeckt) -- das kombinierte tc-Histogramm hat dadurch effektiv ZWEI
    # Beulen (R, L) statt einer, was die n-te Harmonische (Grundannahme:
    # EIN Dichte-Peak pro Teilung) strukturell fehlkonditioniert, egal ob
    # roh kombiniert, zirkulaer gemittelt, oder dichte-balanciert addiert
    # (alle drei Varianten trafen dasselbe falsche Minimum -- getestet).
    #
    # Fix: da off_R und off_L je EINZELN zuverlaessig sind (nur eine Beule
    # pro Seite), reicht ihr Mittelwert -- man muss nur die richtige der
    # zwei moeglichen "Seiten" waehlen (off_L direkt, oder off_L um eine
    # Teilung verschoben; beide Mittelwerte unterscheiden sich um genau
    # eine halbe Teilung). Statt das per Distanz-Heuristik zu raten (schlug
    # fehl, siehe oben), werden BEIDE Kandidaten tatsaechlich ausprobiert:
    # fuer jeden wird tooth_all berechnet und per R/L-Naechster-Nachbar-
    # Konsistenz bewertet (jeder R-Punkt mit seinem naechsten L-Punkt in tc
    # gepaart -- physisch praktisch immer derselbe Zahn -- teilen sie
    # tooth_all?). Der Kandidat mit hoeherer Konsistenz gewinnt.
    off_R = _phi_b_phase_offset(tc[mask_R_ok], n_teeth, tooth_period)
    off_L = _phi_b_phase_offset(tc[mask_L_ok], n_teeth, tooth_period)
    off_tc_cand_A = float(((off_R + off_L) / 2.0) % tooth_period)
    off_tc_cand_B = float((off_tc_cand_A + tooth_period / 2.0) % tooth_period)

    # WICHTIG (2026-07-15, zweite Korrektur): der naive "naechster L-Punkt
    # in tc" ist bei dicken Zaehnen selbst mehrdeutig! Am Zahn-FUSS (grosser
    # |tc_rel|) liegt der naechste L-Punkt oft NICHT auf demselben Zahn,
    # sondern auf der Nachbarzahnflanke (der Zahn-ZWISCHENRAUM ist schmaler
    # als der Zahn selbst, siehe [[project_gear_pipeline]]) -- das machte die
    # erste Version dieses Sicherheitsnetzes selbst unzuverlaessig (bevorzugte
    # den FALSCHEN Kandidaten). Fix: nur Paare mit KLEINEM Winkelabstand
    # (< Teilung/4) werden gewertet -- das ist die einzige Zone, in der
    # "naechster Nachbar" GARANTIERT vom selben Zahn ist (Zahnkopf-nahe
    # Punkte, wo R und L sich am naechsten kommen). Per synthetischem
    # Ground-Truth-Test bestaetigt: mit dieser Restriktion liefert der
    # richtige Kandidat 100% Konsistenz, der falsche 0% -- klar entscheidbar,
    # waehrend die ungefilterte Version ~67%/~33% zeigte (uneindeutig).
    idx_R_chk = np.where(mask_R_ok)[0]
    idx_L_chk = np.where(mask_L_ok)[0]
    tree_L_chk = None
    samp_R = idx_R_chk
    safe_mask_chk = None
    if len(idx_R_chk) > 0 and len(idx_L_chk) > 0:
        n_chk = min(20000, len(idx_R_chk))
        samp_R = np.random.choice(idx_R_chk, n_chk, replace=False) \
            if len(idx_R_chk) > n_chk else idx_R_chk
        L_xy = np.column_stack([np.cos(tc[idx_L_chk]), np.sin(tc[idx_L_chk])])
        R_xy = np.column_stack([np.cos(tc[samp_R]), np.sin(tc[samp_R])])
        tree_L_chk = KDTree(L_xy)
        d_chord, nn_L_chk = tree_L_chk.query(R_xy, k=1, workers=-1)
        partner_idx_chk = idx_L_chk[nn_L_chk]
        ang_dist_chk = 2.0 * np.arcsin(np.clip(d_chord / 2.0, 0.0, 1.0))
        safe_mask_chk = ang_dist_chk < (tooth_period / 4.0)

    def _tooth_all_for(off_cand):
        return (np.round(
            ((tc - off_cand + 4.0 * np.pi) % (2.0 * np.pi)) / tooth_period
        ).astype(int) % n_teeth)

    def _rl_consistency(ta):
        if tree_L_chk is None or safe_mask_chk is None or safe_mask_chk.sum() < 20:
            return 1.0, 0
        diff = (ta[samp_R[safe_mask_chk]].astype(int)
                - ta[partner_idx_chk[safe_mask_chk]].astype(int)) % n_teeth
        diff_signed = np.where(diff > n_teeth // 2, diff - n_teeth, diff)
        return float((diff_signed == 0).mean()), int(safe_mask_chk.sum())

    tooth_all_A = _tooth_all_for(off_tc_cand_A)
    tooth_all_B = _tooth_all_for(off_tc_cand_B)
    consist_A, n_safe_A = _rl_consistency(tooth_all_A)
    consist_B, n_safe_B = _rl_consistency(tooth_all_B)

    if consist_A >= consist_B:
        off_tc, tooth_all, best_consist = off_tc_cand_A, tooth_all_A, consist_A
    else:
        off_tc, tooth_all, best_consist = off_tc_cand_B, tooth_all_B, consist_B

    log(f"  Zahnzentrum-Bestimmung (tc-Profil): R-Peak={np.degrees(off_R):.3f}°  "
        f"L-Peak={np.degrees(off_L):.3f}°  Kandidat A={np.degrees(off_tc_cand_A):.3f}° "
        f"(R/L-Konsistenz={consist_A*100:.1f}%, n_safe={n_safe_A})  "
        f"Kandidat B={np.degrees(off_tc_cand_B):.3f}° "
        f"(R/L-Konsistenz={consist_B*100:.1f}%, n_safe={n_safe_B})  "
        f"→ gewaehlt: {np.degrees(off_tc):.3f}°")
    if best_consist < 0.9:
        log(f"  WARNUNG R/L-Zahnindex-Konsistenz: bester Kandidat erreicht nur "
            f"{best_consist*100:.1f}% -- tooth_all vermutlich weiterhin "
            f"fehlerhaft, Ergebnis pruefen!")

    # Zahnzentrum für jeden Punkt → vorzeichenbehafteter Abstand
    tooth_angles_tc = (np.arange(n_teeth) * tooth_period + off_tc) % (2.0 * np.pi)
    ta_assigned     = tooth_angles_tc[tooth_all]
    tc_rel = ((tc - ta_assigned + np.pi) % (2.0 * np.pi)) - np.pi

    # ── Cross-Check: Zahn-Index phi_b vs. tc-Profil ───────────────────────────
    tooth_phi_R = s5["tooth_R"]
    tooth_phi_L = s5["tooth_L"]

    agree_R = (tooth_all[mask_R_ok] == tooth_phi_R[mask_R_ok])
    agree_L = (tooth_all[mask_L_ok] == tooth_phi_L[mask_L_ok])

    log(f"  Zahn-Index Agreement (tc-Profil vs. phi_b):")
    log(f"    R-Punkte: {agree_R.mean()*100:.1f}%  "
        f"({agree_R.sum():,}/{mask_R_ok.sum():,})")
    log(f"    L-Punkte: {agree_L.mean()*100:.1f}%  "
        f"({agree_L.sum():,}/{mask_L_ok.sum():,})")

    # Verschiebungs-Histogramm (off-by-N-Fehler)
    # WICHTIG: für R-Punkte gegen tooth_phi_R vergleichen, für L-Punkte
    # gegen tooth_phi_L — vorher wurde hier für BEIDE Seiten immer
    # tooth_phi_R verwendet. In den aktuell aktiven Modi (phase5_dbscan_flank
    # etc.) sind tooth_phi_R und tooth_phi_L zwar identische Arrays
    # (tooth_R=tooth_L=tooth_all.copy()), also ändert dieser Teil dort
    # nichts — aber für phase5_tooth_assignment (wo R/L tatsächlich getrennte
    # Zahn-Indizes haben können) war das ein echter Vergleichsfehler.
    mask_seg  = mask_R_ok | mask_L_ok
    tooth_phi_seg = np.where(mask_R_ok, tooth_phi_R,
                              np.where(mask_L_ok, tooth_phi_L, 0))[mask_seg]
    diff_raw = (tooth_all[mask_seg].astype(int) -
                tooth_phi_seg.astype(int)) % n_teeth
    diff_signed = np.where(diff_raw > n_teeth // 2, diff_raw - n_teeth, diff_raw)
    total_seg = int(mask_seg.sum())
    log(f"  Delta Zahn-Index (Profil - phi_b), R+L-Punkte:")
    for d in range(-3, 4):
        cnt = int((diff_signed == d).sum())
        if cnt == 0:
            continue
        bar = chr(9608) * min(cnt * 35 // max(total_seg, 1), 35)
        flag = "  <- off-by-one!" if abs(d) == 1 and cnt > total_seg * 0.05 else ""
        log(f"    Delta={d:+d}: {cnt:8,}  {bar}{flag}")

    # ── Seiten-Konsistenz ─────────────────────────────────────────────────────
    r_sign_frac = float((tc_rel[mask_R_ok] > 0).mean())
    r_side_positive = r_sign_frac > 0.5
    log(f"  Seiten-Konvention: R-Punkte mit tc_rel>0: {r_sign_frac*100:.1f}%  "
        f"({'tc_rel>0 ist R' if r_side_positive else 'tc_rel<0 ist R'})")

    in_flank = np.abs(tc_rel) < tooth_period * 0.40
    mask_R_prof = in_flank & (tc_rel > 0 if r_side_positive else tc_rel < 0)
    mask_L_prof = in_flank & (tc_rel < 0 if r_side_positive else tc_rel > 0)

    both_R   = int((mask_R_ok & mask_R_prof).sum())
    both_L   = int((mask_L_ok & mask_L_prof).sum())
    cross_RL = int((mask_R_ok & mask_L_prof).sum())
    cross_LR = int((mask_L_ok & mask_R_prof).sum())
    denom = max(both_R + both_L + cross_RL + cross_LR, 1)
    inkonsistenz = (cross_RL + cross_LR) / denom
    log(f"  Seiten-Konsistenz: R&R={both_R:,}  R&L={cross_RL:,}  "
        f"L&R={cross_LR:,}  L&L={both_L:,}")
    if inkonsistenz > 0.10:
        log(f"  [Warnung] {inkonsistenz*100:.1f}% Seiten-Inkonsistenz — "
            f"L/R-Konvention pruefen!")
    else:
        log(f"  OK: phi_b und Scan-Profil konsistent ({inkonsistenz*100:.1f}% Abw.)")

    ratio = mask_R_ok.sum() / max(mask_L_ok.sum(), 1)
    log(f"  Masken: R={mask_R_ok.sum():,}  L={mask_L_ok.sum():,}  "
        f"Ratio={ratio:.2f}", t0)

    return dict(
        mask_R_ok=mask_R_ok,
        mask_L_ok=mask_L_ok,
        tooth_all=tooth_all,
        tooth_angles=tooth_angles_tc,
        tc_rel=tc_rel,
        r_side_positive=r_side_positive,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5c — Graph-Spatial-Kohärenz  (Cluster-Bereinigung)
# ═══════════════════════════════════════════════════════════════════════════════

def phase5c_spatial_coherence(s2, s3, s5b, cfg):
    """
    Räumliche Kohärenz-Bereinigung via Graph-Zusammenhangskomponenten.

    Problem: phi_b Winner-Takes-All kann Punkte einem Zahn zuweisen, obwohl
    sie räumlich weit entfernt liegen (z.B. Rauschen oder Scan-Artefakte
    mit zufällig kleinem phi_b-Residuum).

    Methode: Für jeden (tooth_id, side)-Cluster:
      1. Punkte als Knoten; Kanten = k-NN-Verbindungen INNERHALB des Clusters
      2. Zusammenhangskomponenten (ZK) via scipy.sparse.csgraph
      3. Nur die größte ZK behalten; alle anderen entfernen

    Der k-NN-Graph aus Phase 3 (nn_idx) wird direkt wiederverwendet —
    kein zusätzlicher KDTree-Build nötig.
    """
    t0 = time.time()
    log("Phase 5c: Graph-Spatial-Kohärenz (Zusammenhangskomponenten)")

    nn_idx   = s3["nn_idx"]       # (N_band, k_nn) — aus Phase 3
    n_teeth  = s2["n_teeth"]
    tooth_all = s5b["tooth_all"]

    mask_R_ok = s5b["mask_R_ok"].copy()
    mask_L_ok = s5b["mask_L_ok"].copy()

    total_removed_R = 0
    total_removed_L = 0
    n_split = 0
    n_fragmented_skip = 0  # Cluster mit GZK < min_gzk_frac → nicht bereinigt

    # Mindest-GZK-Anteil: kNN-Graph verbindet benachbarte Zähne (Cross-Tooth-
    # Kanten), daher sind intra-Cluster-ZK oft winzig. Wenn GZK < min_gzk_frac
    # des Gesamt-Clusters → kNN-Auflösung zu grob, Bereinigung würde zu viel
    # echte Daten entfernen.
    min_gzk_frac = float(cfg.get("min_gzk_frac", 0.50))

    for ti in range(n_teeth):
        for side, mask_ok in [("R", mask_R_ok), ("L", mask_L_ok)]:
            cluster_idx = np.where(mask_ok & (tooth_all == ti))[0]
            n_c = len(cluster_idx)
            if n_c < 3:
                continue

            # Sparse Adjazenzmatrix (nur intra-cluster Kanten)
            idx_set  = set(cluster_idx.tolist())
            local_map = {g: l for l, g in enumerate(cluster_idx)}
            rows, cols = [], []
            for li, gi in enumerate(cluster_idx):
                for nb in nn_idx[gi]:
                    if nb in idx_set:
                        rows.append(li)
                        cols.append(local_map[nb])
            if not rows:
                continue

            data   = np.ones(len(rows), dtype=np.float32)
            adj    = csr_matrix((data, (rows, cols)), shape=(n_c, n_c))
            n_comp, labels = connected_components(adj, directed=False,
                                                   return_labels=True)
            if n_comp <= 1:
                continue

            # Größte Komponente
            comp_sizes = np.bincount(labels)
            largest    = int(np.argmax(comp_sizes))

            # Sicherheitscheck: GZK-Anteil zu klein → kNN-Graph hat zu viele
            # Cross-Tooth-Kanten, Bereinigung wäre destruktiv → überspringen
            gzk_frac = comp_sizes[largest] / n_c
            if gzk_frac < min_gzk_frac:
                n_fragmented_skip += 1
                continue

            remove_local = np.where(labels != largest)[0]
            removed = len(remove_local)

            for li in remove_local:
                mask_ok[cluster_idx[li]] = False

            n_split += 1
            if side == "R":
                total_removed_R += removed
            else:
                total_removed_L += removed
            if removed > 10:
                log(f"  Zahn {ti:2d} {side}: {n_c} Pkt, {n_comp} ZK "
                    f"-> {removed} entfernt (GZK: {comp_sizes[largest]}, "
                    f"{gzk_frac*100:.0f}%)")

    log(f"  Gesamt: {n_split} Cluster bereinigt  "
        f"entfernt R={total_removed_R:,}  L={total_removed_L:,}  "
        f"(übersprungen: {n_fragmented_skip} mit GZK<{min_gzk_frac*100:.0f}%)")
    ratio = mask_R_ok.sum() / max(mask_L_ok.sum(), 1)
    log(f"  Nach Bereinigung: R={mask_R_ok.sum():,}  "
        f"L={mask_L_ok.sum():,}  Ratio={ratio:.2f}", t0)

    return dict(
        mask_R_ok=mask_R_ok,
        mask_L_ok=mask_L_ok,
        tooth_all=tooth_all,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 6 — Soll-Verifikation  (erster MY_GEAR-Einsatz — optional)
# ═══════════════════════════════════════════════════════════════════════════════

def phase6_soll_verify(s1, s2, s3, s4, s5, cfg):
    """
    Verifikation der Segmentierung gegen die Soll-Oberfläche.

    Erzeugt eine Soll-Punktwolke aus MY_GEAR, baut einen KD-Tree,
    und berechnet für jeden segmentierten Ist-Punkt den Abstand zur Soll-Flanke.

    Punkte mit Abstand > soll_dist_flag werden als verdächtig gemeldet.
    Punkte mit Abstand > soll_dist_drop werden aus dem Cluster entfernt.

    Gibt korrigierte mask_R_ok / mask_L_ok zurück.
    """
    t0 = time.time()
    log("Phase 6: Soll-Verifikation")

    try:
        from gear_soll import MY_GEAR, compute_derived_geometry, generate_soll_flank
    except ImportError as e:
        # Nur "Modul/Namen nicht gefunden" gilt als "MY_GEAR noch nicht
        # befuellt" und wird still uebersprungen. Ein echter Bug in
        # gear_soll.py (Syntaxfehler, kaputte Funktion) wirft i.d.R. KEINEN
        # ImportError, sondern SyntaxError/AttributeError/etc. -- die NICHT
        # hier abgefangen werden, damit ein echter Fehler laut auffaellt
        # statt als "MY_GEAR leer" misinterpretiert zu werden.
        log(f"  ⚠ gear_soll nicht importierbar ({e}) — Verifikation übersprungen")
        return dict(mask_R_ok=s5["mask_R_ok"], mask_L_ok=s5["mask_L_ok"],
                    soll_available=False)

    # Prüfe ob MY_GEAR-Parameter ausgefüllt sind
    try:
        derived = compute_derived_geometry(MY_GEAR)
        if derived['r_b'] <= 0:
            raise ValueError("r_b=0")
    except Exception as e:
        log(f"  ⚠ MY_GEAR nicht vollständig ({e}) — Verifikation übersprungen")
        return dict(mask_R_ok=s5["mask_R_ok"], mask_L_ok=s5["mask_L_ok"],
                    soll_available=False)

    # Konsistenz-Check: ist-erkanntes r_b vs MY_GEAR.d_b/2
    r_b_soll = derived['r_b']
    r_b_ist  = s4["r_b"]
    delta_rb = abs(r_b_ist - r_b_soll)
    log(f"  r_b_ist={r_b_ist:.4f}mm  r_b_soll={r_b_soll:.4f}mm  "
        f"Δ={delta_rb*1000:.1f}µm  "
        f"({'✓ konsistent' if delta_rb < 0.5 else '⚠ Diskrepanz'})")

    beta_soll = MY_GEAR.beta_0_deg
    beta_ist  = s4["beta_ist"]
    delta_b   = abs(beta_ist - beta_soll)

    # [Diag] Steigungsraten-Vergleich k_star_ist vs Soll-c (2026-07-14):
    # der beta_ist/beta_soll-Vergleich oben ist KEIN Vergleich gleichartiger
    # Groessen -- beta_ist wird bei r_ref (Scan-Referenzradius ~102.9mm)
    # ausgewertet, beta_soll=MY_GEAR.beta_0_deg ist der Teilkreis-
    # Schraegungswinkel (bei r_0). Fuer _fold_to_canonical() relevant ist
    # dagegen die Steigungsrate dtheta/dz: Ist nutzt k_star_ist direkt (roh
    # aus Phase 4 gefittet), waehrend generate_soll_flank() intern
    # hand*derived['c'] nutzt (c=tan(beta_b)/r_b, GRUNDKREIS-Schraegungs-
    # winkel, nicht beta_0). Diese zwei werden hier direkt (gleiche Einheit,
    # 1/mm) verglichen, und die daraus zu erwartende z-getriebene
    # Tangentialverschiebung am Zahnbreiten-Rand abgeschaetzt, um die
    # z-gebinnte Fehlerkurve unten quantitativ zu erklaeren statt nur zu
    # beschreiben.
    k_star_ist_val = float(s4["k_star"])
    k_soll_equiv   = float(MY_GEAR.hand) * float(derived['c'])
    dk_ist_soll    = k_star_ist_val - k_soll_equiv
    half_b         = float(MY_GEAR.b) / 2.0
    r_lever        = derived['r_b']
    pred_edge_um   = abs(dk_ist_soll) * half_b * r_lever * 1000.0
    log(f"  [Diag] Steigungsrate: k_star_ist={k_star_ist_val:.7f}/mm  "
        f"k_soll_equiv(hand*c)={k_soll_equiv:.7f}/mm  "
        f"dk={dk_ist_soll:+.7f}/mm  "
        f"vorhergesagte Randverschiebung (bei z=0/b, Hebel={half_b:.1f}mm, "
        f"r={r_lever:.1f}mm) ~= {pred_edge_um:.0f}um")
    log(f"  β_ist={beta_ist:.4f}°  β_soll={beta_soll:.4f}°  "
        f"Δ={delta_b:.4f}°  "
        f"({'✓ konsistent' if delta_b < 0.3 else '⚠ Diskrepanz'})")

    # Kanonische Soll-Oberfläche: NUR EIN Zahn (2 Flanken), nicht alle N.
    # Ein Soll-Zahn unterscheidet sich von jedem anderen ausschließlich durch
    # seine Rotationslage (das Soll-Modell ist exakt N-fach rotationsperiodisch,
    # keine Näherung) — statt N Kopien zu erzeugen und zu KD-Tree'n, wird jeder
    # Ist-Punkt anhand seines (in Phase 5 bereits bekannten) Zahn-Index in die
    # Rotationslage von "Zahn 0" zurückgedreht und dort verglichen. Spart
    # ~N× Speicher/Aufbauzeit für den KD-Tree (z.B. N=83 → 83× kleiner).
    n_teeth = s2["n_teeth"]
    # n_u/n_v=300x200 getestet (2026-07-14, synthetischer Ground-Truth-Test):
    # ~19x mehr Soll-Punkte, ~13x Laufzeit (63s -> 807s), aber mean_dist nur
    # 1437um -> 1407um (~2%) -- Gitterdiskretisierung/Kruemmungs-Aliasing als
    # Ursache des ~1.4mm-Rest-Fehlers AUSGESCHLOSSEN. Zurueckgesetzt auf 80x40
    # (siehe [[project_gear_pipeline]] fuer die vollstaendige Diagnose-Kette).
    n_u, n_v = 80, 40
    soll_pts  = []
    soll_nrml = []

    log(f"  Generiere kanonische Soll-Flanke (Zahn 0): 2 Flanken × {n_u}×{n_v} "
        f"(Ist hat {n_teeth} Zähne, per Rotationsfaltung verglichen)")
    for side in ['R', 'L']:
        fl = generate_soll_flank(MY_GEAR, n_u=n_u, n_v=n_v, tooth_i=0, side=side)
        soll_pts .append(fl['pts_xyz'].reshape(-1, 3))
        soll_nrml.append(fl['normals'].reshape(-1, 3))

    soll_pts  = np.vstack(soll_pts).astype(np.float64)
    soll_nrml = np.vstack(soll_nrml).astype(np.float64)

    # 1-DOF Rotations-Alignment (nur phi0, Achse + Zentrum aus Phase 1)
    axis     = s1["axis"]
    axis_u   = axis / np.linalg.norm(axis)
    center   = s1["center"]
    pts_b    = s3["pts_b_xyz"].astype(np.float64)
    tooth_all = s5["tooth_all"]
    pitch_soll = 2.0 * np.pi / n_teeth
    mask_all = s5["mask_R_ok"] | s5["mask_L_ok"]

    # Subsample für Alignment-Suche
    idx_al = np.where(mask_all)[0]
    if len(idx_al) > 5000:
        idx_al = np.random.choice(idx_al, 5000, replace=False)
    pts_al   = pts_b[idx_al]
    tooth_al = tooth_all[idx_al]

    # Koordinaten-Transformation Soll → Welt:
    #   Soll-Flanken werden in lokalen Gear-Koords generiert:
    #     XY = (0,0) (Zahnrad-Achse), z ∈ [0, b] (Zahnbreite ab Unterkante)
    #   Ist-Punktwolke = rohe Welt-Koordinaten:
    #     XY ≈ center.xy (Zahnrad-Achse in Welt), z ∈ [z_bottom, z_bottom + b]
    #   Benötigte Verschiebung:
    #     XY: center.xy  (≈0 falls Gear bereits bei Welt-Ursprung)
    #     Z:  center.z + z_off = Welt-z der Gear-Unterkante
    #         (weil pts_b_xyz.z = raw-world-z, center.z + z_off = min(world-z))
    z_off_val  = float(s1.get("z_off", 0.0))
    z_shift    = float(center[2]) + z_off_val
    xyz_shift  = np.array([center[0], center[1], z_shift])
    log(f"  [Debug] center=({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})mm  "
        f"z_off={z_off_val:.2f}mm  z_shift={z_shift:.2f}mm  "
        f"soll_z∈[{soll_pts[:,2].min():.1f}, {soll_pts[:,2].max():.1f}]mm  "
        f"→ nach Shift: [{soll_pts[:,2].min()+z_shift:.1f}, {soll_pts[:,2].max()+z_shift:.1f}]mm")

    def _fold_to_canonical(pts_world, tooth_idx, phi):
        """
        Dreht Welt-Punkte in die (unrotierte, unverschobene) Soll-Ortslage von
        Zahn 0 zurück: erst Welt→Soll-Ursprung verschieben, dann um
        -(phi + tooth_idx·pitch) um `axis` zurückdrehen. Approximation: nimmt
        exakte Soll-Teilung (ti·pitch_soll) an — für diesen groben Ausreißer-
        Check (soll_dist_flag/drop im Bereich 0.3–0.8mm) unbedenklich, da
        reale Teilungsfehler typischerweise << dieser Toleranz liegen. Für die
        mikrometergenaue Soll-Ist-Differenz nutzt phase4_uv.py ohnehin die
        Ist-eigene Medianphase je Zahn, nicht die hier angenommene Nominal-Teilung.
        """
        angles = -(phi + tooth_idx * pitch_soll)
        return _rotate_about_axis(pts_world - xyz_shift, axis_u, angles)

    tree_soll = KDTree(soll_pts)   # EINMAL gebaut — bleibt über die ganze Suche fix,
                                   # da jetzt die Ist-Punkte gedreht werden, nicht der Soll.

    def alignment_cost(phi):
        folded = _fold_to_canonical(pts_al, tooth_al, phi)
        d, _ = tree_soll.query(folded, k=1, workers=-1)
        return float(np.mean(d))

    # Grobes Gitter — VOLLER Kreis, nicht nur eine Teilung (Bugfix 2026-07-14).
    # Vorher: np.linspace(0, 2*pi/n_teeth, 60) -- nahm an, phi0 sei auf eine
    # einzelne Teilungsbreite beschraenkt. Das waere nur richtig, wenn
    # alignment_cost() periodisch mit Periode=pitch_soll waere -- ist es aber
    # NICHT: soll_pts enthaelt nur EINEN Zahn (Zahn 0), keinen vollen,
    # rotationssymmetrischen Ring, also hat alignment_cost() keine kleinere
    # Periode als 2*pi. Welcher der 83 realen Zaehne von Phase 5 zufaellig
    # als "Index 0" gelabelt wurde (abhaengig vom arbitraeren off_tc-Bezug)
    # hat KEINEN Bezug dazu, wo generate_soll_flank(tooth_i=0) seinen Zahn 0
    # platziert (nahe Weltwinkel 0) -- der noetige Korrekturwinkel kann daher
    # ueberall im vollen Kreis liegen. Bestaetigt per [Diag]-Fehlerzerlegung:
    # gemessener Restfehler war ueberwiegend planar/radial (|Δxy|=140mm vs
    # |Δz|=20mm) und der Winkel zwischen folded- und soll-Zentroid lag bei
    # ~96° (~22 Teilungen) -- weit ausserhalb des alten ±¼-Teilung-Suchfensters.
    # Aufloesung proportional zur alten Gitterdichte beibehalten (60 Punkte je
    # Teilung), nur ueber den vollen Kreis statt einer einzelnen Teilung.
    n_grid = 60 * n_teeth
    phi_candidates = np.linspace(0, 2*np.pi, n_grid, endpoint=False)
    costs = np.array([alignment_cost(p) for p in phi_candidates])

    # Mehrdeutigkeits-Fix (2026-07-14): ein synthetischer Ground-Truth-Test
    # bewies, dass diese Grobsuche in ein FALSCHES lokales Minimum laufen
    # kann, das zufaellig einen kleineren mean_dist hat als das wahre
    # globale Optimum -- bei vielen aehnlichen, glatten Zaehnen findet eine
    # um genau eine halbe Teilung falsche Rotation trotzdem fast ueberall
    # einen nahen, aber GEOMETRISCH FALSCHEN Soll-Punkt (R-Flanke matched
    # auf L-Flanke). Am wahren Optimum fiel der Fehler auf Rauschniveau
    # (~15um), am falschen lag er >1mm -- aber die reine Abstandsmetrik war
    # dafuer blind. Siehe [[project_gear_pipeline]] fuer die Diagnose-Kette.
    #
    # Fix: die M besten, hinreichend weit auseinanderliegenden lokalen
    # Minima werden verfeinert und zusaetzlich per R/L-Seiten-Konsistenz
    # geprueft (bei korrektem phi0 muss eine ist-R-Flanke auf eine soll-R-
    # Flanke matchen, nicht zufaellig L) -- scharf fuer genau diese Art
    # Fehler (~50% Konsistenz bei falschem phi0 vs. nahe 100% bei richtigem).
    n_soll_per_side = n_u * n_v
    soll_is_L = np.concatenate([
        np.zeros(n_soll_per_side, dtype=bool),
        np.ones(n_soll_per_side, dtype=bool),
    ])
    side_al = s5["mask_L_ok"][idx_al]   # True=L, False=R -- dieselbe Stichprobe wie pts_al/tooth_al

    def _side_consistency(phi):
        folded = _fold_to_canonical(pts_al, tooth_al, phi)
        _, nn_idx = tree_soll.query(folded, k=1, workers=-1)
        return float(np.mean(soll_is_L[nn_idx] == side_al))

    min_sep = pitch_soll / 2.0
    order = np.argsort(costs)
    candidates = []
    for i in order:
        phi_i = phi_candidates[i]
        if all(min(abs(phi_i - c), 2 * np.pi - abs(phi_i - c)) >= min_sep
               for c in candidates):
            candidates.append(phi_i)
        if len(candidates) >= 5:
            break

    cand_results = []
    for phi_c in candidates:
        res_c = minimize_scalar(
            alignment_cost,
            bounds=(phi_c - np.pi / n_teeth / 4, phi_c + np.pi / n_teeth / 4),
            method='bounded', options={'xatol': 1e-5}
        )
        phi_refined = float(res_c.x)
        cost_refined = float(res_c.fun)
        consist = _side_consistency(phi_refined)
        cand_results.append((phi_refined, cost_refined, consist))

    log(f"  phi0-Grobsuche (voller Kreis, {n_grid} Punkte): "
        f"{len(cand_results)} getrennte Kandidaten verfeinert:")
    for phi_c, cost_c, consist_c in sorted(cand_results, key=lambda c: c[1]):
        log(f"    phi0={np.degrees(phi_c):8.3f}deg  mean_dist={cost_c*1000:7.1f}um  "
            f"R/L-Konsistenz={consist_c*100:5.1f}%")

    good = [c for c in cand_results if c[2] >= 0.9]
    if good:
        phi0, align_err, best_consist = min(good, key=lambda c: c[1])
        log(f"  Rotations-Alignment: phi0={np.degrees(phi0):.4f}deg  "
            f"mean_dist={align_err*1000:.1f}um  "
            f"(R/L-Konsistenz={best_consist*100:.1f}%, "
            f"{len(good)}/{len(cand_results)} Kandidaten bestanden)")
    else:
        phi0, align_err, best_consist = min(cand_results, key=lambda c: c[1])
        log(f"  WARNUNG Rotations-Alignment: KEIN Kandidat erreichte >=90% "
            f"R/L-Konsistenz -- nutze besten mean_dist als Fallback. "
            f"phi0={np.degrees(phi0):.4f}deg  mean_dist={align_err*1000:.1f}um  "
            f"R/L-Konsistenz={best_consist*100:.1f}% -- Ergebnis mit Vorsicht pruefen!")

    phi0_rough = phi0   # fuer die nachfolgende Feinsuche-Diagnose weiterverwenden

    # [Diag] Feinsuche um phi0 (2026-07-14): prueft ob die Grobsuche (voller
    # Kreis, n_grid Punkte) + gebundene Brent-Verfeinerung (+/- Teilung/4)
    # tatsaechlich das globale Minimum von alignment_cost(phi) gefunden hat,
    # oder ob eine feinere/breitere Suche einen deutlich besseren Wert
    # findet. Motivation: ein synthetischer Ground-Truth-Testlauf zeigte
    # einen ~1.4mm Rest-Fehler trotz nur 15um injiziertem Rauschen und
    # perfekt erkanntem N/k_star/r_b -- die Frage ist, ob dieser Rest ein
    # Suchproblem ist (behebbar) oder ein echter, durch reine Rotation nicht
    # aufloesbarer Fehlerboden (Hinweis auf eine andere Fehlerquelle).
    n_fine = 6000
    phi_fine = np.linspace(phi0_rough - pitch_soll, phi0_rough + pitch_soll, n_fine)
    costs_fine = [alignment_cost(p) for p in phi_fine]
    best_fine_idx = int(np.argmin(costs_fine))
    phi0_fine = float(phi_fine[best_fine_idx])
    mean_dist_fine = float(costs_fine[best_fine_idx])
    improve_pct = (align_err - mean_dist_fine) / align_err * 100.0 if align_err > 0 else 0.0
    log(f"  [Diag] Feinsuche um phi0 (+/- 1 Teilung, {n_fine} Punkte): "
        f"bestes phi0_fine={np.degrees(phi0_fine):.4f}deg  "
        f"mean_dist_fine={mean_dist_fine*1000:.1f}um  "
        f"(aktuell: phi0={np.degrees(phi0):.4f}deg mean_dist={align_err*1000:.1f}um, "
        f"Verbesserung={improve_pct:.1f}%)")

    # ── [Diag] Fehler-Zerlegung (2026-07-14) ─────────────────────────────────
    # hand-Flip allein hat den Alignment-Fehler NICHT behoben (von James
    # bestaetigt). Statt weiter einzelne MY_GEAR-Parameter zu raten: zerlegt
    # den mean_dist in z-Anteil (deutet auf z_shift-Fehler) vs. XY/radial-
    # Anteil (deutet auf Rotations-/Flanken-/Groessen-Fehler), damit die
    # naechste Hypothese datenbasiert statt geraten gewaehlt werden kann.
    folded_best = _fold_to_canonical(pts_al, tooth_al, phi0)
    d_best, idx_best = tree_soll.query(folded_best, k=1, workers=-1)
    matched_soll = soll_pts[idx_best]

    dz_arr  = folded_best[:, 2] - matched_soll[:, 2]
    dxy_arr = np.linalg.norm(folded_best[:, :2] - matched_soll[:, :2], axis=1)
    r_folded       = np.linalg.norm(folded_best[:, :2], axis=1)
    r_soll_matched = np.linalg.norm(matched_soll[:, :2], axis=1)

    log(f"  [Diag] Fehler-Zerlegung: |Δz|_median={np.median(np.abs(dz_arr)):.2f}mm  "
        f"|Δxy|_median={np.median(dxy_arr):.2f}mm  "
        f"(zum Vergleich: Gesamt mean_dist={align_err:.2f}mm)")
    log(f"  [Diag] Radial (Achsabstand): folded median={np.median(r_folded):.2f}mm  "
        f"soll(matched) median={np.median(r_soll_matched):.2f}mm  "
        f"Δr={np.median(r_folded) - np.median(r_soll_matched):+.2f}mm")
    log(f"  [Diag] Zentroid: folded=({folded_best[:,0].mean():.2f},"
        f"{folded_best[:,1].mean():.2f},{folded_best[:,2].mean():.2f})mm  "
        f"soll=({soll_pts[:,0].mean():.2f},{soll_pts[:,1].mean():.2f},"
        f"{soll_pts[:,2].mean():.2f})mm  "
        f"z-Bereich folded=[{folded_best[:,2].min():.1f},{folded_best[:,2].max():.1f}]mm  "
        f"soll=[{soll_pts[:,2].min():.1f},{soll_pts[:,2].max():.1f}]mm")

    # [Diag] Z- und R-gebinnte Fehleranalyse (2026-07-14): der Translationsfit
    # unten hilft nur wenig (~12% in einem ersten Testlauf), und die
    # |f_y_raw|-Perzentile sind sehr breit gestreut -- das spricht eher fuer
    # einen POSITIONSABHAENGIGEN Fehler (z.B. ein beta/k_star-Restfehler, der
    # sich ueber die Zahnbreite aufsummiert) als fuer eine konstante Achsen-/
    # Zentrum-/z_shift-Verschiebung. Testet das direkt: bin die bereits
    # berechneten dxy_arr/d_best nach z-Position (Zahnbreiten-Richtung) und
    # nach Radius (Profil-Richtung) -- ein Trend in z deutet auf einen
    # Steigungs-/Lead-Fehler (beta bzw. k_star), ein Trend in r eher auf
    # einen Profil-/Groessen-Fehler (r_b, Kopfspiel, Profilverschiebung).
    z_pos = folded_best[:, 2]
    n_bins = 6
    z_edges = np.linspace(z_pos.min(), z_pos.max(), n_bins + 1)
    z_bin_idx = np.clip(np.digitize(z_pos, z_edges[1:-1]), 0, n_bins - 1)
    log("  [Diag] Fehler vs. z-Position (Zahnbreite, Lead-Richtung):")
    for b in range(n_bins):
        sel = z_bin_idx == b
        if sel.sum() < 5:
            continue
        log(f"    z∈[{z_edges[b]:5.1f},{z_edges[b+1]:5.1f}]mm  n={sel.sum():6d}  "
            f"median|Δxy|={np.median(dxy_arr[sel])*1000:6.0f}um  "
            f"median|Δz|={np.median(np.abs(dz_arr[sel]))*1000:6.0f}um  "
            f"median d_best={np.median(d_best[sel])*1000:6.0f}um")

    r_pos = r_folded
    r_edges = np.linspace(r_pos.min(), r_pos.max(), n_bins + 1)
    r_bin_idx = np.clip(np.digitize(r_pos, r_edges[1:-1]), 0, n_bins - 1)
    log("  [Diag] Fehler vs. Radius (Profil-Richtung):")
    for b in range(n_bins):
        sel = r_bin_idx == b
        if sel.sum() < 5:
            continue
        log(f"    r∈[{r_edges[b]:6.2f},{r_edges[b+1]:6.2f}]mm  n={sel.sum():6d}  "
            f"median|Δxy|={np.median(dxy_arr[sel])*1000:6.0f}um  "
            f"median d_best={np.median(d_best[sel])*1000:6.0f}um")

    def _cost_xyz(delta):
        d, _ = tree_soll.query(folded_best + delta, k=1, workers=-1)
        return float(np.mean(d))

    res_xyz = minimize(_cost_xyz, x0=np.zeros(3), method='Nelder-Mead',
                        options={'xatol': 1e-4, 'fatol': 1e-6})
    delta_opt = res_xyz.x
    log(f"  [Diag] Zusatz-Translationsfit (phi0-Suche optimiert nur Rotation; "
        f"xyz_shift stammt unverifiziert aus Phase 1): "
        f"delta_xyz=({delta_opt[0]:+.2f},{delta_opt[1]:+.2f},{delta_opt[2]:+.2f})mm  "
        f"mean_dist {align_err*1000:.0f}um -> {res_xyz.fun*1000:.0f}um")

    # [Diag] Perzentil-Verteilung von |f_y_raw|, UNCONDITIONAL (2026-07-14):
    # laeuft auch dann, wenn der align_max-Gate gleich danach das eigentliche
    # Cleaning uebersprint. Grund: soll_dist_flag/soll_dist_drop koennen nur
    # datenbasiert kalibriert werden, wenn man die tatsaechliche Restfehler-
    # Verteilung sieht -- auch aus Laeufen mit (noch) nicht ausreichendem
    # Alignment. Recomputes the same mask_seg/f_y_raw the block below computes
    # again for the actual flag/drop step -- intentional, keeps this diagnostic
    # fully independent of the align_max gate without touching that code path.
    _mask_seg_diag   = s5["mask_R_ok"] | s5["mask_L_ok"]
    _tooth_seg_diag  = tooth_all[_mask_seg_diag]
    _folded_seg_diag = _fold_to_canonical(pts_b[_mask_seg_diag], _tooth_seg_diag, phi0)
    _d_diag, _nn_diag = tree_soll.query(_folded_seg_diag, k=1, workers=-1)
    _delta_diag = _folded_seg_diag - soll_pts[_nn_diag]
    _f_y_diag   = np.abs(np.einsum('ij,ij->i', _delta_diag, soll_nrml[_nn_diag]))
    _pct = np.percentile(_f_y_diag, [10, 25, 50, 75, 90, 95, 99])
    log(f"  [Diag] |f_y_raw|-Perzentile (um): "
        f"P10={_pct[0]*1000:.0f} P25={_pct[1]*1000:.0f} P50={_pct[2]*1000:.0f} "
        f"P75={_pct[3]*1000:.0f} P90={_pct[4]*1000:.0f} P95={_pct[5]*1000:.0f} "
        f"P99={_pct[6]*1000:.0f}  -- Basis fuer soll_dist_flag/drop-Kalibrierung; "
        f"solange der Alignment-Restfehler oben nicht beseitigt ist, nur als "
        f"Diagnose lesen, nicht als Schwellenwert-Quelle")

    # Plausibilitätscheck: > 5 mm → Soll-Geometrie passt nicht (MY_GEAR unvollständig?)
    align_max = float(cfg.get("soll_align_max_dist", 5.0))  # [mm]
    if align_err > align_max:
        log("  WARN Alignment-Fehler zu gross vs Schwelle "
            "-- MY_GEAR-Parameter unvollstaendig oder z_shift falsch "
            "-- Phase 6 uebersprungen, Masken aus Phase 5 unveraendert",
            )
        log(f"  (align_err={align_err*1000:.0f}um  align_max={align_max*1000:.0f}um)")
        return dict(
            mask_R_ok=s5["mask_R_ok"], mask_L_ok=s5["mask_L_ok"],
            soll_available=False, phi0=phi0,
        )

    # Residual-Distanz für alle segmentierten Punkte — im gefalteten (Zahn-0)
    # Referenzrahmen. f_y_raw ist ein Skalarprodukt zweier gleich rotierter
    # Vektoren (delta, normal) → rotationsinvariant, identisch zum Ergebnis
    # im Weltrahmen, kein "Zurückdrehen" nötig.
    mask_seg   = s5["mask_R_ok"] | s5["mask_L_ok"]
    tooth_seg  = tooth_all[mask_seg]
    folded_seg = _fold_to_canonical(pts_b[mask_seg], tooth_seg, phi0)
    d_all, nn_idx = tree_soll.query(folded_seg, k=1, workers=-1)

    # Vorzeichenbehaftete Abweichung
    delta    = folded_seg - soll_pts[nn_idx]
    f_y_raw  = np.einsum('ij,ij->i', delta, soll_nrml[nn_idx])

    flag_thr = cfg["soll_dist_flag"]
    drop_thr = cfg["soll_dist_drop"]
    n_flag   = int((np.abs(f_y_raw) > flag_thr).sum())
    n_drop   = int((np.abs(f_y_raw) > drop_thr).sum())
    log(f"  |f_y_raw| > {flag_thr*1000:.0f}\u00b5m: {n_flag}  "
        f"|f_y_raw| > {drop_thr*1000:.0f}\u00b5m (verworfen): {n_drop}  "
        f"von {mask_seg.sum():,} Punkten", t0)

    # Maske mit verworfenen Punkten bereinigen
    drop_global = np.zeros(len(pts_b), dtype=bool)
    drop_global[np.where(mask_seg)[0][np.abs(f_y_raw) > drop_thr]] = True

    mask_R_clean = s5["mask_R_ok"] & ~drop_global
    mask_L_clean = s5["mask_L_ok"] & ~drop_global

    log(f"  Nach Bereinigung: R={mask_R_clean.sum():,}  L={mask_L_clean.sum():,}")

    return dict(
        mask_R_ok=mask_R_clean,
        mask_L_ok=mask_L_clean,
        soll_available=True,
        phi0=phi0,
        f_y_raw=f_y_raw,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 7 — Cluster bauen + HDF5-Ausgabe
# ═══════════════════════════════════════════════════════════════════════════════

def phase7_build_and_save(s1, s2, s3, s4, s5, s6, cfg):
    t0 = time.time()
    log("Phase 7: Cluster bauen + HDF5-Ausgabe")

    pts_xyz  = s3["pts_b_xyz"]
    pts_cyl  = s3["pts_b_cyl"]
    normals  = s3["normals"]
    n_teeth  = s2["n_teeth"]
    k_star   = s4["k_star"]
    r_b      = s4["r_b"]
    rg_min   = cfg["rg_min_pts"]

    # Finale Masken (nach Soll-Verifikation wenn verfügbar)
    mask_R_ok = s6["mask_R_ok"]
    mask_L_ok = s6["mask_L_ok"]
    tooth_all = s5["tooth_all"]

    # Optionales Sicherheitsnetz: robuster r-Perzentil-Trim je Cluster.
    edge_trim_pct = float(cfg.get("edge_trim_pct", 0.0))

    # d_lot-Vorberechnung: β_b aus Phase-4-Ergebnis
    beta_b_rad   = float(np.arctan(k_star * r_b))
    tooth_period = s2["tooth_period"]
    phi_b_R_all  = np.asarray(s5["phi_b_R"])
    phi_b_L_all  = np.asarray(s5["phi_b_L"])

    clusters = []
    for ti in range(n_teeth):
        for side, mask_ok, phi_b_all in [
            ("R", mask_R_ok, phi_b_R_all),
            ("L", mask_L_ok, phi_b_L_all),
        ]:
            mask = mask_ok & (tooth_all == ti)
            if edge_trim_pct > 0.0 and mask.sum() > 0:
                r_cl  = pts_cyl[mask, 0]
                lo, hi = np.percentile(r_cl, [edge_trim_pct, 100 - edge_trim_pct])
                mask = mask & (pts_cyl[:, 0] >= lo) & (pts_cyl[:, 0] <= hi)
            n_pts = int(mask.sum())
            if n_pts < rg_min:
                clusters.append(dict(tooth_id=ti, side=side,
                                      valid=False, n_pts=n_pts))
                continue

            # Analytische d_lot-Berechnung (kein MY_GEAR benötigt)
            phi_b_cl = phi_b_all[mask]
            d_lot_cl, phi_b_mean_cl = _dlot_analytical(
                phi_b_cl, r_b, beta_b_rad, side, tooth_period
            )
            d_lot_um = d_lot_cl * 1000.0  # mm → µm

            # Robuster Ausreisser-Filter fuer die ZUSAMMENFASSENDEN Kennwerte
            # (F_Sigma/F_beta/F_alpha/mean/std) -- NICHT fuer die gespeicherten
            # Rohpunkte, die bleiben vollstaendig erhalten (s. pts_xyz/pts_cyl/
            # normals/d_lot unten, ungefiltert). Grund + Validierung (166/166
            # Cluster, 2026-07-15): s. robust_mad_outlier_mask()-Docstring und
            # project_gear_pipeline.md -- die getroffenen Punkte zeigen
            # ueberwiegend die spezifische Zahnkopf/Flankenscheitel-Mehrdeutig-
            # keits-Signatur (nahe Zahnmitte + radiusnah am Kopf), kein
            # zufaelliges Rauschen, und betreffen <0.003% aller Punkte.
            outlier_mask = robust_mad_outlier_mask(d_lot_um)
            n_trim = int(outlier_mask.sum())
            d_lot_um_clean = d_lot_um[~outlier_mask]
            r_cl_clean = pts_cyl[mask, 0][~outlier_mask]
            z_cl_clean = pts_cyl[mask, 2][~outlier_mask]
            if n_trim > 0:
                log(f"    [Ausreisser-Filter] tooth_{ti:03d}_{side}: {n_trim}/{n_pts} "
                    f"Punkte robust getrimmt vor F_Σ/F_β/F_α-Berechnung "
                    f"({100*n_trim/n_pts:.3f}%)")

            # Grobe ISO-Proxies (Gesamt-Spannweite als obere Schranke)
            # F_Sigma: Gesamtspannweite d_lot — Proxy für Gesamtformabweichung
            F_sigma_um = float(np.ptp(d_lot_um_clean)) if len(d_lot_um_clean) else float('nan')
            # F_beta_proxy: d_lot-Spannweite nahe mittlerem r (Flankenlinienabweichung)
            r_mid = (r_cl_clean.min() + r_cl_clean.max()) / 2.0 if len(r_cl_clean) else 0.0
            r_hw  = (r_cl_clean.max() - r_cl_clean.min()) * 0.15 if len(r_cl_clean) else 0.0
            near_r = np.abs(r_cl_clean - r_mid) < r_hw
            F_beta_um = float(np.ptp(d_lot_um_clean[near_r])) if near_r.sum() > 10 else float('nan')
            # F_alpha_proxy: d_lot-Spannweite nahe mittlerem z (Profilabweichung)
            z_mid = (z_cl_clean.min() + z_cl_clean.max()) / 2.0 if len(z_cl_clean) else 0.0
            z_hw  = (z_cl_clean.max() - z_cl_clean.min()) * 0.15 if len(z_cl_clean) else 0.0
            near_z = np.abs(z_cl_clean - z_mid) < z_hw
            F_alpha_um = float(np.ptp(d_lot_um_clean[near_z])) if near_z.sum() > 10 else float('nan')

            clusters.append(dict(
                tooth_id=ti, side=side, valid=True, n_pts=n_pts,
                pts_xyz = pts_xyz[mask],
                pts_cyl = pts_cyl[mask],
                normals = normals[mask],
                d_lot   = d_lot_cl,        # [mm] -- VOLLSTAENDIG, ungefiltert
                d_lot_mean_um = float(np.mean(d_lot_um_clean)) if len(d_lot_um_clean) else float('nan'),
                d_lot_std_um  = float(np.std(d_lot_um_clean)) if len(d_lot_um_clean) else float('nan'),
                F_sigma_um    = F_sigma_um,
                F_beta_um     = F_beta_um,
                F_alpha_um    = F_alpha_um,
                n_pts_trimmed = n_trim,
            ))

    valid_n = sum(c['valid'] for c in clusters)
    pts_counts = [c['n_pts'] for c in clusters if c['valid']]
    log(f"  {valid_n}/{len(clusters)} Cluster valide  "
        f"(Median {int(np.median(pts_counts)) if pts_counts else 0} Punkte)")

    # HDF5-Ausgabe
    # Atomarer Schreibvorgang: erst in eine .tmp-Datei schreiben, dann per
    # os.replace() an den finalen Pfad verschieben (atomar auf demselben
    # Dateisystem). Verhindert, dass ein Absturz/Abbruch waehrend des
    # ~10s-Schreibvorgangs (166 Gruppen) SOWOHL die alte als auch die neue
    # Datei zerstoert -- vorher wurde die alte Datei per unlink() geloescht,
    # BEVOR die neue vollstaendig geschrieben war.
    out     = Path(cfg["output_hdf5"])
    out_tmp = out.with_suffix(out.suffix + ".tmp")
    out.parent.mkdir(exist_ok=True, parents=True)
    if out_tmp.exists():
        out_tmp.unlink()

    with h5py.File(out_tmp, "w") as f:
        meta = f.create_group("metadata")
        meta.attrs["N_teeth"]      = n_teeth
        meta.attrs["k_star"]       = k_star
        meta.attrs["k_star_ist"]   = k_star
        meta.attrs["r_b_ist"]      = r_b
        meta.attrs["beta_ist"]     = s4["beta_ist"]
        meta.attrs["alpha_t0_ist"] = s4["alpha_t0_ist"]
        meta.attrs["r_ref"]        = s2["r_ref"]
        meta.attrs["beta_deg"]     = s4["beta_ist"]
        meta.attrs["center"]       = s1["center"]
        meta.attrs["axis"]         = s1["axis"]
        meta.attrs["e1"]           = s1["e1"]
        meta.attrs["e2"]           = s1["e2"]
        meta.attrs["z_off"]        = s1["z_off"]
        meta.attrs["tooth_angles"] = s5["tooth_angles"]
        meta.attrs["r_b_R"]        = s4["r_b_R"]
        meta.attrs["r_b_L"]        = s4["r_b_L"]
        meta.attrs["soll_available"] = bool(s6.get("soll_available", False))
        if s6.get("soll_available", False) and "phi0" in s6:
            meta.attrs["soll_phi0"] = s6["phi0"]

        written = 0
        for c in clusters:
            if not c["valid"]:
                continue
            name = f"tooth_{c['tooth_id']:03d}_{c['side']}"
            grp  = f.create_group(name)
            grp.create_dataset("pts_xyz", data=c["pts_xyz"].astype(np.float32),
                               compression="gzip", chunks=True)
            grp.create_dataset("pts_cyl", data=c["pts_cyl"].astype(np.float32),
                               compression="gzip", chunks=True)
            grp.create_dataset("normals", data=c["normals"].astype(np.float32),
                               compression="gzip", chunks=True)
            grp.create_dataset("d_lot", data=c["d_lot"].astype(np.float32),
                               compression="gzip", chunks=True)
            grp.attrs["valid"]         = True
            grp.attrs["tooth_id"]      = c["tooth_id"]
            grp.attrs["side"]          = c["side"]
            grp.attrs["n_pts"]         = c["n_pts"]
            grp.attrs["d_lot_mean_um"] = c["d_lot_mean_um"]
            grp.attrs["d_lot_std_um"]  = c["d_lot_std_um"]
            grp.attrs["F_sigma_um"]    = c["F_sigma_um"]
            grp.attrs["F_beta_um"]     = c["F_beta_um"]
            grp.attrs["F_alpha_um"]    = c["F_alpha_um"]
            grp.attrs["n_pts_trimmed"] = c.get("n_pts_trimmed", 0)   # s. robust_mad_outlier_mask()
            written += 1

    # Nur jetzt, nach vollstaendigem + fehlerfreiem Schreiben, die alte Datei
    # ersetzen -- atomar (Path.replace() = os.replace()), kein Zwischenzustand
    # ohne gueltige clusters.h5.
    out_tmp.replace(out)
    log(f"  HDF5 gespeichert: {out}  ({written} Gruppen)", t0)
    return dict(clusters=clusters, n_valid=valid_n)

# ═══════════════════════════════════════════════════════════════════════════════
# Phase 8 — Visualisierung
# ═══════════════════════════════════════════════════════════════════════════════

def phase8_visualize(s1, s2, s3, s4, s5, s7, cfg):
    """
    Phase 8: Visualisierungsausgabe — 3 PNG-Dateien.

    fig1_nteeth_helix.png  — N-Erkennung, Phasenverfolgung, β-Vergleich
    fig2_segmentation.png  — Segmentierungskarte, per-Zahn-Statistik, φ_b-Cluster
    fig3_tooth_detail.png  — Exemplar-Zahn: Profil, Querschnitt, Formabweichung
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    t0 = time.time()
    log("Phase 8: Visualisierung")

    fig_dir = Path(cfg.get("output_fig_dir", "."))
    fig_dir.mkdir(exist_ok=True, parents=True)

    # ─── Basisdaten ───────────────────────────────────────────────────────────
    n_teeth      = int(s2["n_teeth"])
    tooth_period = float(s2["tooth_period"])
    k_star       = float(s4["k_star"])
    r_b          = float(s4["r_b"])
    beta_ist     = float(s4["beta_ist"])
    r_ref        = float(s2["r_ref"])
    k_fft        = float(s2["k_fft"])

    pts_cyl = s3["pts_b_cyl"].astype(np.float64)   # (M,3): r, θ, z
    r_arr   = pts_cyl[:, 0]
    theta   = pts_cyl[:, 1]
    z_arr   = pts_cyl[:, 2]

    # Helix-korrigierter Winkel + tc_rel (neu berechnet aus s4/s5)
    tc           = (theta - k_star * z_arr) % (2.0 * np.pi)
    tooth_angles = np.asarray(s5["tooth_angles"])
    tooth_all    = np.asarray(s5["tooth_all"])
    ta_per_pt    = tooth_angles[tooth_all]
    tc_rel       = ((tc - ta_per_pt + np.pi) % (2.0 * np.pi)) - np.pi

    mask_R  = np.asarray(s5["mask_R_ok"])
    mask_L  = np.asarray(s5["mask_L_ok"])
    mask_RL = mask_R | mask_L
    phi_b_R = np.asarray(s5["phi_b_R"])
    phi_b_L = np.asarray(s5["phi_b_L"])

    RNG = np.random.default_rng(42)

    def _take(mask, n=25_000):
        idx = np.where(mask)[0]
        return idx if len(idx) <= n else idx[RNG.choice(len(idx), n, replace=False)]

    def _hist_peak(vals, n_bins=300):
        """Histogramm-Peak-Mitte in normiertem [0,1]-Raum."""
        h, e = np.histogram(vals, bins=n_bins, range=(0.0, 1.0))
        pk = int(np.argmax(h))
        return float((e[pk] + e[pk + 1]) / 2.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Figur 1: N-Erkennung + Schrägungswinkel
    # ═══════════════════════════════════════════════════════════════════════════
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))
    fig1.suptitle(
        f"N={n_teeth} Zähne  |  β_ist={beta_ist:.4f}°  |  "
        f"r_b={r_b:.3f}mm  |  k_star={k_star:.7f}",
        fontsize=11, fontweight='bold'
    )

    # 1A — tc mod Teilung: alle N Zähne in EINEM Peak → Helix-Korrektur korrekt
    ax = axes1[0]
    tc_mod_norm = (tc / tooth_period) % 1.0   # [0,1] — ein Wert pro Teilungsperiode
    ax.hist(tc_mod_norm, bins=800, color='steelblue', alpha=0.75, density=True)
    ax.set_xlabel('tc mod Teilung  (normiert [0,1])')
    ax.set_ylabel('Dichte')
    ax.set_title(
        f'N={n_teeth}: helix-korr. tc = θ − k·z  (mod Teilung)\n'
        f'Ein scharfer Peak → alle {n_teeth} Zähne korrekt überlagert'
    )
    ax.grid(True, alpha=0.3)

    # 1B — Phasenverfolgung (z-Scheiben)
    ax = axes1[1]
    z_sl   = s2.get("pt_z_slices")
    ph_m   = s2.get("pt_ph_measured")
    ph_fit = s2.get("pt_ph_fit")
    beta_pt = float('nan')

    if z_sl is not None:
        z_sl    = np.asarray(z_sl)
        ph_m_n  = np.asarray(ph_m)   / n_teeth   # Phase / N → Steigung = k [rad/mm]
        ph_f_n  = np.asarray(ph_fit) / n_teeth
        ref     = ph_m_n[0]
        slope_k = float(np.polyfit(z_sl, ph_m_n, 1)[0])   # k [rad/mm]
        beta_pt = float(np.degrees(np.arctan(slope_k * r_ref)))
        ax.scatter(z_sl, np.degrees(ph_m_n - ref), s=35, c='steelblue',
                   zorder=3, label='Gemessen')
        ax.plot(z_sl, np.degrees(ph_f_n - ref), 'r-', lw=1.5, label='Fit')
        ax.text(0.04, 0.97,
                f'Steigung = {np.degrees(slope_k):.5f} °/mm\n'
                f'k_pt = {slope_k:.7f} rad/mm\n'
                f'β_pt = {beta_pt:.4f}°',
                transform=ax.transAxes, va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, 'Phasenverfolgung\nnicht verfügbar',
                ha='center', va='center', transform=ax.transAxes, fontsize=10)

    ax.set_xlabel('z [mm]')
    ax.set_ylabel('Helix-Phase  [°]  (relativ zu z_min)')
    ax.set_title('Phasenverfolgung  →  k_star\n(Steigung der Phasendrift pro z-Scheibe)')
    ax.grid(True, alpha=0.3)

    # 1C — β-Vergleich aller Methoden (inkl. Ring-XKorr)
    ax = axes1[2]
    beta_fft_v  = float(np.degrees(np.arctan(k_fft * r_ref)))
    _k_xcorr    = s2.get("k_xcorr")
    beta_xcorr  = float(np.degrees(np.arctan(_k_xcorr * r_ref))) if _k_xcorr is not None else float('nan')
    # Ring-XKorr (s. _kstar_ring_xcorr()-Docstring) hat einen dokumentierten,
    # ungelösten systematischen Bias (R/L-Vermischung, da diese Methode in
    # Phase 2 läuft, VOR der R/L-Trennung in Phase 4) und fließt nirgends in
    # k_init/k_star ein -- reine Referenz. Frueher stand der Balken unmarkiert
    # neben den drei tatsaechlich genutzten/uebereinstimmenden Methoden, was
    # in einem Bericht wie ein vierter, gleichwertiger Messwert aussah statt
    # wie eine bewusst verworfene Kontrollrechnung. Jetzt: eigene Schraffur +
    # Beschriftung, UND y-Achsen-Zoom relativ zu den drei VERTRAUTEN Werten
    # (sonst zieht der xcorr-Ausreisser den Zoom so weit, dass die enge
    # Uebereinstimmung der anderen drei nicht mehr sichtbar ist).
    labels3 = [
        'Phasen-\nverfolgung\n(lokal)',
        'Ring-XKorr\n(bekannt verzerrt,\ns. Text)',
        'FFT-Rayleigh\n(Phase 2, global)',
        'Joint-Opt. R\n(Phase 4, best)',
    ]
    vals3 = [beta_pt, beta_xcorr, beta_fft_v, beta_ist]
    clrs3 = ['#4292c6', '#b0a3c2', '#2ca25f', '#e34a33']
    hatch3 = [None, '///', None, None]
    bars3 = ax.bar(labels3, vals3, color=clrs3, alpha=0.85, edgecolor='k', lw=0.5)
    for bar, h in zip(bars3, hatch3):
        if h:
            bar.set_hatch(h)
    y_trusted = [v for v, lbl in zip(vals3, labels3) if np.isfinite(v) and 'xcorr' not in lbl.lower() and 'XKorr' not in lbl]
    if y_trusted:
        ax.set_ylim(min(y_trusted) - 0.25, max(y_trusted) + 0.35)
    for bar, val in zip(bars3, vals3):
        if np.isfinite(val):
            y_txt = min(val, ax.get_ylim()[1] - 0.05) if val > ax.get_ylim()[1] else val
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y_txt + 0.01,
                    f'{val:.4f}°', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('β  [°]')
    ax.set_title(
        f'Schrägungswinkel-Schätzungen\n(alle bei r_ref = {r_ref:.2f}mm; '
        f'Ring-XKorr zu Referenzzwecken gezeigt, nicht in k_star verwendet)',
        fontsize=9
    )
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='x', labelsize=7)

    fig1.tight_layout()
    p1 = fig_dir / "fig1_nteeth_helix.png"
    fig1.savefig(p1, dpi=150, bbox_inches='tight')
    plt.close(fig1)
    log(f"  {p1.name}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figur 2: Segmentierungsübersicht
    # ═══════════════════════════════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(18, 10))
    gs2  = GridSpec(2, 2, figure=fig2, hspace=0.38, wspace=0.30)

    # 2A — "DNA"-Streifen: tc vs z, R/L/Rauschen farbig
    ax2a = fig2.add_subplot(gs2[0, 0])
    tc_full = tc / (2.0 * np.pi) * n_teeth   # [0, n_teeth]: ein Zahn = 1 Einheit
    idxR = _take(mask_R, 25_000)
    idxL = _take(mask_L, 25_000)
    idxN = _take(~mask_RL, 5_000)
    ax2a.scatter(tc_full[idxN], z_arr[idxN], s=0.3, c='lightgray',
                 alpha=0.25, rasterized=True)
    ax2a.scatter(tc_full[idxR], z_arr[idxR], s=0.5, c='#e34a33',
                 alpha=0.5, rasterized=True, label='R-Flanke')
    ax2a.scatter(tc_full[idxL], z_arr[idxL], s=0.5, c='#2b8cbe',
                 alpha=0.5, rasterized=True, label='L-Flanke')
    ax2a.set_xlabel('tc / Teilung  (Zahn-Index + Offset  [0 … N])')
    ax2a.set_ylabel('z [mm]')
    ax2a.set_title(
        f'Segmentierungskarte  R={mask_R.sum():,}  L={mask_L.sum():,}\n'
        f'(vertikale Streifen = Zähne; R/L innerhalb jedes Streifens)'
    )
    leg = ax2a.legend(fontsize=7, markerscale=6, loc='upper right')
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    ax2a.set_xlim(0, n_teeth)

    # 2B — per-Zahn Punktanzahl R+L
    ax2b = fig2.add_subplot(gs2[0, 1])
    r_cnt = np.bincount(tooth_all[mask_R], minlength=n_teeth).astype(float)
    l_cnt = np.bincount(tooth_all[mask_L], minlength=n_teeth).astype(float)
    x = np.arange(n_teeth)
    w = 0.45
    ax2b.bar(x - w / 2, r_cnt, w, color='#e34a33', alpha=0.8,
             label=f'R  ∅{r_cnt.mean():.0f}')
    ax2b.bar(x + w / 2, l_cnt, w, color='#2b8cbe', alpha=0.8,
             label=f'L  ∅{l_cnt.mean():.0f}')
    ratio_rl = r_cnt.mean() / max(l_cnt.mean(), 1.0)
    ax2b.set_xlabel('Zahn-Index')
    ax2b.set_ylabel('Punkte pro Zahn')
    ax2b.set_title(
        f'Punktanzahl R vs L  (R/L≈{ratio_rl:.1f})\n'
        f'Gleichmäßige Kurve → Scanner-Geometrie, kein Klassifikationsfehler'
    )
    ax2b.legend(fontsize=8)
    ax2b.set_xlim(-1, n_teeth)
    ax2b.grid(True, alpha=0.3, axis='y')

    # 2C — φ_b Histogramm: Evolventenkonstanz R vs L
    ax2c = fig2.add_subplot(gs2[1, 0])
    eps      = float(cfg.get('dbscan_eps', 0.05))
    bins_phi = np.linspace(0.0, 1.0, 300)

    if mask_R.sum() > 0:
        phR = (phi_b_R[mask_R] / tooth_period) % 1.0
        phR = phR[RNG.choice(len(phR), min(len(phR), 200_000), replace=False)]
        ax2c.hist(phR, bins=bins_phi, color='#e34a33', alpha=0.60,
                  density=True, label=f'R  ({mask_R.sum():,})')
        pk = _hist_peak(phR)
        ax2c.axvspan(max(0.0, pk - eps), min(1.0, pk + eps),
                     alpha=0.12, color='#e34a33')

    if mask_L.sum() > 0:
        phL = (phi_b_L[mask_L] / tooth_period) % 1.0
        phL = phL[RNG.choice(len(phL), min(len(phL), 200_000), replace=False)]
        ax2c.hist(phL, bins=bins_phi, color='#2b8cbe', alpha=0.60,
                  density=True, label=f'L  ({mask_L.sum():,})')
        pk = _hist_peak(phL)
        ax2c.axvspan(max(0.0, pk - eps), min(1.0, pk + eps),
                     alpha=0.12, color='#2b8cbe')

    ax2c.set_xlabel('φ_b mod Teilung  (normiert [0,1])')
    ax2c.set_ylabel('Dichte')
    ax2c.set_title(
        f'φ_b-Cluster-Qualität  (ε-Fenster={eps})\n'
        f'Schärfe R ≫ L → R besser gescannt (Scanner-Geometrie)'
    )
    ax2c.legend(fontsize=8)
    ax2c.grid(True, alpha=0.3)

    # 2D — tc_rel Histogramm: L/R-Vorfilter
    ax2d = fig2.add_subplot(gs2[1, 1])
    pre_frac   = float(cfg.get('dbscan_tc_pre_frac', 0.45))
    period_deg = float(np.degrees(tooth_period))
    cutoff_deg = pre_frac * period_deg
    bins_tc    = np.linspace(-period_deg / 2, period_deg / 2, 200)

    idx_sub = RNG.choice(len(tc_rel), min(150_000, len(tc_rel)), replace=False)
    ax2d.hist(np.degrees(tc_rel[idx_sub]), bins=bins_tc,
              color='lightgray', alpha=0.70, density=True, label='Alle Bandpkt.')
    ax2d.hist(np.degrees(tc_rel[_take(mask_R, 50_000)]), bins=bins_tc,
              color='#e34a33', alpha=0.55, density=True, label='R-Inlier')
    ax2d.hist(np.degrees(tc_rel[_take(mask_L, 50_000)]), bins=bins_tc,
              color='#2b8cbe', alpha=0.55, density=True, label='L-Inlier')
    ax2d.axvline(+cutoff_deg, c='k', ls='--', lw=1,
                 label=f'±{pre_frac:.0%} Teilung (Vorfilter)')
    ax2d.axvline(-cutoff_deg, c='k', ls='--', lw=1)
    ax2d.axvline(0, c='gray', ls=':', lw=0.8)
    ax2d.set_xlabel('tc_rel  [°]')
    ax2d.set_ylabel('Dichte')
    ax2d.set_title('tc_rel-Verteilung\ntc_rel > 0 = R-Flanke, tc_rel < 0 = L-Flanke  (by Construction)')
    ax2d.legend(fontsize=7)
    ax2d.grid(True, alpha=0.3)

    fig2.suptitle(
        f'Segmentierungsübersicht  |  R: {mask_R.sum():,}  L: {mask_L.sum():,}  '
        f'Noise: {(~mask_RL).sum():,}  |  {n_teeth} Zähne',
        fontsize=11, fontweight='bold'
    )
    p2 = fig_dir / "fig2_segmentation.png"
    fig2.savefig(p2, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    log(f"  {p2.name}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Figur 3: Exemplar-Zahn (Zahn mit maximaler L-Abdeckung)
    # ═══════════════════════════════════════════════════════════════════════════
    l_per_tooth = np.bincount(tooth_all[mask_L], minlength=n_teeth)
    ti = int(np.argmax(l_per_tooth))

    mR = mask_R & (tooth_all == ti)
    mL = mask_L & (tooth_all == ti)

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle(
        f'Exemplar-Zahn {ti}  |  R: {mR.sum():,} Pkt  L: {mL.sum():,} Pkt\n'
        f'(Zahn mit maximaler L-Abdeckung — repräsentativ für Analyse-Qualität)',
        fontsize=11, fontweight='bold'
    )

    # 3A — r vs z: Schräghelix + beide Flanken
    ax = axes3[0]
    if mR.sum():
        ax.scatter(z_arr[mR], r_arr[mR], s=0.8, c='#e34a33', alpha=0.4,
                   rasterized=True, label='R-Flanke')
    if mL.sum():
        ax.scatter(z_arr[mL], r_arr[mL], s=0.8, c='#2b8cbe', alpha=0.4,
                   rasterized=True, label='L-Flanke')
    ax.axhline(r_b, c='k', ls=':', lw=1, label=f'r_b = {r_b:.3f}mm')
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('r [mm]')
    ax.set_title(f'Zahn {ti}: Radialprofil über z\n(Schräghelix → r variiert mit tc_rel(z))')
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(True, alpha=0.3)

    # 3B — tc_rel vs r: Involuten-Querschnitt
    ax = axes3[1]
    if mR.sum():
        ax.scatter(np.degrees(tc_rel[mR]), r_arr[mR], s=0.8, c='#e34a33',
                   alpha=0.4, rasterized=True, label='R-Flanke')
    if mL.sum():
        ax.scatter(np.degrees(tc_rel[mL]), r_arr[mL], s=0.8, c='#2b8cbe',
                   alpha=0.4, rasterized=True, label='L-Flanke')
    ax.axhline(r_b, c='k', ls=':', lw=1, label=f'r_b = {r_b:.3f}mm')
    ax.axvline(0, c='gray', ls='--', lw=0.6)
    ax.set_xlabel('tc_rel  [°]  (Zahnmitte = 0)')
    ax.set_ylabel('r [mm]')
    ax.set_title(f'Zahn {ti}: Evolventenprofil im Querschnitt\n(R und L divergieren mit wachsendem r)')
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(True, alpha=0.3)

    # 3C — d_lot vs z: metrisch korrekte Formabweichung (ISO 1328-1 kompatibel)
    # d_lot = r_b·cos(β_b)·Δφ_b  — exakter als Δφ_b×r_b (fehlender cos(β_b) ≈7% bei β=22°)
    ax = axes3[2]

    def _plot_dlot(phi_vals, z_vals, side, clr, lbl):
        if len(phi_vals) < 50:
            return
        d_lot_cl, _ = _dlot_analytical(phi_vals, r_b, np.arctan(k_star * r_b),
                                        side, tooth_period)
        d_lot_um = d_lot_cl * 1000.0
        ax.scatter(z_vals, d_lot_um, s=0.8, c=clr, alpha=0.35,
                   rasterized=True, label=lbl)

    if mR.sum():
        _plot_dlot(phi_b_R[mR], z_arr[mR], 'R', '#e34a33', 'R-Flanke')
    if mL.sum():
        _plot_dlot(phi_b_L[mL], z_arr[mL], 'L', '#2b8cbe', 'L-Flanke')

    ax.axhline(0, c='k', lw=0.8)
    ax.set_xlabel('z [mm]')
    ax.set_ylabel('d_lot  [µm]  (Lot-Abstand zur Idealevolvente)')
    ax.set_title(
        f'Zahn {ti}: d_lot-Formabweichung vs z\n'
        f'(d_lot = r_b·cos(β_b)·Δφ_b; 0 = ideal-Evolvente; '
        f'Trend = Flankenlinienabweichung)'
    )
    ax.legend(fontsize=8, markerscale=4)
    ax.grid(True, alpha=0.3)

    fig3.tight_layout()
    p3 = fig_dir / "fig3_tooth_detail.png"
    fig3.savefig(p3, dpi=150, bbox_inches='tight')
    plt.close(fig3)
    log(f"  {p3.name}", t0)

    # ═══════════════════════════════════════════════════════════════════════════
    # Figur 4: d_lot-Übersicht pro Zahn (F_Σ, F_β*, ⟨d_lot⟩)
    # ═══════════════════════════════════════════════════════════════════════════
    valid_cl = [c for c in s7["clusters"] if c.get("valid", False)
                and "F_sigma_um" in c]
    if valid_cl:
        fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
        fig4.suptitle(
            f'd_lot-Abweichungs-Übersicht  |  {len(valid_cl)} valide Cluster  |  '
            f'N={n_teeth} Zähne  r_b={r_b:.3f}mm  β={beta_ist:.4f}°',
            fontsize=11, fontweight='bold'
        )

        # 4A — F_Sigma pro Zahn × Seite
        ax = axes4[0]
        t_ids_R = [c['tooth_id'] for c in valid_cl if c['side'] == 'R']
        F_sig_R = [c['F_sigma_um'] for c in valid_cl if c['side'] == 'R']
        t_ids_L = [c['tooth_id'] for c in valid_cl if c['side'] == 'L']
        F_sig_L = [c['F_sigma_um'] for c in valid_cl if c['side'] == 'L']
        if t_ids_R:
            ax.bar(t_ids_R, F_sig_R, width=0.4, color='#e34a33', alpha=0.7,
                   align='edge', label='R-Flanke')
        if t_ids_L:
            ax.bar(t_ids_L, F_sig_L, width=-0.4, color='#2b8cbe', alpha=0.7,
                   align='edge', label='L-Flanke')
        ax.set_xlabel('Zahn-Index')
        ax.set_ylabel('F_Σ  [µm]  (Gesamtspannweite d_lot)')
        ax.set_title('Formabweichung F_Σ pro Zahn\n(Proxy: max–min d_lot)')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        # 4B — F_beta_proxy pro Zahn × Seite
        ax = axes4[1]
        Fb_R = [c['F_beta_um'] for c in valid_cl if c['side'] == 'R']
        Fb_L = [c['F_beta_um'] for c in valid_cl if c['side'] == 'L']
        if t_ids_R:
            ax.bar(t_ids_R, Fb_R, width=0.4, color='#e34a33', alpha=0.7,
                   align='edge', label='R-Flanke')
        if t_ids_L:
            ax.bar(t_ids_L, Fb_L, width=-0.4, color='#2b8cbe', alpha=0.7,
                   align='edge', label='L-Flanke')
        ax.set_xlabel('Zahn-Index')
        ax.set_ylabel('F_β*  [µm]  (d_lot-Spannweite bei mittlerem r)')
        ax.set_title('Flankenlinienabweichungs-Proxy F_β*\n(z-Trace bei r_mid ± 15%)')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        # 4C — d_lot Mittelwert pro Zahn
        ax = axes4[2]
        dm_R = [c['d_lot_mean_um'] for c in valid_cl if c['side'] == 'R']
        dm_L = [c['d_lot_mean_um'] for c in valid_cl if c['side'] == 'L']
        if t_ids_R:
            ax.bar(t_ids_R, dm_R, width=0.4, color='#e34a33', alpha=0.7,
                   align='edge', label='R-Flanke')
        if t_ids_L:
            ax.bar(t_ids_L, dm_L, width=-0.4, color='#2b8cbe', alpha=0.7,
                   align='edge', label='L-Flanke')
        ax.axhline(0, c='k', lw=0.8)
        ax.set_xlabel('Zahn-Index')
        ax.set_ylabel('⟨d_lot⟩  [µm]  (Mittelwert pro Zahn)')
        ax.set_title('Zahn-Mittelwert d_lot (Rundlauf-Proxy)\n'
                     'Sinusform → Exzentrizität / Taumelschlag')
        ax.legend(fontsize=8)
        ax.grid(True, axis='y', alpha=0.3)

        fig4.tight_layout()
        p4 = fig_dir / "fig4_dlot_summary.png"
        fig4.savefig(p4, dpi=150, bbox_inches='tight')
        plt.close(fig4)
        log(f"  {p4.name}", t0)

        # Log-Zusammenfassung
        all_F_sig  = [c['F_sigma_um'] for c in valid_cl]
        all_d_mean = [abs(c['d_lot_mean_um']) for c in valid_cl]
        log(f"  d_lot-Statistik: F_Σ_median={np.nanmedian(all_F_sig):.1f}µm  "
            f"max={np.nanmax(all_F_sig):.1f}µm  "
            f"|⟨d_lot⟩|_max={np.nanmax(all_d_mean):.1f}µm")


# ═══════════════════════════════════════════════════════════════════════════════
# Pipeline-Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_pipeline(cfg=None):
    """Fuehrt alle Phasen in Sequenz aus."""
    if cfg is None:
        cfg = dict(CFG)

    t_total = time.time()
    sep = "=" * 70
    log(sep)
    log("gear_pipeline_ist.py -- Start")
    log(sep)

    s1 = phase1_load_and_axis(cfg)
    s2 = phase2_rough_helix_and_teeth(s1, cfg)
    s3 = phase3_normals_and_lr(s1, s2, cfg)
    s4 = phase4_optimize_k_rb(s1, s2, s3, cfg)

    # Phase-5-Dispatcher: DBSCAN (empfohlen) -> Region Growing (L/R-Split) ->
    # Region Growing (klassisch) -> phi_b
    if cfg.get("use_dbscan_flank", False):
        s5 = phase5_dbscan_flank(s1, s2, s3, s4, cfg)
    elif cfg.get("use_region_growing_lr_split", False):
        s5 = phase5_region_growing_lr_split(s1, s2, s3, s4, cfg)
    elif cfg.get("use_region_growing", False):
        s5 = phase5_region_growing(s1, s2, s3, s4, cfg)
    else:
        s5 = phase5_tooth_assignment(s1, s2, s3, s4, cfg)

    s5b = phase5b_scan_profile_crosscheck(s1, s2, s3, s4, s5, cfg)

    # Phase 5b-Ergebnisse in s5 uebernehmen
    # WICHTIG: tooth_angles MUSS zusammen mit tooth_all uebernommen werden --
    # beide werden von phase5b ueber DIESELBE (unabhaengig symmetrisierte)
    # off_tc-Referenz gebildet. phase5_dbscan_flank hat seine EIGENE, leicht
    # andere off_tc-Schaetzung (andere Methode: phi_b-Dichte-Peak statt
    # tc-Profil-FFT-Phase) -- wuerde man hier nur tooth_all uebernehmen und
    # das alte s5["tooth_angles"] (phi_b-Referenz) stehen lassen, entstuende
    # ein Offset zwischen den zwei Referenzen (~2 deg beobachtet), der sich
    # in phase8_visualize als verschobenes tc_rel / Zahnzentrum zeigt.
    s5["mask_R_ok"]    = s5b["mask_R_ok"]
    s5["mask_L_ok"]    = s5b["mask_L_ok"]
    s5["tooth_all"]    = s5b["tooth_all"]
    s5["tooth_angles"] = s5b["tooth_angles"]
    s5["tooth_R"]      = s5b["tooth_all"]
    s5["tooth_L"]      = s5b["tooth_all"]

    if cfg.get("use_dbscan_flank", False):
        log("Phase 5c: uebersprungen (DBSCAN -- Ausreisser bereits als Noise klassifiziert)")
    elif cfg.get("use_region_growing_lr_split", False) or cfg.get("use_region_growing", False):
        log("Phase 5c: uebersprungen (Region Growing -- GZK-Pruefung waere no-op)")
    else:
        s5c = phase5c_spatial_coherence(s2, s3, s5b, cfg)
        s5["mask_R_ok"] = s5c["mask_R_ok"]
        s5["mask_L_ok"] = s5c["mask_L_ok"]
        s5["tooth_all"] = s5c["tooth_all"]
        s5["tooth_R"]   = s5c["tooth_all"]
        s5["tooth_L"]   = s5c["tooth_all"]

    s6 = phase6_soll_verify(s1, s2, s3, s4, s5, cfg)
    s7 = phase7_build_and_save(s1, s2, s3, s4, s5, s6, cfg)
    phase8_visualize(s1, s2, s3, s4, s5, s7, cfg)

    elapsed = time.time() - t_total
    log(f"\nPipeline abgeschlossen  ({elapsed:.1f}s gesamt)")
    log(sep)


if __name__ == "__main__":
    run_pipeline()
