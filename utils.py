"""
utils.py — gemeinsame Hilfsfunktionen für die Zahnrad-Pipeline
"""

import time
import numpy as np


def log(msg: str, t0: float | None = None) -> None:
    """
    Gibt eine Zeile auf stdout aus (flush, damit Jupyter/Terminal sofort sieht).
    Optionaler t0 (time.time()-Wert) fügt die vergangene Zeit an.
    """
    if t0 is not None:
        dt = time.time() - t0
        print(f"{msg}  [{dt:.1f}s]", flush=True)
    else:
        print(msg, flush=True)


def robust_mad_outlier_mask(values, k_mad: float = 8.0):
    """
    Robuster (MAD-basierter) Ausreisser-Filter.

    Markiert Punkte, die weiter als k_mad * 1.4826 * MAD vom Median entfernt
    liegen. MAD (Median Absolute Deviation) hat einen Bruchpunkt von 50% --
    robust gegenueber genau der kleinen Ausreisser-Fraktion (typischerweise
    << 1% pro Cluster), die hier gefiltert werden soll -- im Gegensatz zu
    mean/std, die von denselben Ausreissern bereits verzerrt waeren
    (zirkulaeres Problem: man kann Ausreisser nicht zuverlaessig ueber eine
    Statistik finden, die die Ausreisser selbst schon verzerrt haben).

    K_MAD=8.0 ist NICHT auf einen Einzelfall zugeschnitten -- gewaehlt aus
    dem unabhaengig bekannten Scan-Rauschboden (~15µm, synthetic_validation.py)
    VOR jeder Pruefung an echten Daten, dann ueber alle 166 Cluster eines
    realen Z83-Zahnrads via audit_outlier_trim.py validiert (2026-07-15):
    trifft nur 14.5% der Cluster, 0.0029% aller Punkte, 0 Cluster ueber 1%
    getrimmt. Die getroffenen Punkte zeigten ueberwiegend (~90%, 17/24
    betroffene Cluster) eine spezifische, unabhaengig bestaetigte
    geometrische Signatur (Zahnkopf/Flankenscheitel-Mehrdeutigkeit: nahe
    Zahnmitte in tc_rel UND radiusnah am Zahnkopf) statt zufaellig verteilt
    zu sein -- s. project_gear_pipeline.md (Memory) fuer die volle Herleitung.

    Parameter
    ---------
    values : (N,) array-like -- die zu pruefende Groesse (z.B. d_lot in µm)
    k_mad  : float -- Schwellenwert-Multiplikator (Default 8.0)

    Rueckgabe
    ---------
    outlier_mask : (N,) bool ndarray -- True = Ausreisser (sollte von
                   zusammenfassenden Kennwerten wie F_Sigma/RMS/Chebyshev-Fit
                   ausgeschlossen werden). Die Rohdaten selbst werden NICHT
                   veraendert/geloescht -- nur die Kennwert-Berechnung nutzt
                   die Maske.
    """
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    robust_std = mad * 1.4826
    if robust_std < 1e-9:
        return np.zeros(len(values), dtype=bool)
    thresh = k_mad * robust_std
    return np.abs(values - med) > thresh
