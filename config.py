"""
Pipeline-Konfiguration
======================
Alle Pfade und globalen Einstellungen an einem Ort.
Nur diese Datei muss angepasst werden.

Hinweis: Alle Pfade sind absolute Pfade.
ROOT dient nur als Fallback für relative Layouts.
"""

from pathlib import Path

# ── Verzeichnis-Wurzel ────────────────────────────────────────────────────────
# Automatisch: Ordner in dem diese config.py liegt.
# Wird als Fallback verwendet wenn keine absoluten Pfade gesetzt sind.
ROOT = Path(__file__).parent

# ── Eingabe ───────────────────────────────────────────────────────────────────
# Punktwolke liegt eine Ebene oberhalb des gear_analysisv3-Ordners.
PUNKTWOLKE   = ROOT.parent / "Z83.txt"

# ── Zwischenergebnis ──────────────────────────────────────────────────────────
HDF5         = ROOT / "Output Files" / "clusters.h5"

# ── Ausgaben ──────────────────────────────────────────────────────────────────
CSV          = ROOT / "Output Files" / "Kennwerte" / "din_kennwerte.csv"
FIGURES_DIR  = ROOT / "Output Files" / "Figures"

# ── Chebyshev-Ordnung ─────────────────────────────────────────────────────────
# 4 → 5×5 Matrix (25 Koeffizienten). Höher = feinere Formerfassung,
# aber mehr Rechenzeit und Überanpassungsrisiko. Empfehlung: 4–6.
CHEB_ORDER   = 4
