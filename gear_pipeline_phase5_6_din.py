"""
Zahnrad-Pipeline — Phase 5 + 6: Chebyshev-Fit + DIN-Vergleich
===============================================================
Phase 5: Least-squares Fit eines 2D-Chebyshev-Polynoms an die f_y(u,v) Daten
         jedes Ist-Clusters. Ergebnis: c_ij_ist (5×5 Koeffizienten-Matrix).

Phase 6: Vergleich Δc_ij = c_ij_ist - c_ij_soll, Berechnung der DIN-1328
         Kennwerte und CSV-Output.

DIN-Kennwerte (Paper II Abschnitt 3, vereinfacht):
  Profilballigkeit       C_α_ist = -2·c_20_ist          (mm)
  Flankenballigkeit      C_β_ist = -2·c_02_ist          (mm)
  Flankenwinkel-Mod.     f_Hβ_ist = +2·c_01_ist          (mm)
  Profilwinkel-Mod.      f_Hα_ist = +2·c_10_ist          (mm)
  Konstanter Offset      Δr      = -c_00_ist             (mm)

  Gesamt-Profilabweichung   Fα = max(f_y, u-Richtung) - min  über v=const
  Gesamt-Flankenlinienabw.  Fβ = max(f_y, v-Richtung) - min  über u=const
"""

import numpy as np
import h5py, time
import csv
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")

from gear_soll import compute_derived_geometry
from config import HDF5, CSV, FIGURES_DIR, CHEB_ORDER
from gear_soll import MY_GEAR
from chebyshev import evaluate as cheb_eval, din_from_coeffs
from utils import log


CFG = dict(
    input_hdf5     = str(HDF5),
    output_fig_dir = str(FIGURES_DIR),
    output_csv     = str(CSV),
    cheb_order     = CHEB_ORDER, # Max-Ordnung = 4 → 5×5 Matrix (0..4)
    eval_grid_n    = 30,       # Auswertungsgitter für Fα/Fβ Bestimmung
)


# ─────────────────────────────────────────────────────────────────────────────
# Chebyshev-Fit
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Chebyshev + DIN: aus chebyshev.py importiert (cheb_eval, din_from_coeffs)
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# HAUPTFUNKTION
# ─────────────────────────────────────────────────────────────────────────────

def process_all(cfg, soll_params):
    t0 = time.time()
    print("\n" + "="*62)
    print("  PHASE 5 + 6: DIN-Kennwerte aus PCA-Chebyshev-Fit")
    print("="*62)

    log("Lese c_ist, c_soll, Δc aus Phase 4 (Ist-PCA-Frame)")

    with h5py.File(cfg["input_hdf5"], "r+") as f:
        # c_ist/c_soll kommen aus Phase 4, EIN Fit über dem daten-getriebenen
        # (tatsächlich erreichten) l_y-Bereich. c_soll ist dort bereits per
        # exakter affiner Basisänderung (remap_u_domain) aus der echten
        # Eingriffsstrecke in denselben Bereich umgerechnet worden — Ist und
        # Soll sind also auf demselben u,v-Bezugsbereich normiert, ein
        # gültiger Δc-Vergleich (siehe gear_pipeline_phase4_uv.py's
        # process_all_clusters()-Docstring).
        order = int(f['metadata'].attrs.get('cheb_order', 4))
        n_grid = cfg['eval_grid_n']
        log(f"Chebyshev-Ordnung: {order} ({order+1}×{order+1})")

        results = []
        for key in sorted(f.keys()):
            if not key.startswith('tooth_'): continue
            g = f[key]
            if not bool(g.attrs.get('phase4_valid', False)): continue
            if 'phase4_c_ist' not in g: continue

            c_ist  = g['phase4_c_ist'][:].astype(np.float64)
            c_soll = g['phase4_c_soll'][:].astype(np.float64)
            dc     = c_ist - c_soll
            rms    = float(g.attrs['phase4_rms_mm'])

            # Δc auf Auswertungsgitter
            u_g = np.linspace(-1, 1, n_grid)
            v_g = np.linspace(-1, 1, n_grid)
            U, V = np.meshgrid(u_g, v_g, indexing='ij')
            fy_delta = cheb_eval(dc, U, V)
            fy_delta = fy_delta.reshape(n_grid, n_grid)

            # DIN-Kennwerte direkt aus Δc-Gitter (in mm)
            # ── Fα / fHα / ffα ──────────────────────────────────────
            F_alpha = float((fy_delta.max(axis=0) - fy_delta.min(axis=0)).max())
            fy_u_mean = fy_delta.mean(axis=1)
            fHa_slope = float(np.polyfit(u_g, fy_u_mean, 1)[0])
            f_H_alpha_din = 2.0 * fHa_slope
            ff_alpha = max(F_alpha - abs(f_H_alpha_din), 0.0)

            # ── Fβ / fHβ / ffβ ──────────────────────────────────────
            F_beta = float((fy_delta.max(axis=1) - fy_delta.min(axis=1)).max())
            fy_v_mean = fy_delta.mean(axis=0)
            fHb_slope = float(np.polyfit(v_g, fy_v_mean, 1)[0])
            f_H_beta_din = 2.0 * fHb_slope
            ff_beta = max(F_beta - abs(f_H_beta_din), 0.0)

            # ── Balligkeiten aus Δc-Koeffizienten ───────────────────
            # Für PCA-Koordinaten: c[2,0] ist parabolischer Anteil in u-Richtung
            # In der PCA hat u physikalische Einheit mm → c[2,0] in mm
            # Umrechnung auf µm-Balligkeit:
            # u_range ist die physikalische Breite in mm
            # C_α ≈ -2 · c[2,0] (gilt für normierte Tschebyscheff wie in analytischer Formel)
            C_alpha_ist = float(-2 * c_ist[2, 0]) if c_ist.shape[0] > 2 else 0.0
            C_beta_ist  = float(-2 * c_ist[0, 2]) if c_ist.shape[1] > 2 else 0.0
            f_H_beta_ist = float(2 * c_ist[0, 1]) if c_ist.shape[1] > 1 else 0.0
            f_H_alpha_ist = float(2 * c_ist[1, 0]) if c_ist.shape[0] > 1 else 0.0

            f_y_rms = float(np.sqrt(np.mean(fy_delta**2)))

            kw = dict(
                F_alpha_mm=F_alpha, ff_alpha_mm=ff_alpha,
                f_H_alpha_din_mm=f_H_alpha_din,
                F_beta_mm=F_beta, ff_beta_mm=ff_beta,
                f_H_beta_din_mm=f_H_beta_din,
                f_y_rms_mm=f_y_rms,
                C_alpha_ist_mm=C_alpha_ist,
                C_beta_ist_mm=C_beta_ist,
                f_H_beta_ist_mm=f_H_beta_ist,
                f_H_alpha_ist_mm=f_H_alpha_ist,
                delta_r_mm=float(dc[0, 0]),
                C_alpha_soll_mm=float(-2*c_soll[2,0]) if c_soll.shape[0]>2 else 0.0,
                C_beta_soll_mm=float(-2*c_soll[0,2]) if c_soll.shape[1]>2 else 0.0,
                f_H_alpha_soll_mm=float(2*c_soll[1,0]) if c_soll.shape[0]>1 else 0.0,
                f_H_beta_soll_mm=float(2*c_soll[0,1]) if c_soll.shape[1]>1 else 0.0,
            )

            for ds in ['phase5_c_ist', 'phase6_delta_c', 'phase6_fy_delta']:
                if ds in g: del g[ds]
            g.create_dataset('phase5_c_ist',    data=c_ist.astype(np.float32))
            g.create_dataset('phase6_delta_c',  data=dc.astype(np.float32))
            g.create_dataset('phase6_fy_delta', data=fy_delta.astype(np.float32))
            g.attrs['phase5_rms_mm'] = rms
            for k, val in kw.items():
                g.attrs[f'phase6_{k}'] = val

            results.append(dict(
                key=key,
                tooth=int(g.attrs['tooth_id']),
                side=g.attrs['side'].decode() if isinstance(g.attrs['side'], bytes)
                     else g.attrs['side'],
                rms=rms, c_ist=c_ist, c_soll=c_soll, **kw,
            ))

        log(f"Verarbeitet: {len(results)} Cluster", t0)
        if results:
            log(f"  RMS median: {np.median([r['rms'] for r in results])*1000:.2f} µm")

    # Reales c_soll zurückgeben (identisches analytisches Modell für alle
    # Cluster) statt des früheren Platzhalter-Nullarrays — wird von
    # visualize_final_comparison() für den Ist/Soll/Δ-Vergleich gebraucht.
    c_soll_out = results[0]['c_soll'] if results else np.zeros((order + 1, order + 1))
    return results, c_soll_out


# ─────────────────────────────────────────────────────────────────────────────
# CSV-Output
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(results, cfg):
    """Schreibt vollständige DIN-Kennwerte als CSV pro Cluster."""
    out = Path(cfg["output_csv"])
    out.parent.mkdir(exist_ok=True, parents=True)

    fields = ['tooth', 'side', 'rms_um',
              # Profil
              'F_alpha_um', 'ff_alpha_um', 'f_H_alpha_um',
              # Flankenlinie
              'F_beta_um',  'ff_beta_um',  'f_H_beta_um',
              # Balligkeit
              'C_alpha_ist_um', 'C_beta_ist_um',
              # Winkelmodifikation (aus Koeffizienten)
              'f_H_alpha_ist_um', 'f_H_beta_ist_um',
              # Pitch-Offset
              'delta_r_um',
              # RMS Abweichung
              'f_y_rms_um',
              # Soll-Referenz
              'C_alpha_soll_um', 'C_beta_soll_um',
              'f_H_alpha_soll_um', 'f_H_beta_soll_um']

    with open(out, 'w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=fields)
        writer.writeheader()
        for r in results:
            row = {
                'tooth':              r['tooth'],
                'side':               r['side'],
                'rms_um':             f"{r['rms']*1000:.2f}",
                # Profil
                'F_alpha_um':         f"{r['F_alpha_mm']*1000:.2f}",
                'ff_alpha_um':        f"{r['ff_alpha_mm']*1000:.2f}",
                'f_H_alpha_um':       f"{r['f_H_alpha_din_mm']*1000:.2f}",
                # Flankenlinie
                'F_beta_um':          f"{r['F_beta_mm']*1000:.2f}",
                'ff_beta_um':         f"{r['ff_beta_mm']*1000:.2f}",
                'f_H_beta_um':        f"{r['f_H_beta_din_mm']*1000:.2f}",
                # Balligkeit
                'C_alpha_ist_um':     f"{r['C_alpha_ist_mm']*1000:.2f}",
                'C_beta_ist_um':      f"{r['C_beta_ist_mm']*1000:.2f}",
                # Winkelmodifikation
                'f_H_alpha_ist_um':   f"{r['f_H_alpha_ist_mm']*1000:.2f}",
                'f_H_beta_ist_um':    f"{r['f_H_beta_ist_mm']*1000:.2f}",
                # Pitch
                'delta_r_um':         f"{r['delta_r_mm']*1000:.2f}",
                # RMS
                'f_y_rms_um':         f"{r['f_y_rms_mm']*1000:.2f}",
                # Soll
                'C_alpha_soll_um':    f"{r['C_alpha_soll_mm']*1000:.2f}",
                'C_beta_soll_um':     f"{r['C_beta_soll_mm']*1000:.2f}",
                'f_H_alpha_soll_um':  f"{r['f_H_alpha_soll_mm']*1000:.2f}",
                'f_H_beta_soll_um':   f"{r['f_H_beta_soll_mm']*1000:.2f}",
            }
            writer.writerow(row)
    print(f"  → {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Visualisierung
# ─────────────────────────────────────────────────────────────────────────────

def visualize(results, c_soll, cfg):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t0 = time.time()
    print("\nVisualisiere Phase 5+6 ...")
    if not results:
        print("  Keine Ergebnisse"); return

    fig = plt.figure(figsize=(17, 12))
    gs  = gridspec.GridSpec(3, 4, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle(
        f"Phase 5+6 — Chebyshev-Fit + DIN-Vergleich  ({len(results)} Cluster)",
        fontsize=12, fontweight="bold"
    )

    # 1: RMS pro Cluster
    ax = fig.add_subplot(gs[0, :2])
    keys = [r['key'][6:] for r in results]
    rms_um = [r['rms']*1000 for r in results]
    x = np.arange(len(results))
    ax.bar(x, rms_um, color='#185FA5', alpha=0.85)
    step = max(1, len(results)//20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([keys[i] for i in range(0, len(results), step)],
                        rotation=60, fontsize=7)
    ax.set_ylabel("Fit-RMS [µm]")
    ax.set_title("Chebyshev-Fit Residuum je Cluster")
    ax.grid(alpha=0.3, axis='y')

    # 2: F_alpha und F_beta je Zahn
    ax = fig.add_subplot(gs[0, 2:])
    teeth = sorted(set(r['tooth'] for r in results))
    Fa_R = []; Fa_L = []; Fb_R = []; Fb_L = []
    for ti in teeth:
        rR = next((r for r in results if r['tooth']==ti and r['side']=='R'), None)
        rL = next((r for r in results if r['tooth']==ti and r['side']=='L'), None)
        Fa_R.append(rR['F_alpha_mm']*1000 if rR else 0)
        Fa_L.append(rL['F_alpha_mm']*1000 if rL else 0)
        Fb_R.append(rR['F_beta_mm']*1000 if rR else 0)
        Fb_L.append(rL['F_beta_mm']*1000 if rL else 0)
    ax.plot(teeth, Fa_R, 'o-', color='#1D9E75', label='Fα R', markersize=5)
    ax.plot(teeth, Fa_L, 's-', color='#534AB7', label='Fα L', markersize=5)
    ax.plot(teeth, Fb_R, 'o--', color='#993C1D', label='Fβ R', alpha=0.7, markersize=4)
    ax.plot(teeth, Fb_L, 's--', color='#BA7517', label='Fβ L', alpha=0.7, markersize=4)
    ax.set_xlabel("Zahn-Index"); ax.set_ylabel("[µm]")
    ax.set_title("Fα und Fβ je Zahn × Seite")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # 3-5: 3 Beispiel-Δf_y(u,v) Heatmaps
    example_idx = [0, len(results)//2, len(results)-1]
    for i, idx in enumerate(example_idx):
        if idx >= len(results): continue
        r = results[idx]
        ax = fig.add_subplot(gs[1, i])
        # Lade fy_delta aus HDF5
        with h5py.File(cfg["input_hdf5"], "r") as f:
            fy_delta = f[r['key']]['phase6_fy_delta'][:] * 1000  # µm
        im = ax.imshow(fy_delta.T, origin='lower', extent=[-1, 1, -1, 1],
                        cmap='RdBu_r', vmin=-30, vmax=30, aspect='auto')
        ax.set_xlabel("u (Profil)"); ax.set_ylabel("v (Flankenlinie)")
        ax.set_title(f"{r['key']}\nFα={r['F_alpha_mm']*1000:.1f}µm  Fβ={r['F_beta_mm']*1000:.1f}µm",
                      fontsize=9)
        plt.colorbar(im, ax=ax, label="Δf_y [µm]")

    # 6: c_ij Matrix Median über alle Cluster
    ax = fig.add_subplot(gs[1, 3])
    c_mean = np.mean([r['c_ist'] for r in results], axis=0) * 1000  # µm
    im = ax.imshow(c_mean, cmap='RdBu_r', vmin=-15, vmax=15, aspect='auto')
    ax.set_xticks(range(c_mean.shape[1])); ax.set_yticks(range(c_mean.shape[0]))
    ax.set_xlabel("j (v-Ordnung)"); ax.set_ylabel("i (u-Ordnung)")
    ax.set_title("Mittlere c_ij_ist Matrix [µm]")
    for i in range(c_mean.shape[0]):
        for j in range(c_mean.shape[1]):
            ax.text(j, i, f"{c_mean[i,j]:+.1f}", ha='center', va='center',
                     fontsize=7,
                     color='white' if abs(c_mean[i,j]) > 7 else 'black')
    plt.colorbar(im, ax=ax)

    # 7-8: C_α und C_β (Ist vs Soll) je Cluster
    ax = fig.add_subplot(gs[2, 0])
    Ca_ist = [r['C_alpha_ist_mm']*1000 for r in results]
    ax.hist(Ca_ist, bins=20, color='#185FA5', alpha=0.7)
    ax.axvline(results[0]['C_alpha_soll_mm']*1000, color='#993C1D', ls='--',
                lw=2, label=f"Soll: {results[0]['C_alpha_soll_mm']*1000:.1f}µm")
    ax.axvline(np.mean(Ca_ist), color='#1D9E75', ls=':', lw=2,
                label=f"Ist Mittel: {np.mean(Ca_ist):.1f}µm")
    ax.set_xlabel("C_α [µm]"); ax.set_ylabel("Cluster-Anzahl")
    ax.set_title("Profilballigkeit C_α (Ist vs Soll)")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2, 1])
    Cb_ist = [r['C_beta_ist_mm']*1000 for r in results]
    ax.hist(Cb_ist, bins=20, color='#185FA5', alpha=0.7)
    ax.axvline(results[0]['C_beta_soll_mm']*1000, color='#993C1D', ls='--',
                lw=2, label=f"Soll: {results[0]['C_beta_soll_mm']*1000:.1f}µm")
    ax.axvline(np.mean(Cb_ist), color='#1D9E75', ls=':', lw=2,
                label=f"Ist Mittel: {np.mean(Cb_ist):.1f}µm")
    ax.set_xlabel("C_β [µm]"); ax.set_ylabel("Cluster-Anzahl")
    ax.set_title("Flankenballigkeit C_β (Ist vs Soll)")
    ax.legend(fontsize=8)

    # 9-10: Zusammenfassungs-Text
    ax = fig.add_subplot(gs[2, 2:]); ax.axis('off')
    Fa_med = np.median([r['F_alpha_mm']*1000 for r in results])
    Fb_med = np.median([r['F_beta_mm']*1000 for r in results])
    rms_med = np.median([r['rms']*1000 for r in results])
    Ca_med = np.median(Ca_ist)
    Cb_med = np.median(Cb_ist)
    fHb_med = np.median([r['f_H_beta_ist_mm']*1000 for r in results])

    txt = (
        "═══════════════════════════════════════════\n"
        "  DIN-KENNWERTE — ÜBERSICHT (Median)\n"
        "═══════════════════════════════════════════\n\n"
        "Fit-Qualität:\n"
        f"  Cheb-Fit RMS         : {rms_med:8.2f} µm\n\n"
        "Profilrichtung (u-Achse):\n"
        f"  Profilballigkeit C_α : {Ca_med:+8.2f} µm  (Soll {results[0]['C_alpha_soll_mm']*1000:+.1f})\n"
        f"  Profilabweichung Fα  : {Fa_med:8.2f} µm\n\n"
        "Flankenlinienrichtung (v-Achse):\n"
        f"  Flankenballigkeit C_β: {Cb_med:+8.2f} µm  (Soll {results[0]['C_beta_soll_mm']*1000:+.1f})\n"
        f"  Flankenwinkel f_Hβ   : {fHb_med:+8.2f} µm  (Soll {results[0]['f_H_beta_soll_mm']*1000:+.1f})\n"
        f"  Flankenabweichung Fβ : {Fb_med:8.2f} µm\n\n"
        f"═══════════════════════════════════════════\n"
        f"  Detaillierter CSV-Output:\n"
        f"  {Path(cfg['output_csv']).name}\n"
        f"═══════════════════════════════════════════"
    )
    ax.text(0.0, 0.98, txt, family='monospace', fontsize=9.5,
            va='top', ha='left')

    out = Path(cfg["output_fig_dir"]) / "phase5_6_chebyshev_din.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"→ {out}", t0)


# ─────────────────────────────────────────────────────────────────────────────
# Finaler Vergleich: Ist/Soll/Δ-Flächen + überlagerte Residuen-Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def visualize_final_comparison(results, cfg):
    """
    Finale Vergleichs-Abbildung:
      (C) Ist / Soll / Δ (Ist−Soll) Flächen-Heatmaps für den Cluster mit
          bester Fit-Qualität (kleinste RMS) — direkter visueller
          Soll-Ist-Vergleich auf demselben (u,v)-Gitter.
      (D) Überlagerte Mikro-Residuen-Heatmap: (u,v,Residuum) aller
          gültigen Cluster auf ein gemeinsames Gitter gepoolt/gemittelt.
          Ein systematisches Muster (z.B. Werkzeug-/Prozess-Signatur), das
          in einem Einzel-Cluster im Rauschen untergeht, wird über 166
          gemittelte Flanken hinweg sichtbar — echtes Rauschen mittelt
          sich dagegen zu ~0 heraus.
    """
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    t0 = time.time()
    print("\nVisualisiere finalen Ist/Soll-Vergleich + Residuen-Overlay ...")
    if not results:
        print("  Keine Ergebnisse"); return

    grid_n = 60
    u_g = np.linspace(-1, 1, grid_n)
    v_g = np.linspace(-1, 1, grid_n)
    U, V = np.meshgrid(u_g, v_g, indexing='ij')

    # Repräsentativer Cluster: beste Fit-Güte statt willkürlich erster Eintrag
    r_best = min(results, key=lambda r: r['rms'])
    ist_surf   = cheb_eval(r_best['c_ist'],  U, V) * 1000  # µm
    soll_surf  = cheb_eval(r_best['c_soll'], U, V) * 1000
    delta_surf = ist_surf - soll_surf

    fig = plt.figure(figsize=(17, 9))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle(
        f"Finaler Ist/Soll-Vergleich  (Beispiel: {r_best['key']}, RMS={r_best['rms']*1000:.1f}µm)"
        f"  +  Überlagerte Mikro-Residuen ({len(results)} Cluster)",
        fontsize=12, fontweight="bold")

    lim_is = max(np.percentile(np.abs(ist_surf), 99),
                 np.percentile(np.abs(soll_surf), 99), 1e-3)
    lim_d  = max(np.percentile(np.abs(delta_surf), 99), 1e-3)
    for i, (title, surf, lim) in enumerate([
        ("Ist",           ist_surf,   lim_is),
        ("Soll",          soll_surf,  lim_is),
        ("Δ (Ist−Soll)",  delta_surf, lim_d),
    ]):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(surf.T, origin='lower', extent=[-1, 1, -1, 1],
                        cmap='RdBu_r', vmin=-lim, vmax=lim, aspect='auto')
        ax.set_xlabel("u (Profil)"); ax.set_ylabel("v (Flankenlinie)")
        ax.set_title(title, fontsize=10)
        plt.colorbar(im, ax=ax, label="f_y [µm]")

    # (D) Überlagerte Residuen — alle Cluster gepoolt auf gemeinsames Gitter
    with h5py.File(cfg["input_hdf5"], "r") as f:
        u_all, v_all, res_all = [], [], []
        for r in results:
            g = f[r['key']]
            if 'phase4_u' not in g:
                continue
            u_all.append(g['phase4_u'][:])
            v_all.append(g['phase4_v'][:])
            res_all.append(g['phase4_residuals'][:])

    if u_all:
        u_all   = np.concatenate(u_all)
        v_all   = np.concatenate(v_all)
        res_all = np.concatenate(res_all) * 1000  # µm

        n_bins  = 40
        u_edges = np.linspace(-1, 1, n_bins + 1)
        v_edges = np.linspace(-1, 1, n_bins + 1)
        u_idx = np.clip(np.digitize(u_all, u_edges) - 1, 0, n_bins - 1)
        v_idx = np.clip(np.digitize(v_all, v_edges) - 1, 0, n_bins - 1)
        sum_grid = np.zeros((n_bins, n_bins))
        cnt_grid = np.zeros((n_bins, n_bins))
        np.add.at(sum_grid, (u_idx, v_idx), res_all)
        np.add.at(cnt_grid, (u_idx, v_idx), 1)
        mean_grid = np.where(cnt_grid > 0, sum_grid / np.maximum(cnt_grid, 1), np.nan)

        ax = fig.add_subplot(gs[1, 0])
        lim_r = np.nanpercentile(np.abs(mean_grid), 99)
        lim_r = lim_r if np.isfinite(lim_r) and lim_r > 0 else 1.0
        im = ax.imshow(mean_grid.T, origin='lower', extent=[-1, 1, -1, 1],
                        cmap='RdBu_r', vmin=-lim_r, vmax=lim_r, aspect='auto')
        ax.set_xlabel("u"); ax.set_ylabel("v")
        ax.set_title(f"Überlagerte Mikro-Residuen\n(Mittelwert über {len(results)} Cluster je Bin)",
                     fontsize=9)
        plt.colorbar(im, ax=ax, label="Residuum [µm]")

        ax = fig.add_subplot(gs[1, 1])
        im = ax.imshow(cnt_grid.T, origin='lower', extent=[-1, 1, -1, 1],
                        cmap='viridis', aspect='auto')
        ax.set_xlabel("u"); ax.set_ylabel("v")
        ax.set_title("Punktdichte je Bin (Abdeckung)", fontsize=9)
        plt.colorbar(im, ax=ax, label="Punkte/Bin")

        ax = fig.add_subplot(gs[1, 2])
        # Gleicher Grund wie beim identischen Panel in gear_pipeline_phase4_uv.py:
        # phase4_residuals enthaelt ALLE mask_fit-Punkte inkl. der wenigen
        # absichtlich beibehaltenen Kopf-/Fuss-Ambiguitaets-Ausreisser, was den
        # Auto-Achsenbereich irrefuehrend breit macht, obwohl das Std klein ist.
        # Auf ±6σ clippen und Ausreisser-Anzahl annotieren statt Vollbereich zeigen.
        std_res = float(np.std(res_all))
        lim = 6.0 * std_res if std_res > 1e-9 else max(float(np.max(np.abs(res_all))), 1e-3)
        n_out = int(np.sum(np.abs(res_all) > lim))
        ax.hist(res_all, bins=100, range=(-lim, lim), color='#1D9E75', alpha=0.8)
        ax.set_xlim(-lim, lim)
        ax.set_xlabel("Residuum [µm]"); ax.set_ylabel("Anzahl")
        title = f"Alle Residuen gepoolt (±6σ)\nStd={std_res:.2f}µm  N={len(res_all):,}"
        if n_out:
            title += f"\n{n_out} Ausreisser außerhalb ±6σ nicht dargestellt"
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3)
    else:
        ax = fig.add_subplot(gs[1, :]); ax.axis('off')
        ax.text(0.5, 0.5, "Keine phase4_u/v/residuals Daten gefunden "
                           "(gear_pipeline_phase4_uv.py zuerst ausführen)",
                ha='center')

    Path(cfg["output_fig_dir"]).mkdir(exist_ok=True, parents=True)
    out = Path(cfg["output_fig_dir"]) / "phase6_final_comparison.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"→ {out}", t0)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_all = time.time()
    results, c_soll = process_all(CFG, MY_GEAR)
    save_csv(results, CFG)
    visualize(results, c_soll, CFG)
    visualize_final_comparison(results, CFG)
    print(f"\nFERTIG — {time.time()-t_all:.1f}s")


if __name__ == "__main__":
    main()