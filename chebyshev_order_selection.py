"""
chebyshev_order_selection.py — Einmalige Ordnungs-Validierung
================================================================
Beantwortet: welche Chebyshev-Ordnung (CHEB_ORDER in config.py) ist
für die gesamte Pipeline angemessen — nicht als Suche, die bei jedem
Lauf/pro Cluster neu entschieden wird, sondern EINMAL bestimmt und
dann für alle 166 Cluster konsistent verwendet (Vergleichbarkeit über
alle Zähne hinweg, siehe Diskussion zu CV-Fragilität bei großem N).

Methode:
  1. Repräsentative Stichprobe von Clustern aus clusters.h5 (verschiedene
     Zahn-Indizes, beide Seiten).
  2. Für jede Kandidaten-Ordnung: wiederholter Train/Test-Split (80/20)
     PRO Cluster → gepoolte Held-out-RMSE über (Cluster × Wiederholung).
  3. Trainings-RMSE (optimistisch, sinkt monoton) vs. Held-out-RMSE
     (U-förmig, der eigentliche Indikator) werden beide geplottet.
  4. "1-SE-Regel": kleinste Ordnung, deren Held-out-RMSE innerhalb einer
     Standardabweichung des beobachteten Minimums liegt — robuster als
     der exakte Minimierer, besonders wenn die Kurve bei großem N flach
     wird (siehe Projekt-Diskussion).
  5. Rauschgrenze (Default 5µm, aus phase4_uv.py-Diagnose übernommen)
     als harte Obergrenze: nie über die Ordnung hinausgehen, die schon
     nahe an die Rauschgrenze herankommt, selbst wenn Held-out-RMSE
     technisch noch (marginal) weiter sinkt.

Ergebnis: eine EMPFEHLUNG für CHEB_ORDER, mit Beleg-Plot. Kein
automatischer Suchlauf, der bei jeder Pipeline-Ausführung erneut
entscheidet — das würde 166 potenziell unterschiedliche Ordnungen
erzeugen und Ist-Kennwerte über Zähne hinweg unvergleichbar machen.

Nutzung: einmal ausführen, Empfehlung in config.py (CHEB_ORDER) eintragen.
"""

import numpy as np
import h5py
import time
from pathlib import Path

from gear_soll import compute_derived_geometry, MY_GEAR
from gear_pipeline_phase4_uv import cluster_to_uv_fy
from chebyshev import fit as cheb_fit, evaluate as cheb_eval
from config import HDF5, FIGURES_DIR
from utils import log


CFG = dict(
    input_hdf5   = str(HDF5),
    output_fig_dir = str(FIGURES_DIR),
    order_range  = list(range(2, 9)),   # Kandidaten-Ordnungen 2..8
    n_sample_clusters = 8,              # über Zahn-Index + Seite verteilt
    n_repeats    = 15,                  # Train/Test-Wiederholungen pro Cluster×Ordnung
    test_frac    = 0.2,
    noise_floor_um = 5.0,               # aus phase4_uv.py-Diagnose ("ideal: <10µm, 5µm Rauschen")
    uv_margin    = 0.05,
    min_pts      = 200,                 # Cluster mit weniger Punkten werden übersprungen
    random_seed  = 0,
)


def _pick_representative_clusters(hdf5_path, n_sample):
    """Wählt Cluster über den Zahn-Index-Bereich verteilt, abwechselnd R/L."""
    with h5py.File(hdf5_path, "r") as f:
        keys = sorted(k for k in f.keys() if k.startswith("tooth_")
                      and bool(f[k].attrs.get("valid", False)))
    if not keys:
        raise RuntimeError(
            "Keine gültigen Cluster in clusters.h5 gefunden — "
            "gear_pipeline_ist.py zuerst ausführen."
        )
    idx = np.linspace(0, len(keys) - 1, min(n_sample, len(keys))).astype(int)
    return [keys[i] for i in sorted(set(idx))]


def _get_uvfy_for_cluster(hdf5_file, key, derived, soll_params, k_star, tooth_angles):
    g = hdf5_file[key]
    pts_cyl = g["pts_cyl"][:].astype(np.float64)
    side = key.split("_")[-1]
    ti = int(g.attrs["tooth_id"])
    tc_ist = float(tooth_angles[ti])
    uvfy = cluster_to_uv_fy(pts_cyl, derived, soll_params, k_star, side, tc_ist)
    return uvfy["u"].astype(np.float64), uvfy["v"].astype(np.float64), uvfy["f_y"].astype(np.float64)


def run(cfg=None):
    cfg = cfg or CFG
    t0 = time.time()
    log("Chebyshev-Ordnungsvalidierung — Start")

    derived = compute_derived_geometry(MY_GEAR)
    rng = np.random.default_rng(cfg["random_seed"])

    cluster_keys = _pick_representative_clusters(cfg["input_hdf5"], cfg["n_sample_clusters"])
    log(f"  Stichprobe: {len(cluster_keys)} Cluster: {cluster_keys}")

    with h5py.File(cfg["input_hdf5"], "r") as f:
        meta = f["metadata"]
        k_star = float(meta.attrs["k_star_ist"])
        tooth_angles = np.array(meta.attrs["tooth_angles"])

        cluster_data = []
        for key in cluster_keys:
            u, v, f_y = _get_uvfy_for_cluster(f, key, derived, MY_GEAR, k_star, tooth_angles)
            margin = cfg["uv_margin"]
            mask = (np.abs(u) <= 1 + margin) & (np.abs(v) <= 1 + margin)
            if mask.sum() < cfg["min_pts"]:
                log(f"  ⚠ {key}: nur {mask.sum()} Punkte im Bereich — übersprungen")
                continue
            cluster_data.append(dict(key=key, u=u[mask], v=v[mask], f_y=f_y[mask]))

    if not cluster_data:
        log("  Keine verwertbaren Cluster — Abbruch.")
        return

    orders = cfg["order_range"]
    test_frac = cfg["test_frac"]
    n_repeats = cfg["n_repeats"]

    # train_rms[order] = Liste aller (Cluster × Wiederholung) Trainings-RMS
    # test_rms[order]  = Liste aller (Cluster × Wiederholung) Held-out-RMS
    train_rms = {o: [] for o in orders}
    test_rms  = {o: [] for o in orders}

    for cd in cluster_data:
        n_pts = len(cd["u"])
        for order in orders:
            n_coeffs = (order + 1) ** 2
            if n_pts < n_coeffs * 4:
                continue  # zu wenig Punkte für einen stabilen Test-Split bei dieser Ordnung
            for _rep in range(n_repeats):
                perm = rng.permutation(n_pts)
                n_test = max(int(n_pts * test_frac), n_coeffs)
                test_idx  = perm[:n_test]
                train_idx = perm[n_test:]

                train_mask_full = np.zeros(n_pts, dtype=bool)
                train_mask_full[train_idx] = True

                coeffs, rms_train, _ = cheb_fit(
                    cd["u"], cd["v"], cd["f_y"], order, mask=train_mask_full
                )
                pred_test = cheb_eval(coeffs, cd["u"][test_idx], cd["v"][test_idx])
                resid_test = cd["f_y"][test_idx] - pred_test
                rms_test = float(np.sqrt(np.mean(resid_test ** 2)))

                train_rms[order].append(rms_train)
                test_rms[order].append(rms_test)

        log(f"  {cd['key']}: verarbeitet ({n_pts:,} Punkte im Fit-Bereich)")

    # ── Aggregation + 1-SE-Regel ──────────────────────────────────────────────
    orders_valid = [o for o in orders if test_rms[o]]
    mean_train = np.array([np.mean(train_rms[o]) for o in orders_valid]) * 1000  # µm
    mean_test  = np.array([np.mean(test_rms[o])  for o in orders_valid]) * 1000
    se_test    = np.array([np.std(test_rms[o]) / np.sqrt(len(test_rms[o]))
                            for o in orders_valid]) * 1000

    i_min = int(np.argmin(mean_test))
    threshold = mean_test[i_min] + se_test[i_min]
    i_1se = next(i for i in range(len(orders_valid)) if mean_test[i] <= threshold)
    order_1se = orders_valid[i_1se]

    noise_floor = cfg["noise_floor_um"]
    i_noise = next((i for i in range(len(orders_valid)) if mean_test[i] <= noise_floor),
                    len(orders_valid) - 1)
    order_noise = orders_valid[i_noise]

    order_recommended = min(order_1se, order_noise)

    log(f"\n  Held-out-RMSE je Ordnung [µm]:")
    for i, o in enumerate(orders_valid):
        flag = ""
        if o == order_1se:
            flag += "  <- 1-SE-Minimum"
        if o == order_noise:
            flag += "  <- erreicht Rauschgrenze"
        log(f"    Ordnung {o}: train={mean_train[i]:6.2f}  "
            f"test={mean_test[i]:6.2f} ± {se_test[i]:.2f}{flag}")

    log(f"\n  Empfehlung: CHEB_ORDER = {order_recommended}  "
        f"(1-SE-Minimum={order_1se}, Rauschgrenze-Ordnung={order_noise}, "
        f"kleinere von beiden gewählt)")
    log(f"  → In config.py eintragen: CHEB_ORDER = {order_recommended}", t0)

    _plot(orders_valid, mean_train, mean_test, se_test,
          order_1se, order_noise, order_recommended, noise_floor, cfg)

    return dict(orders=orders_valid, train_rms_um=mean_train, test_rms_um=mean_test,
                se_test_um=se_test, order_1se=order_1se, order_noise=order_noise,
                order_recommended=order_recommended)


def _plot(orders, mean_train, mean_test, se_test, order_1se, order_noise,
          order_recommended, noise_floor, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(orders, mean_train, 'o--', color='#185FA5', label='Trainings-RMSE (optimistisch)')
    ax.errorbar(orders, mean_test, yerr=se_test, fmt='o-', color='#993C1D',
                capsize=4, label='Held-out-RMSE ± 1 SE')
    ax.axhline(noise_floor, color='k', ls=':', lw=1.5,
               label=f'Rauschgrenze ({noise_floor:.1f}µm)')
    ax.axvline(order_1se, color='#993C1D', ls='--', lw=1, alpha=0.6,
               label=f'1-SE-Minimum (Ordnung {order_1se})')
    ax.axvline(order_noise, color='k', ls='--', lw=1, alpha=0.4,
               label=f'Rauschgrenze erreicht (Ordnung {order_noise})')
    ax.axvline(order_recommended, color='#1D9E75', lw=2.5, alpha=0.85,
               label=f'Empfehlung: Ordnung {order_recommended}')
    ax.set_xlabel("Chebyshev-Ordnung (je Achse)")
    ax.set_ylabel("RMSE [µm]")
    ax.set_title("Chebyshev-Ordnungsvalidierung\n"
                 "(Train vs. Held-out, gepoolt über Stichprobe repräsentativer Cluster)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    Path(cfg["output_fig_dir"]).mkdir(exist_ok=True, parents=True)
    out = Path(cfg["output_fig_dir"]) / "chebyshev_order_selection.png"
    plt.savefig(out, dpi=120, bbox_inches='tight')
    plt.close()
    log(f"  → {out}")


if __name__ == "__main__":
    run()
