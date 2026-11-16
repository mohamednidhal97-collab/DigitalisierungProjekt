"""
Soll-Evolvente-Generator für Schrägverzahnung
==============================================
Basis: PTB Papers I + II (Härtig & Stein, 2019/2022)

Hauptfunktion: generate_soll_flank(params, n_u, n_v, tooth_i, side)
"""

import numpy as np
from dataclasses import dataclass

_nutzkreise_printed = False   # Verhindert wiederholten Print bei 83×2 Flanken-Aufrufen


@dataclass
class GearParameters:
    """
    Soll-Geometrie-Eingangsparameter.
    Alle Längen in MILLIMETERN, alle Winkel in GRAD.
    Modifikationen (C_α, C_β, f_Hβ) in MIKROMETERN [µm].
    """
    n: int                # Zähnezahl
    m_n: float            # Normalmodul [mm]
    alpha_n_deg: float    # Normaldruckwinkel [°]
    beta_0_deg: float     # Schrägungswinkel [°]
    b: float              # Zahnbreite [mm]
    x: float              # Profilverschiebungsfaktor [-]

    d_a: float            # Kopfkreisdurchmesser [mm]
    d_f: float            # Fußkreisdurchmesser [mm]
    d_0: float            # Teilkreisdurchmesser [mm]
    d_b: float            # Grundkreisdurchmesser [mm]

    d_Ff: float = 0.0     # Nutzkreis Kopf [mm] — 0 = automatisch schätzen
    d_Nf: float = 0.0     # Nutzkreis Fuß  [mm] — 0 = automatisch schätzen

    C_alpha: float = 0.0  # Profilballigkeit [µm]
    C_beta: float  = 0.0  # Flankenballigkeit / Helix Crowning [µm]
    f_H_alpha: float = 0.0 # Profilwinkel-Modifikation / Profile Slope Modification [µm]
    f_H_beta: float = 0.0 # Flankenlinien-Winkelmodifikation / Helix Angle Correction [µm]

    hand: int = +1        # +1 rechtssteigend, -1 linkssteigend
    type_gear: int = +1   # +1 außenverzahnt, -1 innenverzahnt


def estimate_nutzkreise(n, m_n, alpha_n_deg, beta_0_deg, x=0.0,
                         n_partner=None, x_partner=0.0):
    """
    Schätzt d_Ff (Nutzkreis Kopf) und d_Nf (Nutzkreis Fuß).

    Reihenfolge der Kreise von innen nach außen:
      d_b < d_f < d_Nf < d_0 < d_Ff < d_a

    d_Nf: wo die aktive Evolvente am Fuß beginnt.
          Hängt vom Kopfkreis des Gegenzahns ab.
          Näherung: knapp über Grundkreis.

    d_Ff: wo die aktive Evolvente am Kopf endet.
          Näherung: knapp unter Kopfkreis.
    """
    alpha_n = np.radians(alpha_n_deg)
    beta_0  = np.radians(beta_0_deg)
    m_t     = m_n / np.cos(beta_0)

    r_b = n * m_n / (2 * np.sqrt(np.tan(alpha_n)**2 + np.cos(beta_0)**2))
    r_0 = n * m_t / 2
    r_a = r_0 + (1.0 + x) * m_n

    # Nutzkreis Kopf: etwas unter Kopfkreis (Randeffekte vermeiden)
    r_Ff = r_a - 0.05 * m_n

    # Nutzkreis Fuß: knapp über Grundkreis
    # Typisch: r_Nf ≈ r_b + 0.5·m_n (für Standardverzahnung ohne Gegenzahnrad)
    r_Nf = r_b + 0.5 * m_n

    # Sicherheit: r_Nf muss zwischen r_b und r_0 liegen
    r_Nf = max(r_Nf, r_b * 1.001)
    r_Nf = min(r_Nf, r_0)

    print(f"  Geschätzte Nutzkreise (Näherung ohne Gegenzahnrad):")
    print(f"    d_Nf ≈ {2*r_Nf:.3f} mm  (Nutzkreis Fuß,  zwischen d_b={2*r_b:.3f} und d_0={2*r_0:.3f})")
    print(f"    d_Ff ≈ {2*r_Ff:.3f} mm  (Nutzkreis Kopf, knapp unter d_a={2*r_a:.3f})")
    print(f"  Für präzisere Werte: d_Ff und d_Nf in gear_soll.py eintragen.")
    return 2 * r_Nf, 2 * r_Ff


def compute_derived_geometry(p: GearParameters) -> dict:
    """
    Berechnet alle abgeleiteten Größen aus den Eingangsparametern.
    Erkennt automatisch ob d_Ff/d_Nf Durchmesser oder Abwälzlängen sind.
    Wenn d_Ff und d_Nf nicht angegeben (0.0), werden sie geschätzt.
    """
    alpha_n = np.radians(p.alpha_n_deg)
    beta_0  = np.radians(p.beta_0_deg)

    # Stirnschnittgrößen
    alpha_t0 = np.arctan(np.tan(alpha_n) / np.cos(beta_0))
    m_t      = p.m_n / np.cos(beta_0)

    # Radien
    r_b  = p.d_b  / 2.0
    r_0  = p.d_0  / 2.0
    r_a  = p.d_a  / 2.0
    r_f  = p.d_f  / 2.0

    # d_Ff / d_Nf: wenn nicht angegeben (0), aus der theoretischen
    # Eingriffsstrecke ableiten — NICHT aus der Scan-Abdeckung (siehe
    # gear_pipeline_phase4_uv.py: die Soll-Kennwerte C_alpha/C_beta/f_H_beta
    # sind per DIN-Konvention relativ zur ECHTEN Eingriffsstrecke definiert;
    # ein anderer Bereich für Ist macht c_ist und c_soll auf verschiedenen
    # Skalen inkomparabel — "Äpfel mit Birnen").
    #
    # d_Ff (Nutzkreis Kopf): ohne bekanntes Gegenrad ist der volle Kopfkreis
    # d_a die einzig verfügbare, nicht-unterschätzende Näherung.
    #
    # d_Nf (Nutzkreis Fuß): die echte Involvente endet am WERKZEUG-Formkreis
    # (Übergang zur Trochoiden-Fußausrundung), der bei gegebenem Werkzeug-
    # Kopfradius-Koeffizient etwas OBERHALB des Grundkreises liegt — dessen
    # exakte Lage braucht Werkzeuggeometrie, die hier nicht vorliegt. Da
    # d_f > d_b bestätigt ist (kein Unterschnitt), ist d_b eine sichere,
    # konservative untere Schranke: die tatsächliche Involvente reicht nie
    # weiter nach innen. Der Unterschied zum echten Formkreis ist typisch
    # ~0.3-0.5·m_n — klein gegen die hier beobachtete Lücke zwischen Soll-
    # Bereich und tatsächlicher Scan-Abdeckung (mehrere mm).
    d_Ff = p.d_Ff if p.d_Ff > 0 else 0.0
    d_Nf = p.d_Nf if p.d_Nf > 0 else 0.0
    if d_Ff == 0 or d_Nf == 0:
        if d_Ff == 0:
            d_Ff = p.d_a
        if d_Nf == 0:
            d_Nf = p.d_b
        global _nutzkreise_printed
        if not _nutzkreise_printed:
            print(f"  Nutzkreise geschätzt (true Eingriffsstrecke, konservativ): "
                  f"d_Nf≈{d_Nf:.3f}mm (=d_b)  d_Ff≈{d_Ff:.3f}mm (=d_a)")
            print(f"  Für präzisere Werte (z.B. reale Werkzeug-Formkreis-Berechnung "
                  f"oder Zeichnungswert): d_Ff und d_Nf in gear_soll.py eintragen.")
            _nutzkreise_printed = True

    # Auto-Erkennung: Durchmesser oder Abwälzlänge?
    interpret_as_length_of_roll = (d_Ff < p.d_f * 0.9) or (d_Nf < p.d_f * 0.9)
    if interpret_as_length_of_roll:
        l_y_max = d_Ff
        l_y_min = d_Nf
        r_Ff = np.sqrt(r_b**2 + l_y_max**2)
        r_Nf = np.sqrt(r_b**2 + l_y_min**2)
        eval_mode = "length_of_roll"
    else:
        r_Ff = d_Ff / 2.0
        r_Nf = d_Nf / 2.0
        l_y_min = np.sqrt(max(r_Nf**2 - r_b**2, 1e-12))
        l_y_max = np.sqrt(max(r_Ff**2 - r_b**2, 1e-12))
        eval_mode = "diameter"

    # Grundkreis-Schrägungswinkel
    beta_b = np.arccos(np.cos(alpha_n) *
                       np.sqrt(np.tan(alpha_n)**2 + np.cos(beta_0)**2))
    c = np.tan(beta_b) / r_b

    # Zahndicke + Position
    s_t0 = r_0 * np.pi / p.n + 2.0 * p.x * p.m_n * np.tan(alpha_t0)
    theta_t = s_t0 / (2.0 * r_0)
    inv_alpha_t0 = np.tan(alpha_t0) - alpha_t0

    # Konsistenzprüfung (Lehrbuchformel)
    r_b_formula = p.n * p.m_n / (2.0 *
                                  np.sqrt(np.tan(alpha_n)**2 + np.cos(beta_0)**2))
    r_0_formula = 0.5 * p.n * p.m_n / np.cos(beta_0)

    return dict(
        alpha_n=alpha_n, beta_0=beta_0, alpha_t0=alpha_t0, beta_b=beta_b,
        r_b=r_b, r_0=r_0, r_a=r_a, r_f=r_f, r_Ff=r_Ff, r_Nf=r_Nf,
        m_n=p.m_n, m_t=m_t,
        s_t0=s_t0, theta_t=theta_t, inv_alpha_t0=inv_alpha_t0,
        c=c,
        l_y_min=l_y_min, l_y_max=l_y_max,
        pitch_angle=2.0 * np.pi / p.n,
        rel_err_rb=abs(r_b - r_b_formula) / r_b_formula,
        rel_err_r0=abs(r_0 - r_0_formula) / r_0_formula,
        eval_mode=eval_mode,
    )


def generate_soll_flank(params: GearParameters,
                        n_u: int = 50, n_v: int = 25,
                        tooth_i: int = 0, side: str = 'R') -> dict:
    """
    Generiert die Soll-Flanke für einen Zahn × Seite auf (u,v)-Gitter.

    Einheiten:
      - Koordinaten (pts_xyz, l_y_grid, z_grid, r_grid) in mm
      - f_y_soll in mm (C_alpha/C_beta/f_H_beta werden intern von µm → mm umgerechnet)
      - cheb_coeffs_soll in mm
    """
    d = compute_derived_geometry(params)

    u = np.linspace(-1.0, +1.0, n_u)
    v = np.linspace(-1.0, +1.0, n_v)
    U, V = np.meshgrid(u, v, indexing='ij')

    L_y = (U + 1.0) / 2.0 * (d['l_y_max'] - d['l_y_min']) + d['l_y_min']
    Z   = (V + 1.0) / 2.0 * params.b
    R   = np.sqrt(d['r_b']**2 + L_y**2)

    alpha_t = np.arccos(np.clip(d['r_b'] / R, -1.0, 1.0))
    inv_alpha_t = np.tan(alpha_t) - alpha_t

    # FIX: flank-Vorzeichen-Konvention (früher war dieses Block doppelt vorhanden
    # mit widersprüchlichen Vorzeichen — erste Zuweisung war toter Code).
    # R-Flanke: flank = -1 (inv_at wächst mit r → Zahn wird schmaler am Kopf ✓)
    # L-Flanke: flank = +1
    flank  = -1 if side == 'R' else +1
    hand   = params.hand
    type_g = params.type_gear
    pitch_angle  = d['pitch_angle']
    theta_center = +tooth_i * pitch_angle

    # phi_b = Basispunkt am Grundkreis für diesen Zahn × Seite
    if side == 'R':
        phi_b = theta_center + d['theta_t'] + d['inv_alpha_t0']
    else:  # 'L'
        phi_b = theta_center - d['theta_t'] - d['inv_alpha_t0']

    # 3D-Punkt (Paper I Eq 14)
    angle = phi_b + hand * d['c'] * Z + flank * inv_alpha_t
    X = R * np.cos(angle)
    Y = R * np.sin(angle)
    pts_xyz = np.stack([X, Y, Z], axis=-1)

    # Normalen (Paper I Eq 17)
    sign_tf  = type_g * flank
    angle_n  = phi_b + hand * d['c'] * Z + flank * np.tan(alpha_t)
    N_x =  sign_tf * np.cos(d['beta_b']) * np.sin(angle_n)
    N_y = -sign_tf * np.cos(d['beta_b']) * np.cos(angle_n)
    N_z = np.full_like(N_x, type_g * hand * flank * np.sin(d['beta_b']))
    normals = np.stack([N_x, N_y, N_z], axis=-1)
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)

    # FIX: Soll-Modifikationen in mm (C_alpha, C_beta, f_H_alpha, f_H_beta sind in µm → /1000)
    C_a_mm  = params.C_alpha   / 1000.0
    C_b_mm  = params.C_beta    / 1000.0
    fHa_mm  = params.f_H_alpha / 1000.0
    fHb_mm  = params.f_H_beta  / 1000.0
    f_y_soll = (-C_a_mm * U**2
                -C_b_mm * V**2
                +(fHa_mm / 2.0) * U
                +(fHb_mm / 2.0) * V)

    # Analytische Chebyshev-Koeffizienten [mm]
    cheb_coeffs_soll = np.zeros((5, 5))
    cheb_coeffs_soll[0, 0] = -(C_a_mm + C_b_mm) / 2.0
    cheb_coeffs_soll[2, 0] = -C_a_mm / 2.0
    cheb_coeffs_soll[0, 2] = -C_b_mm / 2.0
    cheb_coeffs_soll[1, 0] =  fHa_mm / 2.0
    cheb_coeffs_soll[0, 1] =  fHb_mm / 2.0

    return dict(
        u=u, v=v, U=U, V=V,
        l_y_grid=L_y, z_grid=Z, r_grid=R,
        pts_xyz=pts_xyz, normals=normals,
        f_y_soll=f_y_soll,
        cheb_coeffs_soll=cheb_coeffs_soll,
        derived=d, params=params,
        tooth_id=tooth_i, side=side,
        phi_b=phi_b, flank_sign=flank,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Zahnrad-Parameter — hier einmalig eintragen
# ─────────────────────────────────────────────────────────────────────────────

MY_GEAR = GearParameters(
    n           = 83,       # Zähnezahl
    m_n         = 2.32,      # Normalmodul [mm]
    alpha_n_deg = 19.8,     # Normaldruckwinkel [°]
    beta_0_deg  = 22,    # Schrägungswinkel [°]  ← aus Punktwolke erkannt
    b           = 45,     # Zahnbreite [mm]        ← aus Punktwolke erkannt
    x           = -0.4028,      # Profilverschiebungsfaktor

    d_a         = 212.393,      # Kopfkreisdurchmesser [mm]  ← eintragen
    d_f         = 193.5999,      # Fußkreisdurchmesser [mm]   ← eintragen
    d_0         = 207.6827,      # Teilkreisdurchmesser [mm]  ← eintragen
    d_b         = 193.5999,      # Grundkreisdurchmesser [mm] ← eintragen

    # d_Ff und d_Nf weglassen → werden automatisch geschätzt

    C_alpha     = 2,      # Profilballigkeit [µm]
    C_beta      = 6,      # Flankenballigkeit / Helix Crowning [µm]
    f_H_alpha   = 0,      # Profilwinkel-Modifikation / Profile Slope Modification [µm]
    f_H_beta    = 0,      # Flankenlinien-Winkelmodifikation [µm]

    hand        = 1,       # +1 rechtssteigend, -1 linkssteigend
    type_gear   = 1,       # +1 außenverzahnt
)


def generate_all_soll_flanks(params: GearParameters,
                              n_u: int = 50, n_v: int = 25) -> list:
    """Generiert Soll-Flanken für alle Zähne × beide Seiten."""
    return [generate_soll_flank(params, n_u, n_v, ti, side)
            for ti in range(params.n) for side in ['L', 'R']]
