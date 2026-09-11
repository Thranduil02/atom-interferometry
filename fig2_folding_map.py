"""
fig2_folding_map.py: a 3-panel figure that makes Nyquist folding itself
explicit as its own panel, rather than only showing its consequences (as
fig2_noise_levels.py's panels do).

  (a) Folding relation: the observed/folded frequency as a continuous
      function of the true source frequency f (a triangle wave, via
      fold_frequency(f, f_N)), with osc_b's harmonics (n=1..4, genuine,
      circles) and osc_d's harmonics (n=1..4, folded, squares) marked on
      top -- same colour (red), distinguished only by marker shape. osc_d's
      n=2 harmonic is the one designed to coincide with f_phi (the
      degeneracy used elsewhere in this repo); it's marked with the same
      two-tone (red ring / blue centre) diamond used for that degeneracy in
      fig2_noise_levels.py's panel (d), and annotated "degenerate". A
      horizontal dashed line marks f_phi. This panel's x-axis (true
      frequency, out to just above 4*osc_d['f']) is unrelated to panels
      (b)/(c)'s and is not shared with them.

  (b) Safe case: a real simulated periodogram of osc_b + the ULDM tone
      (compute_phases(osc_b) + uldm_phase_vec(...)), at the 1e-5 rad/rtHz
      "stretch" noise level -- osc_b's harmonics all sit below f_N, so
      nothing folds onto f_phi, and both the comb and the ULDM peak are
      cleanly resolved.

  (c) Degenerate case: the same real simulated periodogram already used in
      fig2_noise_levels.py's panel (d) (compute_phases(osc_d,
      circular=True) + uldm_phase_vec(...), same noise level) -- osc_d's
      n=2 harmonic folds exactly onto f_phi, so the ULDM peak is masked by
      (indistinguishable from) the folded harmonic.

All physics (constants, atom_path/arm_positions/V, compute_phases,
uldm_phase_vec, add_noise, periodogram, fold_frequency, osc_b/osc_d) is
copied verbatim from fig2_noise_levels.py -- fig2_noise_levels.py itself is
not imported or modified, since its lack of a .py extension makes import
awkward and every sibling fig2_* script in this repo is a standalone copy
rather than a cross-import, so this follows that established convention.
"""

import numpy as np
import matplotlib.pyplot as plt

# ── constants (verbatim from fig2_noise_levels.py) ──────────────────────────────
m_atom  = 1.46e-25;  hbar = 1.055e-34
G       = 6.67e-11;  g    = 9.81
n       = 1000
v_kick  = n * hbar * (2 * np.pi / 698e-9) / m_atom
u0      = 24.5        # launch speed [m/s]
T       = 2.5         # half-interferometry time [s]
T_fly   = 2 * T
T_cyc   = 5.0          # cycle time [s]
dt      = 0.01

N_fly   = int(round(T_fly / dt))
t_local = np.arange(N_fly) * dt

T_total  = 3e7   # total observation time [s] (~1 year)
N_cycles = int(T_total / T_cyc)
f_N      = 1.0 / (2 * T_cyc)   # Nyquist frequency = 0.1 Hz

delta_r = 70.0   # gradiometer baseline (matches 100m_oscillators)

SHOT_NOISE_ASD = 1e-4   # rad/sqrt(Hz) -- "stretch" level, the only one used here

# ── colour-by-physical-origin palette (verbatim from fig2_noise_levels.py) ──────
COLORS = dict(
    uldm        = '#1f4e9c',   # blue  -- f_phi (ULDM signal), always
    fundamental = '#c1272d',   # red   -- f0 and its harmonics, always
    harmonic    = '#c1272d',
    folded      = '#c1272d',
)

# ── ULDM parameters (verbatim) ───────────────────────────────────────────────────
f_phi       = 0.085   # moved off f_N = 0.1 Hz
hbar_eV     = 6.582119569e-16                 # hbar in eV*s
m_phi       = 2 * np.pi * f_phi * hbar_eV     # Compton relation
G_nat       = 6.708e-39
d_e         = 1e-3;  d_m_e      = 1e-3
rho_phi     = 0.3
theta       = np.pi / 2;  xi  = 0.06
omega_A     = 2.67e15
L_base      = 140.0
n_lmt       = n;  conv_factor = 2.77e-12
c_light     = 3e8

def uldm_phase_vec(t, f_uldm, m_phi_val):
    delta_omega_A = (omega_A * np.sqrt(G_nat * 4 * np.pi)
                     * (d_m_e + (2 + xi) * d_e)
                     * (np.sqrt(2 * rho_phi) / m_phi_val)
                     * conv_factor)
    omega_phi = 2 * np.pi * f_uldm
    return (8 * (delta_omega_A / omega_phi) * (delta_r / L_base)
            * np.sin(omega_phi * n_lmt * L_base / (2 * c_light))
            * np.sin(omega_phi * (T - (n_lmt - 1) * L_base / c_light) / 2)
            * np.cos(omega_phi * ((T_fly + L_base / c_light) / 2 + t) + theta))

# ── Nyquist folding helper (verbatim) ────────────────────────────────────────────
def fold_frequency(f, fN):
    fs = 2 * fN
    r = f % fs
    return fs - r if r > fN else r

# ── trajectory helpers (verbatim from fig2_noise_levels.py) ─────────────────────
def atom_path(z0, u, t0, t1, floor):
    z, v, zs = z0, u, []
    for _ in np.arange(t0, t1, dt):
        z += v*dt - 0.5*g*dt**2;  v -= g*dt
        if z <= floor and v < 0:  z = floor;  v = 0;  break
        zs.append(z)
    return np.array(zs), v

def arm_positions(z0, floor, upper=False):
    k1 = v_kick if upper else 0.0
    p1, v1 = atom_path(z0, u0 + k1, 0, T, floor)
    k2 = -v_kick if upper else v_kick
    p2, _  = atom_path(p1[-1] if len(p1) else float(floor),
                        v1 + k2, T, T_fly, floor)
    path = np.concatenate([p1, p2])
    if len(path) < N_fly:
        path = np.concatenate([path, np.full(N_fly - len(path), float(floor))])
    return path[:N_fly]

def V(z, h, M, D):
    return -G * M / np.sqrt(D**2 + (z - h)**2)

lower1 = arm_positions(0, 0, upper=False)
upper1 = arm_positions(0, 0, upper=True)
lower2 = arm_positions(delta_r, delta_r, upper=False)
upper2 = arm_positions(delta_r, delta_r, upper=True)

t_starts = np.arange(N_cycles) * T_cyc
t_meas   = t_starts + T_fly

def compute_phases(osc, circular=False, chunk=50_000):
    """Vectorized, chunked over cycles (verbatim from fig2_noise_levels.py)."""
    phases = np.empty(N_cycles)
    for start in range(0, N_cycles, chunk):
        end   = min(start + chunk, N_cycles)
        t_abs = t_starts[start:end][:, None] + t_local[None, :]
        osc_phase = 2 * np.pi * osc['f'] * t_abs
        if circular:
            h_osc = osc['h0'] + osc['amp'] * np.sin(osc_phase)
            D_osc = osc['D'] + osc['amp'] * np.cos(osc_phase)
        else:
            h_osc = osc['h0'] + osc['amp'] * np.cos(osc_phase)
            D_osc = osc['D']
        dphi  = ((V(upper1, h_osc, osc['M'], D_osc) - V(lower1, h_osc, osc['M'], D_osc))
               - (V(upper2, h_osc, osc['M'], D_osc) - V(lower2, h_osc, osc['M'], D_osc)))
        phases[start:end] = (m_atom / hbar) * np.trapezoid(dphi, t_local, axis=1)
        print(f"    cycles {end:,}/{N_cycles:,} done", flush=True)
    return phases

def add_noise(phases, shot_noise_asd):
    return phases + np.random.normal(0.0, shot_noise_asd / np.sqrt(T_cyc), size=N_cycles)

def periodogram(ph):
    f_k  = np.fft.rfftfreq(N_cycles, d=T_cyc)
    S_k  = (T_cyc / N_cycles) * np.abs(np.fft.rfft(ph - ph.mean()))**2
    mask = f_k > 0
    return f_k[mask], S_k[mask]

def peak_at(f_k, S_k_urad, f_target):
    idx = np.argmin(np.abs(f_k - f_target))
    return f_k[idx], S_k_urad[idx]

# ── oscillator definitions (verbatim; only osc_b/osc_d are used here) ───────────
osc_common = dict(amp=0.5, M=1.0, D=3.0, h0=1.0)
osc_b = dict(name='b', f=0.02,   **osc_common)   # sub-Nyquist comb (safe case)
osc_d = dict(name='d', f=0.1425, **osc_common)   # super-Nyquist, n=2 -> f_phi
                                                  # (2D circular oscillator, see below)

def harmonic_table(label, osc, f_k, S_urad, n_max=5):
    """Verbatim format from fig2_noise_levels.py's harmonic_table()."""
    print(f"\nOscillator ({label}): f0 = {osc['f']} Hz, M={osc['M']}, "
          f"amp={osc['amp']}, D={osc['D']}, h0={osc['h0']}")
    print(f"  {'n':>2}  {'n*f0 [Hz]':>10}  {'folded [Hz]':>12}  {'peak [urad/rtHz]':>18}  status")
    rows = []
    for k in range(1, n_max + 1):
        f_true = k * osc['f']
        f_fold = fold_frequency(f_true, f_N)
        _, peak = peak_at(f_k, S_urad, f_fold)
        status = 'genuine (< f_N)' if f_true <= f_N else 'folded/aliased'
        print(f"  {k:>2}  {f_true:>10.4f}  {f_fold:>12.4f}  {peak:>18.4g}  {status}")
        rows.append((k, f_true, f_fold, peak))
    return rows

def multicolor_annotate(ax, xy, parts, xytext, fontsize=5, va='bottom'):
    """Verbatim from fig2_noise_levels.py."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    ax_bbox = ax.get_window_extent(renderer=renderer)
    x, y = ax.transData.transform(xy)
    x += xytext[0]
    y += xytext[1]
    xf = (x - ax_bbox.x0) / ax_bbox.width
    yf = (y - ax_bbox.y0) / ax_bbox.height
    for text, color in parts:
        t = ax.annotate(text, xy=xy, xycoords='data',
                         xytext=(xf, yf), textcoords='axes fraction',
                         fontsize=fontsize, color=color, va=va, ha='left')
        fig.canvas.draw()
        xf = (t.get_window_extent(renderer=renderer).x1 - ax_bbox.x0) / ax_bbox.width

if __name__ == "__main__":
    # ── panel (b): safe case -- osc_b + ULDM, real simulated periodogram ────────
    print(f"Panel (b): oscillator f0={osc_b['f']} Hz + ULDM ({N_cycles:,} cycles)...")
    raw_b = compute_phases(osc_b) + uldm_phase_vec(t_meas, f_phi, m_phi)
    f_k, S_b = periodogram(add_noise(raw_b, SHOT_NOISE_ASD))
    urad_b = np.sqrt(S_b) * 1e6

    # ── panel (c): degenerate case -- osc_d (circular) + ULDM, reused verbatim ──
    print(f"Panel (c): circular oscillator f0={osc_d['f']} Hz + ULDM ({N_cycles:,} cycles)...")
    raw_d = compute_phases(osc_d, circular=True) + uldm_phase_vec(t_meas, f_phi, m_phi)
    _, S_d = periodogram(add_noise(raw_d, SHOT_NOISE_ASD))
    urad_d = np.sqrt(S_d) * 1e6

    rows_b = harmonic_table('b', osc_b, f_k, urad_b, n_max=5)
    rows_d = harmonic_table('d', osc_d, f_k, urad_d, n_max=5)

    f_phi_bin_b, peak_uldm_b = peak_at(f_k, urad_b, f_phi)
    f_phi_bin_d, peak_uldm_d = peak_at(f_k, urad_d, f_phi)
    print(f"\nPanel (b): ULDM peak at f_phi={f_phi} Hz -> {peak_uldm_b:.4g} urad/rtHz")
    print(f"Panel (c): ULDM/2f0-folded peak at f_phi={f_phi} Hz -> {peak_uldm_d:.4g} urad/rtHz")

    # ── figure layout: panel (a) 1.3x the height of (b)/(c); (a) has its own,
    #    unrelated x-axis (true frequency) and is not shared with (b)/(c) ───────
    fig = plt.figure(figsize=(4.5, 12), dpi=200)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 1], hspace=0.35)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2], sharex=ax_b)

    # ── panel (a): folding relation ──────────────────────────────────────────
    f_max_a = 4.15 * osc_d['f']
    f_grid  = np.linspace(0, f_max_a, 2000)
    folded_curve = np.array([fold_frequency(fv, f_N) for fv in f_grid])

    ax_a.plot(f_grid, folded_curve, color='k', lw=1.0, zorder=2)
    ax_a.axhline(f_phi, color=COLORS['uldm'], ls='--', lw=0.8, zorder=1)
    ax_a.annotate(r'$f_\varphi$', (f_max_a, f_phi), textcoords='offset points',
                  xytext=(-4, 4), fontsize=6, color=COLORS['uldm'], ha='right')

    b_harmonics = [(k, k * osc_b['f'], fold_frequency(k * osc_b['f'], f_N)) for k in range(1, 5)]
    d_harmonics = [(k, k * osc_d['f'], fold_frequency(k * osc_d['f'], f_N)) for k in range(1, 5)]

    bx = [h[1] for h in b_harmonics]; by = [h[2] for h in b_harmonics]
    ax_a.scatter(bx, by, color=COLORS['harmonic'], marker='o', s=18, zorder=4)

    dx = [h[1] for h in d_harmonics if h[0] != 2]
    dy = [h[2] for h in d_harmonics if h[0] != 2]
    ax_a.scatter(dx, dy, color=COLORS['harmonic'], marker='s', s=18, zorder=4)

    k2, f_true2, f_fold2 = d_harmonics[1]   # n=2, coincides with f_phi
    ax_a.scatter([f_true2], [f_fold2], color=COLORS['fundamental'], s=45, marker='D', zorder=5)
    ax_a.scatter([f_true2], [f_fold2], color=COLORS['uldm'], s=16, marker='D', zorder=6)
    ax_a.annotate('degenerate', (f_true2, f_fold2), textcoords='offset points',
                  xytext=(6, -10), fontsize=6, color=COLORS['fundamental'])

    ax_a.set_xlim(0, f_max_a)
    ax_a.set_ylim(0, f_N * 1.05)
    ax_a.set_xlabel('true source frequency $f$ [Hz]', fontsize=8)
    ax_a.set_ylabel('folded frequency [Hz]', fontsize=8)
    ax_a.tick_params(labelsize=7)
    ax_a.text(0.99, 0.94, '(a)', transform=ax_a.transAxes, ha='right', va='top',
               fontsize=7, fontweight='bold')

    legend_handles = [
        plt.Line2D([0], [0], marker='o', color='none', markerfacecolor=COLORS['harmonic'],
                   markersize=5, label=r'osc$_b$ harmonics ($n$=1..4)'),
        plt.Line2D([0], [0], marker='s', color='none', markerfacecolor=COLORS['harmonic'],
                   markersize=5, label=r'osc$_d$ harmonics ($n$=1..4)'),
    ]
    ax_a.legend(handles=legend_handles, loc='upper left', fontsize=5, frameon=False,
                handlelength=1.0)

    # ── panels (b)/(c): shared log y-axis, shared x-axis (0, f_N) ──────────────
    y_max = 3 * max(peak_uldm_b, peak_uldm_d,
                     max(r[3] for r in rows_b[:4]), max(r[3] for r in rows_d[:4]))
    y_min = 0.1 * SHOT_NOISE_ASD * 1e6

    for ax in (ax_b, ax_c):
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, f_N)
        ax.tick_params(labelsize=7)
        ax.axvline(f_N, color='k', ls='--', lw=0.5, alpha=0.6)

    for ax in (ax_b, ax_c):
        ax.set_ylabel(r'$\sqrt{S_k}$ [$\mu$rad/$\sqrt{\mathrm{Hz}}$]', fontsize=7)

    # ── panel (b): safe case ─────────────────────────────────────────────────
    ax_b.plot(f_k, urad_b, color='#9c8358', lw=0.3, zorder=2)
    b_offsets = {1: (4, 6), 2: (4, 6), 3: (4, 6), 4: (5, -8)}
    for k, f_true, f_fold, peak in rows_b[:4]:
        col = COLORS['fundamental'] if k == 1 else COLORS['harmonic']
        ax_b.vlines(f_fold, y_min, peak, color=col, lw=0.6, zorder=4)
        ax_b.scatter([f_fold], [peak], color=col, zorder=5, s=7)
        label = r'$f_0$' if k == 1 else rf'${k}f_0$'
        ax_b.annotate(label, (f_fold, peak), textcoords='offset points',
                      xytext=b_offsets[k], fontsize=5, color=col)
    ax_b.vlines(f_phi_bin_b, y_min, peak_uldm_b, color=COLORS['uldm'], lw=0.6, zorder=4)
    ax_b.scatter([f_phi_bin_b], [peak_uldm_b], color=COLORS['uldm'], zorder=5, s=8, marker='D')
    ax_b.annotate(r'$f_\varphi$', (f_phi_bin_b, peak_uldm_b), textcoords='offset points',
                  xytext=(6, -6), fontsize=5, color=COLORS['uldm'])
    ax_b.text(0.99, 0.93, '(b)', transform=ax_b.transAxes, ha='right', va='top',
               fontsize=7, fontweight='bold')

    # ── panel (c): degenerate case ───────────────────────────────────────────
    ax_c.plot(f_k, urad_d, color='#9c8358', lw=0.3, zorder=2)
    d_offsets = {1: (4, -2), 3: (0, 6), 4: (4, -2)}
    for k, f_true, f_fold, peak in rows_d[:4]:
        if abs(f_fold - f_phi) < 1e-6:
            continue   # n=2 folds exactly onto f_phi -- labelled once, below
        col = COLORS['fundamental'] if k == 1 else COLORS['folded']
        ax_c.vlines(f_fold, y_min, peak, color=col, lw=0.6, zorder=4)
        ax_c.scatter([f_fold], [peak], color=col, zorder=5, s=7)
        label = r'$f_0$ (folded)' if k == 1 else rf'${k}f_0$ (folded)'
        ax_c.annotate(label, (f_fold, peak), textcoords='offset points',
                      xytext=d_offsets[k], fontsize=5, color=col)
    ax_c.vlines(f_phi_bin_d, y_min, peak_uldm_d, color=COLORS['fundamental'], lw=0.6, zorder=4)
    ax_c.scatter([f_phi_bin_d], [peak_uldm_d], color=COLORS['fundamental'], zorder=5, s=18, marker='D')
    ax_c.scatter([f_phi_bin_d], [peak_uldm_d], color=COLORS['uldm'], zorder=6, s=7, marker='D')
    multicolor_annotate(ax_c, (f_phi_bin_d, peak_uldm_d),
                         [(r'$f_\varphi$', COLORS['uldm']),
                          (' & ', 'black'),
                          (r'$2f_0$ (folded)', COLORS['fundamental'])],
                         xytext=(-58, 16), fontsize=5)
    ax_c.text(0.99, 0.93, '(c)', transform=ax_c.transAxes, ha='right', va='top',
               fontsize=7, fontweight='bold')
    ax_c.set_xlabel('$f$ [Hz]', fontsize=8)

    fig.savefig(r'C:\Users\Georg\Desktop\fig2_folding_map.jpg', dpi=200, bbox_inches='tight')

    print("\n" + "=" * 78)
    print("Suggested caption:")
    print(f"  dt = T_cyc = {T_cyc:g} s, N_cycles = {N_cycles:,}, T_obs = {T_total:.3g} s, "
          f"f_N = {f_N:g} Hz, shot noise = {SHOT_NOISE_ASD:.0e} rad/sqrt(Hz) (stretch).")
    print(f"  ULDM: f_phi = {f_phi} Hz, m_phi = {m_phi:.4g} eV.")
    print(f"  Oscillator (b, safe): f0 = {osc_b['f']} Hz, M = {osc_b['M']}, "
          f"amp = {osc_b['amp']} m, D = {osc_b['D']} m, h0 = {osc_b['h0']} m.")
    print(f"  Oscillator (d, degenerate): f0 = {osc_d['f']} Hz, M = {osc_d['M']}, "
          f"amp = {osc_d['amp']} m, D = {osc_d['D']} m, h0 = {osc_d['h0']} m.")
    print("=" * 78)

    plt.show()
