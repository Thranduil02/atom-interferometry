"""
fig2_circle_summary.py: a compact 4-panel version of fig2_multi_directional.py
for the 100 m instrument.

  (a) ULDM + noise only -- the reference tone at f_phi over both shot-noise
      floors (baseline 1e-4 and stretch 1e-5 rad/sqrt(Hz), house style from
      fig2_noise_levels.py), nothing else.

  (b) 1D vertical oscillator + ULDM, at a sub-Nyquist f0 = F_SLOW so the
      whole comb sits below f_N and nothing folds. Shows the nonlinearity
      comb (f0, 2f0, 3f0, ...) from the 1/sqrt(D^2+(z-h)^2) potential.

  (c) 2D horizontal circle + ULDM, f0 = F_CIRC (super-Nyquist, so its comb
      FOLDS back below f_N). Same geometry as fig2_multi_directional.py
      panel (c).

  (d) Bar chart: the ratio of the 2f0 peak to the f0 peak (both folded),
      for a circle inclined at THETA_DEG = [0 (horizontal), 30, 45, 60, 90
      (vertical), 120, 135, 150, 180 (horizontal again, mirrored)]. A
      horizontal circle's height never changes, so its motion is a pure
      cos(phi) in D -- its comb is symmetric and 2f0/f0 is small. A
      vertical circle modulates height and horizontal distance out of
      phase, which mixes more power into 2f0. Tilting interpolates between
      the two, so this ratio is a direct, single-number readout of the
      source geometry. Going past 90 deg back to 180 deg retraces the
      horizontal-circle case (mirrored in phase), which is a useful
      internal sanity check: the 180 deg bar should reproduce the 0 deg
      bar.

This script does NOT save any file -- it just calls plt.show() so the
figure appears in whatever interactive backend is active.

Provenance: constants, the ULDM tone, fold_frequency, the trajectory
helpers (atom_path/arm_positions), the potential V, the phase integral
inside compute_phases, add_noise, periodogram, harmonic_table and
multicolor_annotate are all copied verbatim from fig2_multi_directional.py
/ fig2_noise_levels.py. traj_inclined_circle at incline_deg=0 reproduces
traj_horizontal_circle exactly, and at incline_deg=90 reproduces
traj_vertical_circle exactly, so panel (d) reuses ONE trajectory function
for the whole theta sweep (including panel (c)'s theta=0 case, whose
periodogram is reused rather than recomputed).
"""

import numpy as np
import matplotlib.pyplot as plt

# ── constants (verbatim from fig2_noise_levels.py / fig2_multi_directional.py) ──
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

T_total  = 3e7   # total observation time [s]
N_cycles = int(T_total / T_cyc)
f_N      = 1.0 / (2 * T_cyc)   # Nyquist frequency = 0.1 Hz

delta_r = 70.0   # gradiometer baseline (matches 100m_oscillators)

SHOT_NOISE_ASD = 1e-4   # rad/sqrt(Hz) -- reference level for the theta sweep (d)

# both noise floors, house style from fig2_noise_levels.py: baseline (lighter
# sand) plotted behind, stretch (darker tan) plotted on top, in panels a/b/c
NOISE_LEVELS = [
    dict(key='baseline', asd=1e-4, label='baseline (1e-4)', color='#d9c9a8', lw=0.3, zorder=2),
    dict(key='stretch',  asd=1e-5, label='stretch (1e-5)',  color='#9c8358', lw=0.35, zorder=3),
]

# ── geometry parameters specific to this figure ─────────────────────────────────
F_SLOW    = 0.02    # panel (b): sub-Nyquist, shows the nonlinearity comb
F_CIRC    = 0.131   # panels (c)/(d): shared super-Nyquist fundamental
THETA_DEG = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0]   # panel (d) sweep

# ── colour-by-physical-origin palette (verbatim) ────────────────────────────────
COLORS = dict(
    uldm        = '#1f4e9c',   # blue  -- f_phi (ULDM signal), always
    fundamental = '#c1272d',   # red   -- f0 and its harmonics, always
    harmonic    = '#c1272d',
    folded      = '#c1272d',
    noise       = '#9c8358',   # tan   -- periodogram / noise floor
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

# ── trajectory helpers (verbatim) ────────────────────────────────────────────────
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

# ── source trajectories (verbatim) ──────────────────────────────────────────────
def traj_linear(t_abs, p):
    ph = 2 * np.pi * p['f'] * t_abs
    return p['h0'] + p['amp'] * np.cos(ph), p['D']

def traj_inclined_circle(t_abs, p):
    """Circle tilted by alpha out of the horizontal. alpha = 0 -> pure
    horizontal circle (height fixed); alpha = 90 -> pure vertical circle."""
    ph    = 2 * np.pi * p['f'] * t_abs
    alpha = np.deg2rad(p['incline_deg'])
    x = p['D'] + p['amp'] * np.cos(alpha) * np.cos(ph)
    y = p['amp'] * np.sin(ph)
    h = p['h0'] + p['amp'] * np.sin(alpha) * np.cos(ph)
    return h, np.sqrt(x*x + y*y)

def compute_phases(src, chunk=25_000):
    phases = np.empty(N_cycles)
    for start in range(0, N_cycles, chunk):
        end   = min(start + chunk, N_cycles)
        t_abs = t_starts[start:end][:, None] + t_local[None, :]
        h_osc, D_osc = src['traj'](t_abs, src)
        dphi  = ((V(upper1, h_osc, src['M'], D_osc) - V(lower1, h_osc, src['M'], D_osc))
               - (V(upper2, h_osc, src['M'], D_osc) - V(lower2, h_osc, src['M'], D_osc)))
        phases[start:end] = (m_atom / hbar) * np.trapezoid(dphi, t_local, axis=1)
        if end == N_cycles or (end // chunk) % 20 == 0:
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

# ── source definitions ──────────────────────────────────────────────────────────
src_common = dict(amp=0.5, M=1.0, D=3.0, h0=1.0)

src_b = dict(name='b', f=F_SLOW, traj=traj_linear, **src_common)
src_c = dict(name='c', f=F_CIRC, traj=traj_inclined_circle, incline_deg=0.0,
             **src_common)

def harmonic_table(label, src, f_k, S_urad, n_max=4):
    """Same format as fig2_multi_directional.py's harmonic_table() (no
    sideband splitting needed here -- no pendulum in this figure)."""
    print(f"\nPanel ({label}): source f0 = {src['f']:.4f} Hz, M={src['M']}, "
          f"amp={src['amp']}, D={src['D']}, h0={src['h0']} "
          f"[{src['traj'].__name__}, incline={src.get('incline_deg', 'n/a')}]")
    print(f"  {'n':>2}  {'n*f0 [Hz]':>10}  {'folded [Hz]':>12}  {'peak [urad/rtHz]':>18}  status")
    rows = []
    for k in range(1, n_max + 1):
        f_true = k * src['f']
        f_fold = fold_frequency(f_true, f_N)
        f_bin, peak = peak_at(f_k, S_urad, f_fold)
        status = 'genuine (< f_N)' if f_true <= f_N else 'folded/aliased'
        print(f"  {k:>2}  {f_true:>10.4f}  {f_fold:>12.4f}  {peak:>18.4g}  {status}")
        rows.append((k, f_true, f_fold, peak, f_bin))
    return rows

if __name__ == "__main__":
    uldm = uldm_phase_vec(t_meas, f_phi, m_phi)

    print(f"Panel (a): ULDM only ({N_cycles:,} cycles, analytic)...")
    raw_a = uldm

    print(f"Panel (b): traj_linear f0={F_SLOW:.4f} Hz + ULDM ({N_cycles:,} cycles)...")
    raw_b = compute_phases(src_b) + uldm

    print(f"Panel (c): horizontal circle f0={F_CIRC:.4f} Hz + ULDM "
          f"({N_cycles:,} cycles)...")
    raw_c = compute_phases(src_c) + uldm

    # both noise floors, for each of panels a/b/c -- 'urad' (baseline) is used
    # for harmonic tables and marker placement, 'urad_by_level' holds both
    # traces for plotting
    raws = {'a': raw_a, 'b': raw_b, 'c': raw_c}
    urad_by_level = {letter: {} for letter in 'abc'}
    f_k = None
    for letter, raw in raws.items():
        for lvl in NOISE_LEVELS:
            f_k, S = periodogram(add_noise(raw, lvl['asd']))
            urad_by_level[letter][lvl['key']] = np.sqrt(S) * 1e6
    urad = {letter: urad_by_level[letter]['baseline'] for letter in 'abc'}

    rows = {}
    rows['b'] = harmonic_table('b', src_b, f_k, urad['b'])
    rows['c'] = harmonic_table('c', src_c, f_k, urad['c'])

    f_phi_bins = {k: peak_at(f_k, u, f_phi) for k, u in urad.items()}
    print(f"\nULDM peak at f_phi = {f_phi} Hz, per panel:")
    for k in 'abc':
        print(f"  ({k})  {f_phi_bins[k][1]:>12.4g} urad/rtHz")
    print(f"\nShot-noise floor = {SHOT_NOISE_ASD*1e6:.4g} urad/rtHz")

    # ── theta sweep for panel (d): 2f0/f0 ratio vs. circle inclination ──────────
    # theta = 0 is exactly src_c (already computed above) -- reuse it rather
    # than resimulating.
    ratio_2f0_f0 = {}
    peak_f0_theta = {}
    peak_2f0_theta = {}
    for th in THETA_DEG:
        if th == 0.0:
            f0_peak, f2_peak = rows['c'][0][3], rows['c'][1][3]
        else:
            src_th = dict(name=f'theta{th:g}', f=F_CIRC, traj=traj_inclined_circle,
                          incline_deg=th, **src_common)
            print(f"Panel (d) sweep: incline={th:g} deg "
                  f"({N_cycles:,} cycles)...")
            raw_th = compute_phases(src_th) + uldm
            _, S_th = periodogram(add_noise(raw_th, SHOT_NOISE_ASD))
            urad_th = np.sqrt(S_th) * 1e6
            rows_th = harmonic_table(f'd, {th:g} deg', src_th, f_k, urad_th, n_max=2)
            f0_peak, f2_peak = rows_th[0][3], rows_th[1][3]
        peak_f0_theta[th]  = f0_peak
        peak_2f0_theta[th] = f2_peak
        ratio_2f0_f0[th]   = f2_peak / f0_peak

    print(f"\n2f0/f0 folded-peak ratio vs. circle inclination "
          f"(0 deg = horizontal, 90 deg = vertical):")
    for th in THETA_DEG:
        print(f"  theta={th:>5.1f} deg   f0={peak_f0_theta[th]:>12.4g}   "
              f"2f0={peak_2f0_theta[th]:>12.4g}   ratio={ratio_2f0_f0[th]:.4f}")

    # ── plot: 3 stacked periodogram panels (a,b,c) + 1 bar chart (d) ────────────
    fig = plt.figure(figsize=(4.5, 15), dpi=200)
    gs  = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.1], hspace=0.4)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)
    ax_d = fig.add_subplot(gs[3])

    y_max = 3 * max(max(r[3] for r in rows[k][:4]) for k in ('b', 'c'))
    y_max = max(y_max, 3 * max(p for _, p in f_phi_bins.values()))
    y_min = 0.1 * min(lvl['asd'] for lvl in NOISE_LEVELS) * 1e6

    for ax in (ax_a, ax_b, ax_c):
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, f_N)
        ax.tick_params(labelsize=7)
        ax.axvline(f_N, color='k', ls='--', lw=0.5, alpha=0.6)
    ax_b.set_ylabel(r'$\sqrt{S_k}$ [$\mu$rad/$\sqrt{\mathrm{Hz}}$]', fontsize=7)
    for letter, ax in zip('abc', (ax_a, ax_b, ax_c)):
        ax.text(0.99, 0.93, f'({letter})', transform=ax.transAxes, ha='right',
                va='top', fontsize=7, fontweight='bold')
    ax_c.set_xlabel('$f$ [Hz]', fontsize=8)
    ax_a.text(f_N, y_max*0.5, r'  $f_N$', fontsize=6, ha='left', va='center')

    def mark_uldm(ax, letter, xytext=(6, 6)):
        f_bin, peak = f_phi_bins[letter]
        ax.vlines(f_bin, y_min, peak, color=COLORS['uldm'], lw=0.6, zorder=4)
        ax.scatter([f_bin], [peak], color=COLORS['uldm'], zorder=5, s=8, marker='D')
        ax.annotate(r'$f_\varphi$', (f_bin, peak), textcoords='offset points',
                    xytext=xytext, fontsize=5, color=COLORS['uldm'])

    def mark_harmonics(ax, letter, folded, offsets, n_show=4):
        f_bin_phi, _ = f_phi_bins[letter]
        df = f_k[1] - f_k[0]
        for k, f_true, f_fold, peak, f_bin in rows[letter][:n_show]:
            col = COLORS['fundamental'] if k == 1 else (
                  COLORS['folded'] if folded else COLORS['harmonic'])
            if abs(f_fold - f_bin_phi) < 2 * df:
                continue
            ax.vlines(f_bin, y_min, peak, color=col, lw=0.6, zorder=4)
            ax.scatter([f_bin], [peak], color=col, zorder=5, s=7)
            base = r'$f_0$' if k == 1 else rf'${k}f_0$'
            label = base + ' (folded)' if folded and f_true > f_N else base
            ax.annotate(label, (f_bin, peak), textcoords='offset points',
                        xytext=offsets.get(k, (4, 6)), fontsize=5, color=col)

    def plot_noise_stack(ax, letter):
        for lvl in NOISE_LEVELS:
            ax.plot(f_k, urad_by_level[letter][lvl['key']], color=lvl['color'],
                    lw=lvl['lw'], zorder=lvl['zorder'])

    plot_noise_stack(ax_a, 'a')
    mark_uldm(ax_a, 'a')
    ax_a.legend(handles=[plt.Line2D([], [], color=lvl['color'], lw=1.5,
                                     label=lvl['label']) for lvl in NOISE_LEVELS],
                fontsize=5, loc='upper left', frameon=False)

    plot_noise_stack(ax_b, 'b')
    mark_harmonics(ax_b, 'b', folded=False,
                   offsets={1: (4, 6), 2: (4, 6), 3: (4, 6), 4: (5, -8)})
    mark_uldm(ax_b, 'b', xytext=(6, -6))

    plot_noise_stack(ax_c, 'c')
    mark_harmonics(ax_c, 'c', folded=True,
                   offsets={1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)})
    mark_uldm(ax_c, 'c', xytext=(6, -6))

    # panel (d): bar chart of the 2f0/f0 ratio vs. inclination
    x = np.arange(len(THETA_DEG))
    bar_vals = [ratio_2f0_f0[th] for th in THETA_DEG]
    bars = ax_d.bar(x, bar_vals, color=COLORS['fundamental'], width=0.6)
    labels_d = []
    for th in THETA_DEG:
        if th == 0.0:
            labels_d.append('horizontal\n(0°)')
        elif th == 90.0:
            labels_d.append('vertical\n(90°)')
        elif th == 180.0:
            labels_d.append('horizontal\n(180°)')
        else:
            labels_d.append(f'{th:g}°')
    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels_d, fontsize=6)
    ax_d.set_ylabel(r'$\sqrt{S_{2f_0}}\,/\,\sqrt{S_{f_0}}$', fontsize=7)
    ax_d.set_ylim(0, 1.3 * max(bar_vals))
    ax_d.tick_params(labelsize=7)
    ax_d.text(0.99, 0.93, '(d)', transform=ax_d.transAxes, ha='right',
              va='top', fontsize=7, fontweight='bold')

    print("\n" + "=" * 78)
    print("Suggested caption:")
    print(f"  dt = T_cyc = {T_cyc:g} s, N_cycles = {N_cycles:,}, T_obs = {T_total:.3g} s, "
          f"f_N = {f_N:g} Hz, shot noise = {SHOT_NOISE_ASD:.0e} rad/sqrt(Hz) in all panels.")
    print(f"  ULDM: f_phi = {f_phi} Hz, m_phi = {m_phi:.4g} eV (present in every panel).")
    print(f"  Source: M = {src_common['M']}, amp = {src_common['amp']} m, "
          f"D = {src_common['D']} m, h0 = {src_common['h0']} m.")
    print(f"  (b) 1D vertical oscillator, f0 = {F_SLOW} Hz (sub-Nyquist: nonlinearity comb, no folding).")
    print(f"  (c) horizontal circle, f0 = {F_CIRC} Hz (super-Nyquist: comb folds).")
    print(f"  (d) 2f0/f0 folded-peak ratio for a circle inclined at "
          f"{', '.join(f'{th:g}' for th in THETA_DEG)} deg out of the horizontal.")
    print("=" * 78)

    plt.show()
