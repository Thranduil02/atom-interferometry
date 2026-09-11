"""
fig2_directions_1d.py: the fig2_circle_summary.py figure, but for a 1D
LINEAR oscillator instead of a circle, for the 100 m instrument.

  (a) ULDM + noise only -- the reference tone at f_phi over both shot-noise
      floors (baseline 1e-4 and stretch 1e-5 rad/sqrt(Hz), house style from
      fig2_noise_levels.py), nothing else.

  (b) 1D oscillator + ULDM, direction = vertical (beta = 0 deg), at a
      sub-Nyquist f0 = F_SLOW so the whole comb sits below f_N and nothing
      folds. Shows the nonlinearity comb (f0, 2f0, 3f0, ...) from the
      1/sqrt(D^2+(z-h)^2) potential -- here purely from height modulation.

  (c) 1D oscillator + ULDM, direction = horizontal (beta = 90 deg), f0 =
      F_CIRC (super-Nyquist, so its comb FOLDS back below f_N). Same
      frequency as (d)'s sweep, purely from horizontal-distance modulation
      instead of height.

  (d) Bar chart: the ratio of the 2f0 peak to the f0 peak (both folded),
      for a 1D oscillator whose axis of motion is rotated through
      BETA_DEG = [0 (vertical), 30, 45, 60, 90 (horizontal), 120, 135, 150,
      180 (vertical again, mirrored)] measured from vertical. At beta = 0
      the source moves straight up and down (modulating h only); at
      beta = 90 it moves straight in and out along the gradiometer axis
      (modulating D only); in between it does both at once, along a
      straight line (NOT a circle -- there is no second, phase-quadrature
      component the way there is for fig2_circle_summary.py's inclined
      circle). Going past 90 deg back to 180 deg retraces the vertical
      case (mirrored in phase, disp -> -disp), which is a useful internal
      sanity check: the 180 deg bar should reproduce the 0 deg bar.

This script does NOT save any file -- it just calls plt.show() so the
figure appears in whatever interactive backend is active.

Provenance: constants, the ULDM tone, fold_frequency, the trajectory
helpers (atom_path/arm_positions), the potential V, the phase integral
inside compute_phases, add_noise, periodogram and harmonic_table are all
copied verbatim from fig2_circle_summary.py. The ONE difference is the
trajectory: traj_linear_angle(t_abs, p) moves the source along a STRAIGHT
LINE through angle beta from vertical, instead of traj_inclined_circle's
tilted circle -- beta = 0 reproduces fig2_circle_summary.py's traj_linear
exactly (h modulated, D fixed), and beta = 90 is the horizontal analogue
(D modulated, h fixed).
"""

import numpy as np
import matplotlib.pyplot as plt

# ── constants (verbatim from fig2_circle_summary.py) ────────────────────────────
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

SHOT_NOISE_ASD = 1e-4   # rad/sqrt(Hz) -- reference level for the beta sweep (d)

# both noise floors, house style from fig2_noise_levels.py: baseline (lighter
# sand) plotted behind, stretch (darker tan) plotted on top, in panels a/b/c
NOISE_LEVELS = [
    dict(key='baseline', asd=1e-4, label='baseline (1e-4)', color='#d9c9a8', lw=0.3, zorder=2),
    dict(key='stretch',  asd=1e-5, label='stretch (1e-5)',  color='#9c8358', lw=0.35, zorder=3),
]

# ── geometry parameters specific to this figure ─────────────────────────────────
F_SLOW   = 0.02    # panel (b): sub-Nyquist, shows the nonlinearity comb
F_CIRC   = 0.131   # panels (c)/(d): shared super-Nyquist fundamental
BETA_DEG = [0.0, 30.0, 45.0, 60.0, 90.0, 120.0, 135.0, 150.0, 180.0]   # panel (d) sweep

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

# ── source trajectory ────────────────────────────────────────────────────────────
def traj_linear_angle(t_abs, p):
    """1D oscillator moving along a STRAIGHT LINE through angle beta from
    vertical: beta = 0 -> pure vertical motion (h modulated, D fixed,
    identical to fig2_circle_summary.py's traj_linear); beta = 90 -> pure
    horizontal motion (D modulated, h fixed). D >> amp always here, so the
    horizontal distance never changes sign and there is no need for a
    sqrt(x^2+y^2) (there is no second, out-of-line component -- unlike the
    tilted CIRCLE in fig2_circle_summary.py, this source never leaves the
    line)."""
    ph   = 2 * np.pi * p['f'] * t_abs
    beta = np.deg2rad(p['beta_deg'])
    disp = p['amp'] * np.cos(ph)
    h = p['h0'] + disp * np.cos(beta)
    x = p['D'] + disp * np.sin(beta)
    return h, x

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

src_b = dict(name='b', f=F_SLOW, traj=traj_linear_angle, beta_deg=0.0, **src_common)
src_c = dict(name='c', f=F_CIRC, traj=traj_linear_angle, beta_deg=90.0, **src_common)

def harmonic_table(label, src, f_k, S_urad, n_max=4):
    """Same format as fig2_circle_summary.py's harmonic_table()."""
    print(f"\nPanel ({label}): source f0 = {src['f']:.4f} Hz, M={src['M']}, "
          f"amp={src['amp']}, D={src['D']}, h0={src['h0']} "
          f"[{src['traj'].__name__}, beta={src.get('beta_deg', 'n/a')}]")
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

    print(f"Panel (b): vertical 1D oscillator f0={F_SLOW:.4f} Hz + ULDM "
          f"({N_cycles:,} cycles)...")
    raw_b = compute_phases(src_b) + uldm

    print(f"Panel (c): horizontal 1D oscillator f0={F_CIRC:.4f} Hz + ULDM "
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

    # ── beta sweep for panel (d): 2f0/f0 ratio vs. oscillator direction ─────────
    # beta = 90 is exactly src_c (already computed above) -- reuse it rather
    # than resimulating.
    ratio_2f0_f0 = {}
    peak_f0_beta = {}
    peak_2f0_beta = {}
    for be in BETA_DEG:
        if be == 90.0:
            f0_peak, f2_peak = rows['c'][0][3], rows['c'][1][3]
        else:
            src_be = dict(name=f'beta{be:g}', f=F_CIRC, traj=traj_linear_angle,
                          beta_deg=be, **src_common)
            print(f"Panel (d) sweep: beta={be:g} deg ({N_cycles:,} cycles)...")
            raw_be = compute_phases(src_be) + uldm
            _, S_be = periodogram(add_noise(raw_be, SHOT_NOISE_ASD))
            urad_be = np.sqrt(S_be) * 1e6
            rows_be = harmonic_table(f'd, {be:g} deg', src_be, f_k, urad_be, n_max=2)
            f0_peak, f2_peak = rows_be[0][3], rows_be[1][3]
        peak_f0_beta[be]  = f0_peak
        peak_2f0_beta[be] = f2_peak
        ratio_2f0_f0[be]  = f2_peak / f0_peak

    print(f"\n2f0/f0 folded-peak ratio vs. 1D oscillator direction "
          f"(0 deg = vertical, 90 deg = horizontal):")
    for be in BETA_DEG:
        print(f"  beta={be:>5.1f} deg   f0={peak_f0_beta[be]:>12.4g}   "
              f"2f0={peak_2f0_beta[be]:>12.4g}   ratio={ratio_2f0_f0[be]:.4f}")

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

    # panel (d): bar chart of the 2f0/f0 ratio vs. oscillator direction
    x = np.arange(len(BETA_DEG))
    bar_vals = [ratio_2f0_f0[be] for be in BETA_DEG]
    bars = ax_d.bar(x, bar_vals, color=COLORS['fundamental'], width=0.6)
    labels_d = []
    for be in BETA_DEG:
        if be == 0.0:
            labels_d.append('vertical\n(0°)')
        elif be == 90.0:
            labels_d.append('horizontal\n(90°)')
        elif be == 180.0:
            labels_d.append('vertical\n(180°)')
        else:
            labels_d.append(f'{be:g}°')
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
    print(f"  (b) vertical 1D oscillator, f0 = {F_SLOW} Hz (sub-Nyquist: nonlinearity comb, no folding).")
    print(f"  (c) horizontal 1D oscillator, f0 = {F_CIRC} Hz (super-Nyquist: comb folds).")
    print(f"  (d) 2f0/f0 folded-peak ratio for a 1D oscillator whose direction is rotated through "
          f"{', '.join(f'{be:g}' for be in BETA_DEG)} deg from vertical.")
    print("=" * 78)

    plt.show()
