"""
fig2_final.py: a compact 4-panel figure for the 100 m instrument, built on
fig2_circle_summary.py but with the ULDM tone and oscillator combined into
each periodogram panel instead of shown separately.

  (a) 1D vertical oscillator + ULDM together, at a sub-Nyquist f0 = F_SLOW
      so the whole comb sits below f_N and nothing folds. Shows the
      nonlinearity comb (f0, 2f0, 3f0, ...) from the 1/sqrt(D^2+(z-h)^2)
      potential, plus the ULDM reference tone at f_phi -- now placed near
      the middle of the [0, f_N] frequency axis rather than close to f_N.

  (b) Same combination (1D oscillator + ULDM), but the oscillator frequency
      F_CIRC is now super-Nyquist, so its whole comb (including the
      fundamental) folds back below f_N -- this is the Nyquist-folding
      demonstration panel.

  (c) 2D horizontal circle + ULDM, at the SAME fundamental frequency as
      panel (b) (F_CIRC), so its folded comb can be compared directly to
      panel (b)'s folded 1D comb. Same geometry as fig2_multi_directional.py
      panel (c) / fig2_circle_summary.py panel (c).

  (d) Line plot (not a bar chart): the ratio of the 2f0 peak to the f0 peak
      (both folded), for a circle inclined at theta = 0, 5, 10, ..., 360
      deg, tilting the circle about the D axis (the atoms-to-source line):
      0 deg is the D-w (horizontal) plane, 90 deg the D-h (vertical) plane.
      A horizontal circle's height never changes, so its motion is a
      pure cos(phi) in D -- its comb is symmetric and 2f0/f0 is small. A
      vertical circle modulates height and horizontal distance out of
      phase, which mixes more power into 2f0. Sweeping the full 0-360 deg
      range (rather than stopping at 180) shows the ratio retracing itself
      every 180 deg, which is a useful internal sanity check.

A second, separate figure extends panel (c): it overlays the folded combs
of the horizontal (incline=0, D-w plane) and vertical (incline=90, D-h plane) circles, both at
F_CIRC, on the same log-scale periodogram axes (no ULDM tone plotted here),
colour-coded so the two geometries' harmonic content can be compared
directly.

A third figure repeats the figure-1 layout (a, b, c, d) but folds the
horizontal/vertical comparison from figure 2 into panel (c) itself -- both
combs, plus the ULDM tone, on one set of axes -- instead of splitting it
into a separate plot.

Only ONE noise floor is used throughout (baseline, 1e-5 rad/sqrt(Hz)),
plotted in a light green rather than the sand/tan house colour used
elsewhere.

This script does NOT save any file -- it just calls plt.show() so the
figures appear in whatever interactive backend is active.

Provenance: constants, the ULDM tone, fold_frequency, the trajectory
helpers (atom_path/arm_positions), the potential V, the phase integral
inside compute_phases, add_noise, periodogram, harmonic_table and the
annotation helpers are all copied verbatim from fig2_circle_summary.py /
fig2_multi_directional.py / fig2_noise_levels.py.
"""

import time

import numpy as np
import matplotlib.pyplot as plt

# ── constants (verbatim from fig2_circle_summary.py) ────────────────────────────
m_atom  = 1.46e-25;  hbar = 1.055e-34
G       = 6.67e-11;  g    = 9.81
n       = 1000
v_kick  = n * hbar * (2 * np.pi / 698e-9) / m_atom
T       = 2.5         # half-interferometry time [s]
T_fly   = 2 * T
# launch speed [m/s]: chosen so the lower arm lands back on the ground exactly at the end
# of the sequence (net upward shift of the pulses is v_kick*T, so u0 = g*T_fly/2 - v_kick/2)
u0      = g * T_fly / 2 - v_kick / 2
T_cyc   = 5.0          # cycle time [s]
dt      = 0.01

N_fly   = int(round(T_fly / dt))
t_local = np.arange(N_fly) * dt

T_total  = 1e6   # total observation time [s]
N_cycles = int(T_total / T_cyc)
f_N      = 1.0 / (2 * T_cyc)   # Nyquist frequency = 0.1 Hz

delta_r = 70.0   # gradiometer baseline (matches 100m_oscillators)

# single noise floor: baseline, 1e-5 rad/sqrt(Hz), plotted light green
NOISE_LEVELS = [
    dict(key='baseline', asd=1e-5, label='shot noise (1e-5)', color='#a3d9a3', lw=0.35, zorder=3),
]

# ── geometry parameters specific to this figure ─────────────────────────────────
F_SLOW    = 0.02    # panel (a): sub-Nyquist, shows the unfolded nonlinearity comb
F_CIRC    = 0.092   # panels (b)/(c)/(d): shared super-Nyquist fundamental (folds)
THETA_DEG = np.arange(0.0, 360.0 + 1e-9, 5.0)    # panel (d) sweep, every 5 deg, full turn

# ── colour-by-physical-origin palette (verbatim) ────────────────────────────────
COLORS = dict(
    uldm        = '#1f4e9c',   # blue  -- f_phi (ULDM signal), always
    fundamental = '#c1272d',   # red   -- f0 and its harmonics, always
    harmonic    = '#c1272d',
    folded      = '#c1272d',
    noise       = '#a3d9a3',   # light green -- periodogram / noise floor
)

# ── ULDM parameters (verbatim, except f_phi) ─────────────────────────────────────
f_phi       = 0.05    # middle of the [0, f_N] = [0, 0.1] Hz frequency axis
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
    """Circle centred at (D, 0, h0), tilted by alpha about the D axis (the
    horizontal line from the atoms to the source). Axes: D = along that line,
    w = horizontal and perpendicular to it, h = vertical.
    alpha = 0  -> circle in the D-w plane (horizontal, height fixed);
    alpha = 90 -> circle in the D-h plane (vertical, w = 0)."""
    ph    = 2 * np.pi * p['f'] * t_abs
    alpha = np.deg2rad(p['incline_deg'])
    x = p['D'] + p['amp'] * np.cos(ph)                    # D coordinate
    w = p['amp'] * np.cos(alpha) * np.sin(ph)
    h = p['h0'] + p['amp'] * np.sin(alpha) * np.sin(ph)
    return h, np.sqrt(x*x + w*w)

# ── single progress bar for the whole program (an estimate) ────────────────────
# Work units are cycles: one compute_phases pass = N_cycles (dominant cost), each
# noise + periodogram = PERIODOGRAM_WORK * N_cycles, the final plotting/summary = N_cycles.
# log() prints a message above the bar and redraws it.
PERIODOGRAM_WORK = 0.05
N_PHASE_CALLS    = 4 + (len(THETA_DEG) - 1)     # a, b, c, vertical circle + sweep (theta = 0 reused)
N_PERIODOGRAMS   = 7 + (len(THETA_DEG) - 1)     # a, b, c, fig 2 (x2), fig 3 (x2) + sweep
total_work = (N_PHASE_CALLS + PERIODOGRAM_WORK * N_PERIODOGRAMS + 1) * N_cycles
done_work, t_start, _last_bar = 0.0, time.time(), ''

def show_progress(label):
    global _last_bar
    frac = min(done_work / total_work, 1.0)
    el = time.time() - t_start
    eta = el * (1 - frac) / frac if frac > 0 else float('nan')
    bar = '#' * int(30 * frac) + '-' * (30 - int(30 * frac))
    _last_bar = chr(13) + f"[{bar}] {100*frac:5.1f}%  elapsed {el/60:7.1f} min  ETA {eta/60:7.1f} min  {label}   "
    print(_last_bar, end='', flush=True)

def log(*args, **kwargs):
    """print() that keeps the progress bar on the last line."""
    if _last_bar:
        print(chr(13) + ' ' * 110 + chr(13), end='')
    print(*args, **kwargs, flush=True)
    if _last_bar:
        print(_last_bar, end='', flush=True)

def compute_phases(src, chunk=25_000):
    global done_work
    phases = np.empty(N_cycles)
    for start in range(0, N_cycles, chunk):
        end   = min(start + chunk, N_cycles)
        t_abs = t_starts[start:end][:, None] + t_local[None, :]
        h_osc, D_osc = src['traj'](t_abs, src)
        dphi  = ((V(upper1, h_osc, src['M'], D_osc) - V(lower1, h_osc, src['M'], D_osc))
               - (V(upper2, h_osc, src['M'], D_osc) - V(lower2, h_osc, src['M'], D_osc)))
        phases[start:end] = (m_atom / hbar) * np.trapezoid(dphi, t_local, axis=1)
        done_work += end - start
        show_progress(f"phases: {src['name']}")
    return phases

def add_noise(phases, shot_noise_asd):
    return phases + np.random.normal(0.0, shot_noise_asd / np.sqrt(T_cyc), size=N_cycles)

def periodogram(ph):
    global done_work
    done_work += PERIODOGRAM_WORK * N_cycles
    show_progress('spectra')
    f_k  = np.fft.rfftfreq(N_cycles, d=T_cyc)
    S_k  = (T_cyc / N_cycles) * np.abs(np.fft.rfft(ph - ph.mean()))**2
    mask = f_k > 0
    return f_k[mask], S_k[mask]

def peak_at(f_k, S_k_urad, f_target):
    idx = np.argmin(np.abs(f_k - f_target))
    return f_k[idx], S_k_urad[idx]

# ── source definitions ──────────────────────────────────────────────────────────
src_common = dict(amp=0.5, M=1.0, D=3.0, h0=1.0)

src_a = dict(name='a', f=F_SLOW, traj=traj_linear, **src_common)
src_b = dict(name='b', f=F_CIRC, traj=traj_linear, **src_common)
src_c = dict(name='c', f=F_CIRC, traj=traj_inclined_circle, incline_deg=0.0,
             **src_common)
src_c_vert = dict(name='c_vert', f=F_CIRC, traj=traj_inclined_circle, incline_deg=90.0,
                   **src_common)

def harmonic_table(label, src, f_k, S_urad, n_max=4):
    """Same format as fig2_circle_summary.py's harmonic_table() (no
    sideband splitting needed here -- no pendulum in this figure)."""
    log(f"\nPanel ({label}): source f0 = {src['f']:.4f} Hz, M={src['M']}, "
          f"amp={src['amp']}, D={src['D']}, h0={src['h0']} "
          f"[{src['traj'].__name__}, incline={src.get('incline_deg', 'n/a')}]")
    log(f"  {'n':>2}  {'n*f0 [Hz]':>10}  {'folded [Hz]':>12}  {'peak [urad/rtHz]':>18}  status")
    rows = []
    for k in range(1, n_max + 1):
        f_true = k * src['f']
        f_fold = fold_frequency(f_true, f_N)
        f_bin, peak = peak_at(f_k, S_urad, f_fold)
        status = 'genuine (< f_N)' if f_true <= f_N else 'folded/aliased'
        log(f"  {k:>2}  {f_true:>10.4f}  {f_fold:>12.4f}  {peak:>18.4g}  {status}")
        rows.append((k, f_true, f_fold, peak, f_bin))
    return rows

if __name__ == "__main__":
    # ── one interferometer cycle: arm paths of both gradiometer ends ────────────
    # Shown immediately (non-blocking) so it doesn't wait for the long simulation below.
    fig_paths, axs_p = plt.subplots(2, 1, figsize=(6, 6), dpi=200, sharex=True)
    for ax_p, (name, lo, up) in zip(axs_p[::-1], (('IFO 1 (z = 0)', lower1, upper1),
                                                  (f'IFO 2 (z = {delta_r:g} m)', lower2, upper2))):
        ax_p.plot(t_local, lo, color='k', lw=1.2, label='lower arm')
        ax_p.plot(t_local, up, color=COLORS['fundamental'], lw=1.2, label='upper arm')
        for t_pulse in (0.0, T, T_fly):
            ax_p.axvline(t_pulse, color='k', ls=':', lw=0.5, alpha=0.5)
        ax_p.set_ylabel('height $z$ [m]', fontsize=12)
        ax_p.set_title(name, fontsize=9, fontweight='bold')
        ax_p.tick_params(labelsize=11)
        ax_p.legend(fontsize=8, loc='lower center', frameon=False)
    axs_p[-1].set_xlabel('time within one cycle [s]', fontsize=12)
    axs_p[-1].set_xlim(0, T_fly)
    fig_paths.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    uldm = uldm_phase_vec(t_meas, f_phi, m_phi)

    log(f"Panel (a): traj_linear f0={F_SLOW:.4f} Hz + ULDM ({N_cycles:,} cycles)...")
    raw_a = compute_phases(src_a) + uldm

    log(f"Panel (b): traj_linear f0={F_CIRC:.4f} Hz (folds) + ULDM "
          f"({N_cycles:,} cycles)...")
    raw_b = compute_phases(src_b) + uldm

    log(f"Panel (c): horizontal circle f0={F_CIRC:.4f} Hz (folds) + ULDM "
          f"({N_cycles:,} cycles)...")
    phases_c = compute_phases(src_c)
    raw_c = phases_c + uldm

    log(f"Figure 2: vertical circle f0={F_CIRC:.4f} Hz (folds), no ULDM "
          f"({N_cycles:,} cycles)...")
    phases_c_vert = compute_phases(src_c_vert)

    # single noise floor, for each of panels a/b/c
    raws = {'a': raw_a, 'b': raw_b, 'c': raw_c}
    urad_by_level = {letter: {} for letter in 'abc'}
    f_k = None
    for letter, raw in raws.items():
        for lvl in NOISE_LEVELS:
            f_k, S = periodogram(add_noise(raw, lvl['asd']))
            urad_by_level[letter][lvl['key']] = np.sqrt(S) * 1e6
    urad = {letter: urad_by_level[letter]['baseline'] for letter in 'abc'}

    rows = {}
    rows['a'] = harmonic_table('a', src_a, f_k, urad['a'])
    rows['b'] = harmonic_table('b', src_b, f_k, urad['b'])
    rows['c'] = harmonic_table('c', src_c, f_k, urad['c'])

    f_phi_bins = {k: peak_at(f_k, u, f_phi) for k, u in urad.items()}
    log(f"\nULDM peak at f_phi = {f_phi} Hz, per panel:")
    for k in 'abc':
        log(f"  ({k})  {f_phi_bins[k][1]:>12.4g} urad/rtHz")
    log(f"\nShot-noise floor = {NOISE_LEVELS[0]['asd']*1e6:.4g} urad/rtHz")

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
            log(f"Panel (d) sweep: incline={th:g} deg "
                  f"({N_cycles:,} cycles)...")
            raw_th = compute_phases(src_th) + uldm
            _, S_th = periodogram(add_noise(raw_th, NOISE_LEVELS[0]['asd']))
            urad_th = np.sqrt(S_th) * 1e6
            rows_th = harmonic_table(f'd, {th:g} deg', src_th, f_k, urad_th, n_max=2)
            f0_peak, f2_peak = rows_th[0][3], rows_th[1][3]
        peak_f0_theta[th]  = f0_peak
        peak_2f0_theta[th] = f2_peak
        ratio_2f0_f0[th]   = f2_peak / f0_peak

    log(f"\n2f0/f0 folded-peak ratio vs. circle inclination "
          f"(0 deg = horizontal, 90 deg = vertical, full 0-360 deg sweep):")
    for th in THETA_DEG:
        log(f"  theta={th:>5.1f} deg   f0={peak_f0_theta[th]:>12.4g}   "
              f"2f0={peak_2f0_theta[th]:>12.4g}   ratio={ratio_2f0_f0[th]:.4f}")

    # ── plot: 3 stacked periodogram panels (a,b,c) + 1 line graph (d) ───────────
    fig = plt.figure(figsize=(6.5, 15), dpi=200)
    gs  = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.1], hspace=0.55,
                            left=0.16, right=0.95)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2], sharex=ax_a)
    ax_d = fig.add_subplot(gs[3])

    y_max = 3 * max(max(r[3] for r in rows[k][:4]) for k in ('a', 'b', 'c'))
    y_max = max(y_max, 3 * max(p for _, p in f_phi_bins.values()))
    y_min = 0.1 * NOISE_LEVELS[0]['asd'] * 1e6

    PANEL_TITLES = {
        'a': '1D oscillator and ULDM',
        'b': 'Post-Nyquist folding',
        'c': '2D horizontal circle',
        'd': 'Tilt angle sweep',
    }

    for ax in (ax_a, ax_b, ax_c):
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, f_N)
        ax.tick_params(labelsize=11)
        ax.axvline(f_N, color='k', ls='--', lw=0.5, alpha=0.6)
    ax_b.set_ylabel(r'$\sqrt{S_k}$ [$\mu$rad/$\sqrt{\mathrm{Hz}}$]', fontsize=12)
    for letter, ax in zip('abc', (ax_a, ax_b, ax_c)):
        ax.set_title(PANEL_TITLES[letter], fontsize=9, fontweight='bold')
    ax_c.set_xlabel('$f$ [Hz]', fontsize=12)
    ax_a.text(f_N, y_max*0.5, r'  $f_N$', fontsize=9, ha='left', va='center')

    def mark_uldm(ax, letter, xytext=(6, 6)):
        f_bin, peak = f_phi_bins[letter]
        ax.vlines(f_bin, y_min, peak, color=COLORS['uldm'], lw=0.6, zorder=4)
        ax.scatter([f_bin], [peak], color=COLORS['uldm'], zorder=5, s=8, marker='D')
        ax.annotate(r'$f_\varphi$', (f_bin, peak), textcoords='offset points',
                    xytext=xytext, fontsize=8, color=COLORS['uldm'])

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
                        xytext=offsets.get(k, (4, 6)), fontsize=8, color=col)

    def plot_noise_stack(ax, letter):
        for lvl in NOISE_LEVELS:
            ax.plot(f_k, urad_by_level[letter][lvl['key']], color=lvl['color'],
                    lw=lvl['lw'], zorder=lvl['zorder'])

    plot_noise_stack(ax_a, 'a')
    mark_harmonics(ax_a, 'a', folded=False,
                   offsets={1: (4, -8), 2: (4, 6), 3: (4, 6), 4: (5, -8)})
    mark_uldm(ax_a, 'a', xytext=(6, -6))
    ax_a.legend(handles=[plt.Line2D([], [], color=lvl['color'], lw=1.5,
                                     label=lvl['label']) for lvl in NOISE_LEVELS],
                fontsize=8, loc='upper right', frameon=False)

    plot_noise_stack(ax_b, 'b')
    mark_harmonics(ax_b, 'b', folded=True,
                   offsets={1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)})
    mark_uldm(ax_b, 'b', xytext=(6, -6))

    plot_noise_stack(ax_c, 'c')
    mark_harmonics(ax_c, 'c', folded=True,
                   offsets={1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)})
    mark_uldm(ax_c, 'c', xytext=(6, -6))

    # panel (d): line plot of the 2f0/f0 ratio vs. inclination, every 10 deg
    line_vals = [ratio_2f0_f0[th] for th in THETA_DEG]
    ax_d.plot(THETA_DEG, line_vals, color=COLORS['fundamental'], lw=1.2)
    ax_d.set_xlim(0, 360)
    ax_d.set_xticks(np.arange(0, 361, 45))
    ax_d.set_xlabel('circle inclination $\\theta$ [deg] (0=horiz., 90=vert.)', fontsize=12)
    ax_d.set_ylabel(r'$\sqrt{S_{2f_0}}\,/\,\sqrt{S_{f_0}}$', fontsize=12)
    ax_d.set_ylim(0, 1.3 * max(line_vals))
    ax_d.tick_params(labelsize=11)
    ax_d.set_title(PANEL_TITLES['d'], fontsize=9, fontweight='bold', pad=10)
    ax_d.axvline(90, color='k', ls=':', lw=0.4, alpha=0.5)
    ax_d.axvline(270, color='k', ls=':', lw=0.4, alpha=0.5)
    ax_d.axhline(ratio_2f0_f0[90.0], color='k', ls=':', lw=0.4, alpha=0.5)

    # ── figure 2: horizontal vs. vertical circle comb, overlaid ─────────────────
    # one shared noise realization for both traces, so they sit on the same
    # noise floor rather than each carrying its own independent draw.
    noise_vec  = np.random.normal(0.0, NOISE_LEVELS[0]['asd'] / np.sqrt(T_cyc), size=N_cycles)
    _, S_horiz = periodogram(phases_c + noise_vec)
    urad_horiz = np.sqrt(S_horiz) * 1e6
    _, S_vert  = periodogram(phases_c_vert + noise_vec)
    urad_vert  = np.sqrt(S_vert) * 1e6

    rows_horiz = harmonic_table('fig2, horizontal', src_c, f_k, urad_horiz)
    rows_vert  = harmonic_table('fig2, vertical', src_c_vert, f_k, urad_vert)

    COLOR_HORIZ = COLORS['fundamental']   # red, matches panel (c)
    COLOR_VERT  = COLORS['uldm']          # blue, visually distinct from horizontal

    fig2, ax2 = plt.subplots(figsize=(6, 4), dpi=200)
    ax2.set_yscale('log')
    ax2.set_xlim(0, f_N)
    y_min2 = 0.1 * NOISE_LEVELS[0]['asd'] * 1e6
    y_max2 = 3 * max(max(r[3] for r in rows_horiz), max(r[3] for r in rows_vert))
    ax2.set_ylim(y_min2, y_max2)
    ax2.axvline(f_N, color='k', ls='--', lw=0.5, alpha=0.6)
    ax2.plot(f_k, urad_horiz, color=COLOR_HORIZ, lw=0.5, alpha=0.5,
             label='horizontal circle (incline=0°, D-w plane)')
    ax2.plot(f_k, urad_vert, color=COLOR_VERT, lw=0.5, alpha=0.5,
             label='vertical circle (incline=90°, D-h plane)')

    for rows_geom, col in ((rows_horiz, COLOR_HORIZ), (rows_vert, COLOR_VERT)):
        for k, f_true, f_fold, peak, f_bin in rows_geom:
            ax2.vlines(f_bin, y_min2, peak, color=col, lw=0.6, zorder=4)
            ax2.scatter([f_bin], [peak], color=col, zorder=5, s=10)
            base = r'$f_0$' if k == 1 else rf'${k}f_0$'
            label = base + ' (folded)' if f_true > f_N else base
            ax2.annotate(label, (f_bin, peak), textcoords='offset points',
                         xytext=(4, 6), fontsize=6, color=col)

    ax2.set_xlabel('$f$ [Hz]', fontsize=9)
    ax2.set_ylabel(r'$\sqrt{S_k}$ [$\mu$rad/$\sqrt{\mathrm{Hz}}$]', fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.legend(fontsize=7, loc='upper right', frameon=False)
    ax2.set_title(f'Folded comb: horizontal vs. vertical circle, $f_0$={F_CIRC:g} Hz',
                  fontsize=9)
    fig2.tight_layout()

    # ── figure 3: same 4-panel layout as figure 1, but panel (c) overlays the
    # horizontal and vertical circle combs (as in figure 2) instead of showing
    # only the horizontal circle. Both traces include the ULDM tone and share
    # one noise realization, so they sit on equal footing.
    raw_horiz_c3 = phases_c + uldm + noise_vec
    raw_vert_c3  = phases_c_vert + uldm + noise_vec
    _, S_horiz_c3 = periodogram(raw_horiz_c3)
    _, S_vert_c3  = periodogram(raw_vert_c3)
    urad_horiz_c3 = np.sqrt(S_horiz_c3) * 1e6
    urad_vert_c3  = np.sqrt(S_vert_c3) * 1e6

    rows_horiz_c3 = harmonic_table('fig3 (c), horizontal', src_c, f_k, urad_horiz_c3)
    rows_vert_c3  = harmonic_table('fig3 (c), vertical', src_c_vert, f_k, urad_vert_c3)
    f_phi_bin_c3  = peak_at(f_k, urad_horiz_c3, f_phi)

    COLOR_VERT_C3 = '#e07b00'   # orange -- distinct from red (horizontal) and
                                 # blue (ULDM marker, which now also appears here)

    fig3 = plt.figure(figsize=(6.5, 15), dpi=200)
    gs3  = fig3.add_gridspec(4, 1, height_ratios=[1, 1, 1, 1.1], hspace=0.55,
                              left=0.16, right=0.95)
    ax_a3 = fig3.add_subplot(gs3[0])
    ax_b3 = fig3.add_subplot(gs3[1], sharex=ax_a3)
    ax_c3 = fig3.add_subplot(gs3[2], sharex=ax_a3)
    ax_d3 = fig3.add_subplot(gs3[3])

    for ax in (ax_a3, ax_b3, ax_c3):
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, f_N)
        ax.tick_params(labelsize=11)
        ax.axvline(f_N, color='k', ls='--', lw=0.5, alpha=0.6)
    ax_b3.set_ylabel(r'$\sqrt{S_k}$ [$\mu$rad/$\sqrt{\mathrm{Hz}}$]', fontsize=12)
    ax_a3.text(f_N, y_max*0.5, r'  $f_N$', fontsize=9, ha='left', va='center')

    PANEL_TITLES_FIG3 = dict(PANEL_TITLES, c='2D circle: horizontal vs. vertical')
    for letter, ax in zip('abc', (ax_a3, ax_b3, ax_c3)):
        ax.set_title(PANEL_TITLES_FIG3[letter], fontsize=9, fontweight='bold')
    ax_c3.set_xlabel('$f$ [Hz]', fontsize=12)

    plot_noise_stack(ax_a3, 'a')
    mark_harmonics(ax_a3, 'a', folded=False,
                   offsets={1: (4, -8), 2: (4, 6), 3: (4, 6), 4: (5, -8)})
    mark_uldm(ax_a3, 'a', xytext=(6, -6))
    ax_a3.legend(handles=[plt.Line2D([], [], color=lvl['color'], lw=1.5,
                                      label=lvl['label']) for lvl in NOISE_LEVELS],
                 fontsize=8, loc='upper right', frameon=False)

    plot_noise_stack(ax_b3, 'b')
    mark_harmonics(ax_b3, 'b', folded=True,
                   offsets={1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)})
    mark_uldm(ax_b3, 'b', xytext=(6, -6))

    # panel (c): ONE noisy periodogram (green, like panels a/b -- the horizontal
    # and vertical traces share the same noise realization, so plotting both in
    # full would just show the same noise floor twice) with the horizontal and
    # vertical circle harmonics marked on top, colour-coded. Horizontal and
    # vertical share the same fundamental (F_CIRC), so their harmonic orders
    # land at the same frequencies -- label each order (f0, 2f0, ...) ONCE,
    # at fontsize=8 to match panels (a)/(b), rather than once per geometry.
    for lvl in NOISE_LEVELS:
        ax_c3.plot(f_k, urad_horiz_c3, color=lvl['color'], lw=lvl['lw'], zorder=lvl['zorder'])
    c3_offsets = {1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)}
    for k, ((_, f_true_h, f_fold_h, peak_h, f_bin_h),
            (_, f_true_v, f_fold_v, peak_v, f_bin_v)) in enumerate(
                zip(rows_horiz_c3[:4], rows_vert_c3[:4]), start=1):
        if abs(f_fold_h - f_phi_bin_c3[0]) < 2 * (f_k[1] - f_k[0]):
            continue
        ax_c3.vlines(f_bin_h, y_min, peak_h, color=COLOR_HORIZ, lw=0.6, zorder=4)
        ax_c3.scatter([f_bin_h], [peak_h], color=COLOR_HORIZ, zorder=5, s=7)
        ax_c3.vlines(f_bin_v, y_min, peak_v, color=COLOR_VERT_C3, lw=0.6, zorder=4)
        ax_c3.scatter([f_bin_v], [peak_v], color=COLOR_VERT_C3, zorder=5, s=7)
        base = r'$f_0$' if k == 1 else rf'${k}f_0$'
        label = base + ' (folded)' if f_true_h > f_N else base
        top_bin, top_peak = (f_bin_h, peak_h) if peak_h >= peak_v else (f_bin_v, peak_v)
        ax_c3.annotate(label, (top_bin, top_peak), textcoords='offset points',
                       xytext=c3_offsets.get(k, (4, 6)), fontsize=8, color='black')
    ax_c3.vlines(f_phi_bin_c3[0], y_min, f_phi_bin_c3[1], color=COLORS['uldm'],
                 lw=0.6, zorder=4)
    ax_c3.scatter([f_phi_bin_c3[0]], [f_phi_bin_c3[1]], color=COLORS['uldm'],
                  zorder=5, s=8, marker='D')
    ax_c3.annotate(r'$f_\varphi$', f_phi_bin_c3, textcoords='offset points',
                   xytext=(6, -6), fontsize=8, color=COLORS['uldm'])
    ax_c3.legend(handles=[
        plt.Line2D([], [], color=COLOR_HORIZ, lw=1.5, label='horizontal'),
        plt.Line2D([], [], color=COLOR_VERT_C3, lw=1.5, label='vertical'),
    ], fontsize=6, loc='upper center', frameon=False)

    # panel (d): identical tilt-angle sweep as figure 1
    ax_d3.plot(THETA_DEG, line_vals, color=COLORS['fundamental'], lw=1.2)
    ax_d3.set_xlim(0, 360)
    ax_d3.set_xticks(np.arange(0, 361, 45))
    ax_d3.set_xlabel('circle inclination $\\theta$ [deg] (0=horiz., 90=vert.)', fontsize=12)
    ax_d3.set_ylabel(r'$\sqrt{S_{2f_0}}\,/\,\sqrt{S_{f_0}}$', fontsize=12)
    ax_d3.set_ylim(0, 1.3 * max(line_vals))
    ax_d3.tick_params(labelsize=11)
    ax_d3.set_title(PANEL_TITLES['d'], fontsize=9, fontweight='bold', pad=10)
    ax_d3.axvline(90, color='k', ls=':', lw=0.4, alpha=0.5)
    ax_d3.axvline(270, color='k', ls=':', lw=0.4, alpha=0.5)
    ax_d3.axhline(ratio_2f0_f0[90.0], color='k', ls=':', lw=0.4, alpha=0.5)

    log("\n" + "=" * 78)
    log("Suggested caption:")
    log(f"  dt = T_cyc = {T_cyc:g} s, N_cycles = {N_cycles:,}, T_obs = {T_total:.3g} s, "
          f"f_N = {f_N:g} Hz, shot noise = {NOISE_LEVELS[0]['asd']:.0e} rad/sqrt(Hz) in all panels.")
    log(f"  ULDM: f_phi = {f_phi} Hz, m_phi = {m_phi:.4g} eV (present in every panel).")
    log(f"  Source: M = {src_common['M']}, amp = {src_common['amp']} m, "
          f"D = {src_common['D']} m, h0 = {src_common['h0']} m.")
    log(f"  (a) 1D oscillator + ULDM, f0 = {F_SLOW} Hz (sub-Nyquist: nonlinearity comb, no folding).")
    log(f"  (b) 1D oscillator + ULDM, f0 = {F_CIRC} Hz (super-Nyquist: comb folds).")
    log(f"  (c) horizontal circle + ULDM, f0 = {F_CIRC} Hz (super-Nyquist: comb folds).")
    log(f"  (d) 2f0/f0 folded-peak ratio for a circle inclined at 0-360 deg (every 10 deg) "
          f"out of the horizontal.")
    log(f"  Figure 3: as figure 1, but panel (c) overlays the horizontal and vertical "
          f"circle combs (+ ULDM) instead of showing the horizontal circle alone.")
    log("=" * 78)

    done_work = total_work;  show_progress('done');  print()
    plt.show()
