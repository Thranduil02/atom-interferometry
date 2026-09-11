"""
fig2_multi_directional.py: how the *direction* of the source-mass motion
shows up in the gradiometer periodogram, for the 100 m instrument.

Six stacked panels, same style as fig2_noise_levels.py / fig2_folding_map.py
(log-y periodogram, tan noise trace, red stems + markers for the oscillator
comb, blue diamond for the ULDM tone). Shot noise at SHOT_NOISE_ASD is
included in every panel, and the ULDM tone is present in every panel:

  (a) ULDM + noise only -- the reference: one clean tone at f_phi over a
      flat shot-noise floor, nothing else.

  (b) 1D vertical oscillator + ULDM, at a sub-Nyquist f0 = 0.02 Hz so the
      whole comb sits below f_N and nothing folds. This panel is about
      NONLINEARITY, not folding: the 1/sqrt(D^2+(z-h)^2) potential turns a
      pure-cosine source motion into a harmonic comb f0, 2f0, 3f0, ... whose
      amplitudes fall off geometrically. Nothing here is aliased -- every
      line is where the source really put it.

  (c) 2D horizontal circle + ULDM, f0 = F_CIRC = 0.131 Hz (super-Nyquist, so
      its comb FOLDS back below f_N). The mass runs round a horizontal
      circle of radius amp about a centre at horizontal distance D: its
      height never changes, only its horizontal distance from the
      gradiometer axis does.

  (d) 2D vertical circle + ULDM, same f0 = F_CIRC as (c) so the two are
      directly comparable. This is the circular_oscillator / 100m_circular
      geometry: the mass runs round a circle in the vertical plane
      containing the gradiometer axis, so both its height and its horizontal
      distance are modulated.

  (e) Circle inclined at INCLINE_DEG = 45 deg + ULDM, same f0 = F_CIRC --
      the same circle family as (c), tilted out of the horizontal, so it
      interpolates between (c) (0 deg) and a vertical circle (90 deg).

  (f) 3D spherical pendulum under gravity + ULDM. The frequency here is NOT
      a free parameter -- it is set by the pendulum length, f0 =
      sqrt(g/L_PEND)/2pi, and L_PEND = 14.5 m is chosen so that f0 comes out
      at ~0.1309 Hz, i.e. essentially the same fundamental as (c)-(e), so
      all four geometries are compared at one frequency. The bob traces a
      slowly precessing ellipse (small-amplitude spherical pendulum with the
      leading nonlinear precession, Omega = (3/8) w0 A B / L^2), and its
      HEIGHT oscillates at 2*f0 rather than f0, since the bob rises at both
      ends of every swing.

(c) and (d) deliberately share a frequency, per the geometry-comparison
point of the figure. F_CIRC = 0.131 Hz (not the 0.1425 Hz used elsewhere in
this repo) so that no harmonic folds onto f_phi: this figure is about
telling the geometries apart, not about the folded-harmonic/ULDM degeneracy,
which fig2_noise_levels.py / fig2_folding_map.py already cover.

The script saves TWO versions of the figure from the same computed data: the
usual log-y periodogram (fig2_multi_directional.jpg) and a second, linear-y
version (fig2_multi_directional_linear.jpg) with the same six panels, same
markers/labels, same styling -- just set_yscale('linear') instead of 'log'.
Since a shared linear range would flatten the smaller panels flat next to
the tallest harmonics, each panel on the linear version gets its own y-range
sized to its own tallest marked peak (with 15% headroom) rather than sharing
one global range the way the log panels do.

A THIRD output (fig2_multi_directional_comparison.jpg) is a single bar chart
of the fundamental's peak height for each geometry (b/c/d/e/f), all on one
shared linear axis. The periodogram panels' log-y axis has to span 4+ decades
to fit the ULDM peak and the noise floor on the same plot, which makes the
(only ~2-3x) height differences BETWEEN geometries -- the actual point of
this figure -- occupy a small, easy-to-miss fraction of a decade. The bar
chart puts that ratio front and centre instead.

Provenance: the constants, the ULDM tone, fold_frequency, the trajectory
helpers (atom_path/arm_positions), the potential V, the phase integral
inside compute_phases, add_noise and periodogram are all copied verbatim
from fig2_noise_levels.py. The ONE generalization is that compute_phases now
takes a trajectory callable returning (h, D_horizontal) for the source at
each instant, instead of hard-coding the linear/circular pair -- the phase
integral it feeds is unchanged, and traj_linear/traj_vertical_circle
reproduce fig2_noise_levels.py's circular=False/circular=True cases exactly.
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

T_total  = 3e6   # total observation time [s] (~1 year)
N_cycles = int(T_total / T_cyc)
f_N      = 1.0 / (2 * T_cyc)   # Nyquist frequency = 0.1 Hz

delta_r = 70.0   # gradiometer baseline (matches 100m_oscillators)

SHOT_NOISE_ASD = 1e-4   # rad/sqrt(Hz), included in every panel

# ── geometry parameters specific to this figure ─────────────────────────────────
F_SLOW      = 0.02    # panel (b): sub-Nyquist, shows the nonlinearity comb
F_CIRC      = 0.131   # panels (c)/(d)/(e): shared super-Nyquist fundamental
INCLINE_DEG = 45.0    # panel (e): tilt of the circle out of the horizontal
L_PEND      = 14.5    # panel (f): pendulum length [m] -> f0 = sqrt(g/L)/2pi
A_PEND      = 0.5     # semi-major axis of the pendulum's ellipse [m]
B_PEND      = 0.25    # semi-minor axis [m]

# ── colour-by-physical-origin palette (verbatim from fig2_noise_levels.py) ──────
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

# ── source trajectories ─────────────────────────────────────────────────────────
# Each returns (h, D_horizontal) for the source mass at absolute times t_abs.
# V() only ever sees the source height h and its horizontal distance D from the
# gradiometer axis, so a fully 3D source is captured by these two numbers:
# D = sqrt(x^2 + y^2) with the axis at x = y = 0.

def traj_linear(t_abs, p):
    """1D vertical oscillator -- identical to fig2_noise_levels.py's
    circular=False branch: height modulated, horizontal distance fixed."""
    ph = 2 * np.pi * p['f'] * t_abs
    return p['h0'] + p['amp'] * np.cos(ph), p['D']

def traj_vertical_circle(t_abs, p):
    """2D circle in the vertical plane containing the gradiometer axis --
    identical to fig2_noise_levels.py's circular=True branch."""
    ph = 2 * np.pi * p['f'] * t_abs
    return p['h0'] + p['amp'] * np.sin(ph), p['D'] + p['amp'] * np.cos(ph)

def traj_horizontal_circle(t_abs, p):
    """2D circle in the horizontal plane: height fixed, only the horizontal
    distance from the axis is modulated (and only through cos, so the
    modulation is symmetric about the near/far points)."""
    ph = 2 * np.pi * p['f'] * t_abs
    x = p['D'] + p['amp'] * np.cos(ph)
    y = p['amp'] * np.sin(ph)
    return p['h0'], np.sqrt(x*x + y*y)

def traj_inclined_circle(t_abs, p):
    """Circle whose plane is tilted by alpha out of the horizontal, about the
    transverse horizontal axis. alpha = 0 reproduces traj_horizontal_circle."""
    ph    = 2 * np.pi * p['f'] * t_abs
    alpha = np.deg2rad(p['incline_deg'])
    x = p['D'] + p['amp'] * np.cos(alpha) * np.cos(ph)
    y = p['amp'] * np.sin(ph)
    h = p['h0'] + p['amp'] * np.sin(alpha) * np.cos(ph)
    return h, np.sqrt(x*x + y*y)

def traj_pendulum(t_abs, p):
    """3D (spherical) pendulum under gravity, small-amplitude solution with
    the leading nonlinear precession: the bob traces an ellipse (semi-axes
    A, B) whose plane precesses at Omega = (3/8) w0 A B / L^2. The bob hangs
    a distance sqrt(L^2 - X^2 - Y^2) below the pivot, so its HEIGHT runs at
    2*f0 while its horizontal position runs at f0."""
    w0    = np.sqrt(g / p['L'])
    Omega = 0.375 * w0 * p['A'] * p['B'] / p['L']**2
    x0 = p['A'] * np.cos(w0 * t_abs)
    y0 = p['B'] * np.sin(w0 * t_abs)
    cO = np.cos(Omega * t_abs);  sO = np.sin(Omega * t_abs)
    X  = x0 * cO - y0 * sO
    Y  = x0 * sO + y0 * cO
    # pivot placed L above the nominal source height, so the bob hangs at ~h0
    h  = p['h0'] + p['L'] - np.sqrt(p['L']**2 - X*X - Y*Y)
    x  = p['D'] + X
    return h, np.sqrt(x*x + Y*Y)

def compute_phases(src, chunk=25_000):
    """Vectorized, chunked over cycles. The phase integral, the arm paths and
    V() are verbatim from fig2_noise_levels.py -- the only generalization is
    that the source position comes from src['traj'](t_abs, src) instead of a
    hard-coded linear/circular pair."""
    phases = np.empty(N_cycles)
    n_chunks = (N_cycles + chunk - 1) // chunk
    for i, start in enumerate(range(0, N_cycles, chunk), start=1):
        end   = min(start + chunk, N_cycles)
        t_abs = t_starts[start:end][:, None] + t_local[None, :]
        h_osc, D_osc = src['traj'](t_abs, src)
        dphi  = ((V(upper1, h_osc, src['M'], D_osc) - V(lower1, h_osc, src['M'], D_osc))
               - (V(upper2, h_osc, src['M'], D_osc) - V(lower2, h_osc, src['M'], D_osc)))
        phases[start:end] = (m_atom / hbar) * np.trapezoid(dphi, t_local, axis=1)
        if i % 20 == 0 or end == N_cycles:
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

def peak_near(f_k, S_k_urad, f_target, half_width=0.0):
    """Largest bin within +/- half_width of f_target (half_width = 0 falls back
    to the nearest single bin, i.e. peak_at). Needed for the pendulum: its
    ellipse precesses at Omega, so the lab-frame horizontal motion is a
    DOUBLET at f0 +/- Omega/2pi rather than a single line at f0, and a marker
    pinned to exactly f0 would land in the valley between the two sidebands
    and report a peak height an order of magnitude too small."""
    if half_width <= 0:
        return peak_at(f_k, S_k_urad, f_target)
    sel = np.abs(f_k - f_target) <= half_width
    if not sel.any():
        return peak_at(f_k, S_k_urad, f_target)
    idx = np.flatnonzero(sel)[np.argmax(S_k_urad[sel])]
    return f_k[idx], S_k_urad[idx]

# ── source definitions (same geometry constants as the other fig2_* scripts) ────
src_common = dict(amp=0.5, M=1.0, D=3.0, h0=1.0)
f_pend = np.sqrt(g / L_PEND) / (2 * np.pi)   # set by gravity, not chosen

src_b = dict(name='b', f=F_SLOW, traj=traj_linear,            **src_common)
src_c = dict(name='c', f=F_CIRC, traj=traj_horizontal_circle, **src_common)
src_d = dict(name='d', f=F_CIRC, traj=traj_vertical_circle,   **src_common)
src_e = dict(name='e', f=F_CIRC, traj=traj_inclined_circle,
             incline_deg=INCLINE_DEG, **src_common)
src_f = dict(name='f', f=f_pend, traj=traj_pendulum,
             L=L_PEND, A=A_PEND, B=B_PEND, **src_common)

def harmonic_table(label, src, f_k, S_urad, n_max=5, split_per_n=0.0):
    """Same format as fig2_noise_levels.py's harmonic_table(), plus the bin the
    peak was actually found in. split_per_n > 0 widens the search to
    +/- 3*n*split_per_n around the n-th harmonic, for sources whose lines are
    split into sidebands (the precessing pendulum)."""
    print(f"\nPanel ({label}): source f0 = {src['f']:.4f} Hz, M={src['M']}, "
          f"amp={src['amp']}, D={src['D']}, h0={src['h0']} "
          f"[{src['traj'].__name__}]")
    print(f"  {'n':>2}  {'n*f0 [Hz]':>10}  {'folded [Hz]':>12}  {'peak bin [Hz]':>14}"
          f"  {'peak [urad/rtHz]':>18}  status")
    rows = []
    for k in range(1, n_max + 1):
        f_true = k * src['f']
        f_fold = fold_frequency(f_true, f_N)
        f_bin, peak = peak_near(f_k, S_urad, f_fold, 3 * k * split_per_n)
        status = 'genuine (< f_N)' if f_true <= f_N else 'folded/aliased'
        print(f"  {k:>2}  {f_true:>10.4f}  {f_fold:>12.4f}  {f_bin:>14.6f}"
              f"  {peak:>18.4g}  {status}")
        rows.append((k, f_true, f_fold, peak, f_bin))
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
    uldm = uldm_phase_vec(t_meas, f_phi, m_phi)

    print(f"Panel (a): ULDM only ({N_cycles:,} cycles, analytic)...")
    raw_a = uldm

    specs = {}
    for src in (src_b, src_c, src_d, src_e, src_f):
        print(f"Panel ({src['name']}): {src['traj'].__name__} f0={src['f']:.4f} Hz "
              f"+ ULDM ({N_cycles:,} cycles)...")
        specs[src['name']] = compute_phases(src) + uldm

    f_k, S_a = periodogram(add_noise(raw_a, SHOT_NOISE_ASD))
    urad = {'a': np.sqrt(S_a) * 1e6}
    for key, raw in specs.items():
        _, S = periodogram(add_noise(raw, SHOT_NOISE_ASD))
        urad[key] = np.sqrt(S) * 1e6

    # the pendulum's ellipse precesses, splitting each line into sidebands
    f_prec = 0.375 * np.sqrt(g / L_PEND) * A_PEND * B_PEND / L_PEND**2 / (2 * np.pi)
    rows = {src['name']: harmonic_table(
                src['name'], src, f_k, urad[src['name']],
                split_per_n=(f_prec if src is src_f else 0.0))
            for src in (src_b, src_c, src_d, src_e, src_f)}

    f_phi_bins = {k: peak_at(f_k, u, f_phi) for k, u in urad.items()}
    print(f"\nULDM peak at f_phi = {f_phi} Hz, per panel:")
    for k in 'abcdef':
        print(f"  ({k})  {f_phi_bins[k][1]:>12.4g} urad/rtHz")
    print(f"\nShot-noise floor = {SHOT_NOISE_ASD*1e6:.4g} urad/rtHz")
    print(f"Pendulum: L = {L_PEND} m -> f0 = {f_pend:.4f} Hz "
          f"(vertical motion at 2*f0 = {2*f_pend:.4f} Hz); "
          f"precession Omega/2pi = "
          f"{0.375*np.sqrt(g/L_PEND)*A_PEND*B_PEND/L_PEND**2/(2*np.pi):.3e} Hz")

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(6, 1, figsize=(4.5, 24), sharex=True, dpi=200)
    ax_map = dict(zip('abcdef', axes))

    y_max = 3 * max(max(r[3] for r in rr[:4]) for rr in rows.values())
    print(f"\nPendulum line splitting: f0 doublet at f0 +/- {f_prec:.3e} Hz "
          f"(precessing ellipse); the 2*f0 height line is NOT split, since "
          f"X^2+Y^2 is invariant under the precession.")
    y_max = max(y_max, 3 * max(p for _, p in f_phi_bins.values()))
    y_min = 0.1 * SHOT_NOISE_ASD * 1e6

    for letter, ax in ax_map.items():
        ax.set_yscale('log')
        ax.set_ylim(y_min, y_max)
        ax.set_xlim(0, f_N)
        ax.tick_params(labelsize=7)
        ax.axvline(f_N, color='k', ls='--', lw=0.5, alpha=0.6)
        ax.set_ylabel(r'$\sqrt{S_k}$ [$\mu$rad/$\sqrt{\mathrm{Hz}}$]', fontsize=7)
        ax.text(0.99, 0.93, f'({letter})', transform=ax.transAxes, ha='right',
                va='top', fontsize=7, fontweight='bold')
    axes[-1].set_xlabel('$f$ [Hz]', fontsize=8)
    axes[0].text(f_N, y_max*0.5, r'  $f_N$', fontsize=6, ha='left', va='center')

    def mark_uldm(ax, letter, ymin, xytext=(6, 6)):
        f_bin, peak = f_phi_bins[letter]
        ax.vlines(f_bin, ymin, peak, color=COLORS['uldm'], lw=0.6, zorder=4)
        ax.scatter([f_bin], [peak], color=COLORS['uldm'], zorder=5, s=8, marker='D')
        ax.annotate(r'$f_\varphi$', (f_bin, peak), textcoords='offset points',
                    xytext=xytext, fontsize=5, color=COLORS['uldm'])

    def mark_harmonics(ax, letter, folded, offsets, ymin, n_show=4):
        """Stem + marker + label for each harmonic, in the house style. A
        harmonic landing on f_phi gets the two-tone degeneracy marker."""
        f_bin_phi, _ = f_phi_bins[letter]
        df = f_k[1] - f_k[0]
        for k, f_true, f_fold, peak, f_bin in rows[letter][:n_show]:
            col = COLORS['fundamental'] if k == 1 else (
                  COLORS['folded'] if folded else COLORS['harmonic'])
            if abs(f_fold - f_bin_phi) < 2 * df:
                continue   # coincides with f_phi -- left to the combined marker
            ax.vlines(f_bin, ymin, peak, color=col, lw=0.6, zorder=4)
            ax.scatter([f_bin], [peak], color=col, zorder=5, s=7)
            base = r'$f_0$' if k == 1 else rf'${k}f_0$'
            label = base + ' (folded)' if folded and f_true > f_N else base
            ax.annotate(label, (f_bin, peak), textcoords='offset points',
                        xytext=offsets.get(k, (4, 6)), fontsize=5, color=col)

    PANEL_OFFSETS = {
        'b': {1: (4, 6), 2: (4, 6), 3: (4, 6), 4: (5, -8)},
        'c': {1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)},
        'd': {1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)},
        'e': {1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)},
        'f': {1: (4, -2), 2: (-14, 6), 3: (0, 6), 4: (4, -8)},
    }

    def draw_panel(ax, letter, ymin):
        """All panel-specific styling (stems, markers, offsets, geometry
        annotations) in one place, so it can be reused unchanged for both the
        log-scale and linear-scale versions of this figure."""
        ax.plot(f_k, urad[letter], color=COLORS['noise'], lw=0.3, zorder=2)
        if letter == 'a':
            mark_uldm(ax, 'a', ymin)
            return
        mark_harmonics(ax, letter, folded=(letter != 'b'),
                       offsets=PANEL_OFFSETS[letter], ymin=ymin)
        mark_uldm(ax, letter, ymin, xytext=(6, -6))
        if letter == 'e':
            ax.text(0.02, 0.93, rf'incline ${INCLINE_DEG:g}^\circ$',
                    transform=ax.transAxes, ha='left', va='top', fontsize=5,
                    color=COLORS['fundamental'])
        elif letter == 'f':
            ax.text(0.02, 0.93, rf'$L={L_PEND:g}$ m $\rightarrow f_0={f_pend:.4f}$ Hz',
                    transform=ax.transAxes, ha='left', va='top', fontsize=5,
                    color=COLORS['fundamental'])

    for letter, ax in ax_map.items():
        draw_panel(ax, letter, y_min)

    fig.savefig(r'C:\Users\Georg\Desktop\fig2_multi_directional.jpg', dpi=200,
                bbox_inches='tight')

    # ── same six panels again, on LINEAR (non-log) y-axes -- since a shared
    #    linear scale would flatten the smaller panels to invisibility next to
    #    the biggest harmonics, each panel gets its own linear y-range sized to
    #    its own content (headroom = 1.15x its tallest marked peak) ───────────
    y_max_lin = {}
    for letter in 'abcdef':
        vals = [f_phi_bins[letter][1]]
        if letter != 'a':
            vals += [r[3] for r in rows[letter][:4]]
        y_max_lin[letter] = 1.15 * max(vals)

    fig_lin, axes_lin = plt.subplots(6, 1, figsize=(4.5, 24), dpi=200)
    ax_map_lin = dict(zip('abcdef', axes_lin))

    for letter, ax in ax_map_lin.items():
        ax.set_ylim(0, y_max_lin[letter])
        ax.set_xlim(0, f_N)
        ax.tick_params(labelsize=7)
        ax.axvline(f_N, color='k', ls='--', lw=0.5, alpha=0.6)
        ax.set_ylabel(r'$\sqrt{S_k}$ [$\mu$rad/$\sqrt{\mathrm{Hz}}$]', fontsize=7)
        ax.text(0.99, 0.93, f'({letter})', transform=ax.transAxes, ha='right',
                va='top', fontsize=7, fontweight='bold')
        draw_panel(ax, letter, 0.0)
    axes_lin[-1].set_xlabel('$f$ [Hz]', fontsize=8)
    axes_lin[0].text(f_N, y_max_lin['a']*0.9, r'  $f_N$', fontsize=6, ha='left', va='center')

    fig_lin.savefig(r'C:\Users\Georg\Desktop\fig2_multi_directional_linear.jpg',
                     dpi=200, bbox_inches='tight')

    # ── direct ratio comparison: the fundamental's peak height, one bar per
    #    geometry, on a shared LINEAR axis -- on the periodogram panels above
    #    (log-y, spanning 4+ decades to fit the ULDM peak and the noise floor
    #    on the same axis) a ~2-3x difference between geometries is only a
    #    fraction of a decade and barely visible; here it's the whole point ──
    letters_cmp = ['b', 'c', 'd', 'e', 'f']
    bar_labels = {
        'b': '1D linear\n(b)',
        'c': 'horiz. circle\n(c)',
        'd': 'vert. circle\n(d)',
        'e': f'incline {INCLINE_DEG:g}°\n(e)',
        'f': 'pendulum\n(f)',
    }
    peak_f0 = {letter: rows[letter][0][3] for letter in letters_cmp}   # n=1 row
    ref = peak_f0['c']   # horizontal circle as the reference direction

    print("\nFundamental peak height by geometry (same amp = "
          f"{src_common['amp']} m source), relative to horizontal circle (c):")
    for letter in letters_cmp:
        print(f"  ({letter})  {peak_f0[letter]:>12.4g} urad/rtHz   "
              f"ratio to (c) = {peak_f0[letter]/ref:.3f}")

    fig_cmp, ax_cmp = plt.subplots(figsize=(4.5, 3.2), dpi=200)
    x = np.arange(len(letters_cmp))
    bars = ax_cmp.bar(x, [peak_f0[l] for l in letters_cmp],
                       color=COLORS['fundamental'], width=0.6)
    for xi, letter in zip(x, letters_cmp):
        v = peak_f0[letter]
        ax_cmp.annotate(f'{v:,.0f}\n(×{v/ref:.2f})', (xi, v),
                         textcoords='offset points', xytext=(0, 4),
                         fontsize=6, ha='center', color=COLORS['fundamental'])
    ax_cmp.set_xticks(x)
    ax_cmp.set_xticklabels([bar_labels[l] for l in letters_cmp], fontsize=6.5)
    ax_cmp.set_ylabel(r'$\sqrt{S_k}$ at $f_0$ [$\mu$rad/$\sqrt{\mathrm{Hz}}$]', fontsize=7)
    ax_cmp.set_ylim(0, 1.25 * max(peak_f0.values()))
    ax_cmp.tick_params(labelsize=7)
    ax_cmp.set_title('Fundamental peak height by source geometry (100 m instrument)',
                      fontsize=7.5)
    fig_cmp.savefig(r'C:\Users\Georg\Desktop\fig2_multi_directional_comparison.jpg',
                     dpi=200, bbox_inches='tight')

    print("\n" + "=" * 78)
    print("Suggested caption:")
    print(f"  dt = T_cyc = {T_cyc:g} s, N_cycles = {N_cycles:,}, T_obs = {T_total:.3g} s, "
          f"f_N = {f_N:g} Hz, shot noise = {SHOT_NOISE_ASD:.0e} rad/sqrt(Hz) in all panels.")
    print(f"  ULDM: f_phi = {f_phi} Hz, m_phi = {m_phi:.4g} eV (present in every panel).")
    print(f"  Source: M = {src_common['M']}, amp = {src_common['amp']} m, "
          f"D = {src_common['D']} m, h0 = {src_common['h0']} m.")
    print(f"  (b) 1D vertical oscillator, f0 = {F_SLOW} Hz (sub-Nyquist: nonlinearity comb, no folding).")
    print(f"  (c) horizontal circle, (d) vertical circle, (e) circle inclined "
          f"{INCLINE_DEG:g} deg -- all at f0 = {F_CIRC} Hz (super-Nyquist: comb folds).")
    print(f"  (f) spherical pendulum, L = {L_PEND} m -> f0 = {f_pend:.4f} Hz, "
          f"ellipse {A_PEND} x {B_PEND} m; height runs at 2*f0.")
    print("=" * 78)

    plt.show()
