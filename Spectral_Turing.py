import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

try:
    from tqdm import tqdm
    USE_TQDM = True
except ImportError:
    USE_TQDM = False

# ============================================================
# 0. Global parameters
# ============================================================
Du = 2e-5   # diffusion coefficient for u
Dv = 1e-5   # diffusion coefficient for v

N = 400           # N x N grid (try 512, 600, 800 for prettier patterns)
L = 1.0           # domain size
dx = L / (N - 1)

# Time step; FFT semi-implicit allows dt ~ 0.03–0.1 safely for these params
dt = 0.05

# Simulation & sampling parameters
TOTAL_STEPS  = 40000   # total PDE time steps
SAMPLE_EVERY = 100      # store every this many PDE steps
FPS          = 30      # frames per second in saved videos

N_FRAMES = TOTAL_STEPS // SAMPLE_EVERY

# ======================= Precompute Fourier space operators (periodic BCs) =======================
# Frequency grids (cycles per unit length)
kx = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)
ky = 2.0 * np.pi * np.fft.fftfreq(N, d=dx)

# Squared wave numbers |k|^2 on 2D grid
k2 = kx[:, None]**2 + ky[None, :]**2

# Denominators for semi-implicit update:
# (I - dt*D*∇²) -> (1 + dt*D*|k|^2) in Fourier space
denom_u = 1.0 + dt * Du * k2
denom_v = 1.0 + dt * Dv * k2

# ======================= FFT-based semi-implicit step =======================
def fft_step(u, v, dt, F, k):
    # Reaction terms (real space)
    uv2 = u * (v * v)
    R_u = -uv2 + F * (1.0 - u)
    R_v =  uv2 - (F + k) * v

    # Transform u, v and reaction to Fourier space
    u_hat  = np.fft.fft2(u)
    v_hat  = np.fft.fft2(v)
    Ru_hat = np.fft.fft2(R_u)
    Rv_hat = np.fft.fft2(R_v)

    # Semi-implicit update in Fourier domain:
    # (u^{n+1})^ = (u^ + dt * R_u^ ) / (1 + dt * D_u * |k|^2)
    u_new_hat = (u_hat + dt * Ru_hat) / denom_u
    v_new_hat = (v_hat + dt * Rv_hat) / denom_v

    # Back to real space
    u_new = np.fft.ifft2(u_new_hat).real
    v_new = np.fft.ifft2(v_new_hat).real

    # Optional clamping if needed:
    # u_new = np.clip(u_new, 0.0, 1.5)
    # v_new = np.clip(v_new, 0.0, 1.5)

    return u_new, v_new

# ======================= Initial conditions =======================
def init_fields(seed=0):
    """
    Create initial u, v fields:
    - u ~ 1 everywhere
    - central square with perturbed u, v
    - small random noise
    """
    rng = np.random.default_rng(seed)

    u = np.ones((N, N), dtype=float)
    v = np.zeros((N, N), dtype=float)

    r = slice(N // 2 - 10, N // 2 + 10)
    u[r, r] = 0.50
    v[r, r] = 0.25

    u += 0.01 * rng.random((N, N))
    v += 0.01 * rng.random((N, N))

    return u, v

# ======================= Simulation + snapshots =======================
def simulate_with_snapshots(F_val, k_val, total_steps, sample_every, desc="Simulating"):

    # simulate Gray–Scott for total_steps, store v every sample_every steps.
    # returns list of v-snapshots.

    n_frames = total_steps // sample_every
    print(f"{desc}: total_steps={total_steps}, sampled_frames={n_frames}")

    u, v = init_fields(seed=0)
    snapshots = []

    if USE_TQDM:
        iterator = tqdm(range(total_steps), desc=desc)
    else:
        iterator = range(total_steps)

    for n in iterator:
        u, v = fft_step(u, v, dt, F_val, k_val)

        if n % sample_every == 0:
            snapshots.append(v.copy())

    return snapshots

# ======================= Make animation for a single pattern =======================
def make_pattern_animation(name, F_val, k_val):
    desc = f"{name} (F={F_val:.3f}, k={k_val:.3f})"
    snapshots = simulate_with_snapshots(
        F_val, k_val,
        total_steps=TOTAL_STEPS,
        sample_every=SAMPLE_EVERY,
        desc=desc
    )

    fig, ax = plt.subplots()
    img = ax.imshow(
        snapshots[0],
        origin="lower",
        cmap="inferno",
        extent=[0, L, 0, L],
        interpolation="bicubic"
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(f"{name}\nF={F_val:.3f}, k={k_val:.3f}")
    fig.colorbar(img, ax=ax, label="v")

    def update_frame(i):
        img.set_data(snapshots[i])
        ax.set_title(
            f"{name}\nF={F_val:.3f}, k={k_val:.3f}, frame {i+1}/{len(snapshots)}"
        )
        return [img]

    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=len(snapshots),
        interval=1000.0 / FPS,
        blit=False
    )

    # Build a simple filename from the pattern name
    safe_name = name.lower().replace(" ", "_").replace("+", "plus").replace("/", "_")
    filename = f"gray_scott_{safe_name}.mp4"
    print(f"Saving animation to {filename} (FPS={FPS}) ...")
    ani.save(filename, fps=FPS, dpi=200)
    plt.close(fig)
    print(f"Saved {filename}.")

# ======================= Pattern gallery: one animation per pattern =======================
def main():
    pattern_params = {
        "Spots":                 (0.029, 0.057), # F, k
        "Stripes":               (0.040, 0.060),
        "Mixed (spots+stripes)": (0.022, 0.051),
        "Self-replicating":      (0.018, 0.051),
        "Chaotic / turbulent":   (0.030, 0.055),
        "Near-homogeneous":      (0.055, 0.062),
    }

    for name, (F_val, k_val) in pattern_params.items():
        make_pattern_animation(name, F_val, k_val)

        v_final = run_snapshot(F_val, k_val, n_steps=TOTAL_STEPS, seed=0)
        plt.figure(figsize=(4, 4))
        plt.imshow(
            v_final,
            origin="lower",
            cmap="magma",
            extent=[0, L, 0, L],
            interpolation="bicubic"
        )
        plt.title(f"{name}\nF={F_val:.3f}, k={k_val:.3f}")
        plt.xticks([])
        plt.yticks([])
        plt.colorbar(label="v")
        plt.tight_layout()
        plt.show()

def run_snapshot(F_val, k_val, n_steps=4000, seed=0):
    u_loc, v_loc = init_fields(seed=seed)
    for _ in range(n_steps):
        u_loc, v_loc = fft_step(u_loc, v_loc, dt, F_val, k_val)
    return v_loc

if __name__ == "__main__":
    main()
