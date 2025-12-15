import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from IPython.display import HTML

# --- 1. SETTINGS & PARAMETERS ---
# I renamed these to match what the function expects below
# WORM PARAMETERS
F_val = 0.086
k_val = 0.059

def get_diffusion(grid):
    return (
        np.roll(grid, 1, axis=0) + np.roll(grid, -1, axis=0) +
        np.roll(grid, 1, axis=1) + np.roll(grid, -1, axis=1) -
        4 * grid
    )

def compute_lighting(grid):
    # 1. Gradient
    dy, dx = np.gradient(grid * 60)

    normal_x = -dx
    normal_y = -dy
    normal_z = np.ones_like(grid)

    len_vec = np.sqrt(normal_x**2 + normal_y**2 + normal_z**2)
    nx = normal_x / len_vec
    ny = normal_y / len_vec
    nz = normal_z / len_vec

    # Light direction
    lx, ly, lz = 0.5, 0.5, 1.0
    l_len = np.sqrt(lx**2 + ly**2 + lz**2)
    lx, ly, lz = lx/l_len, ly/l_len, lz/l_len

    diffuse = (nx * lx + ny * ly + nz * lz)

    # 2. Specular (Metallic Shine)
    specular = np.power(diffuse, 150) * 0.9

    shading = 0.1 + (0.4 * diffuse) + specular

    return np.clip(shading, 0, 1)

def simulate_pattern(N, steps, F, k):
    Du, Dv = 0.16, 0.08
    dt = 1.0

    u = np.ones((N, N))
    v = np.zeros((N, N))

    mid = N // 2
    r = 30

    noise = np.random.uniform(0, 0.05, (2*r, 2*r))
    u[mid-r:mid+r, mid-r:mid+r] = 0.50 + noise
    v[mid-r:mid+r, mid-r:mid+r] = 0.25 + noise

    for i in range(steps):
        Lu = get_diffusion(u)
        Lv = get_diffusion(v)
        reaction = u * v * v
        u += dt * (Du * Lu - reaction + F * (1 - u))
        v += dt * (Dv * Lv + reaction - (F + k) * v)
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        yield v

def run_animation():
    # --- RESOLUTION UPGRADE ---
    N = 350
    # Passing the correct F_val and k_val variables here
    sim = simulate_pattern(N, steps=50000, F=F_val, k=k_val)

    fig, ax = plt.subplots(figsize=(8, 8), dpi=150)

    fig.patch.set_facecolor('#111111')
    ax.set_axis_off()

    v_data = next(sim)
    shaded_data = compute_lighting(v_data)

    im = ax.imshow(shaded_data, cmap='gray', interpolation='bicubic')

    # --- HUD DISPLAY ---
    ax.text(
        0.02, 0.98,
        f"F={F_val:.4f}\nk={k_val:.4f}",
        transform=ax.transAxes,
        color='white',
        fontsize=12,
        fontfamily='monospace',
        verticalalignment='top',
        bbox=dict(facecolor='black', alpha=0.5, edgecolor='none')
    )

    def update(frame):
        for _ in range(40):
            try:
                v_data = next(sim)
            except StopIteration:
                break

        shaded_frame = compute_lighting(v_data)
        im.set_array(shaded_frame)
        return [im]

    ani = FuncAnimation(fig, update, frames=300, interval=30, blit=True)

    # --- DOWNLOAD FUNCTIONALITY ---
    print("Rendering video... (This might take a minute)")
    ani.save('reaction_diffusion_worms.mp4', writer='ffmpeg', fps=30)
    print("Video saved as 'reaction_diffusion_worms.mp4'")

    plt.close()
    return HTML(ani.to_jshtml())

run_animation()
