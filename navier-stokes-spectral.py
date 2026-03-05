import matplotlib.pyplot as plt
import numpy as np

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid)
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""


def build_fourier_operators(N, L, dt, nu):
    """Precompute spectral operators used throughout the simulation."""
    klin = 2.0 * np.pi / L * np.arange(-N / 2, N / 2)
    kmax = np.max(klin)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)

    kSq = kx**2 + ky**2
    kSq_inv = np.zeros_like(kSq)
    nonzero = kSq != 0
    kSq_inv[nonzero] = 1.0 / kSq[nonzero]

    operators = {
        "kx": kx,
        "ky": ky,
        "ikx": 1j * kx,
        "iky": 1j * ky,
        "kx_kSq_inv": kx * kSq_inv,
        "ky_kSq_inv": ky * kSq_inv,
        "diffusion_factor": 1.0 / (1.0 + dt * nu * kSq),
        "dealias": (
            (np.abs(kx) < (2.0 / 3.0) * kmax)
            & (np.abs(ky) < (2.0 / 3.0) * kmax)
        ),
    }
    return operators


def project_incompressible(vx_hat, vy_hat, kx, ky, kx_kSq_inv, ky_kSq_inv):
    """Project a Fourier-space vector field onto the divergence-free subspace."""
    k_dot_v_hat = kx * vx_hat + ky * vy_hat
    vx_hat -= kx_kSq_inv * k_dot_v_hat
    vy_hat -= ky_kSq_inv * k_dot_v_hat
    return vx_hat, vy_hat


def compute_vorticity(vx_hat, vy_hat, ikx, iky, ifft2):
    """Return real-space vorticity from Fourier-space velocity."""
    return ifft2(ikx * vy_hat - iky * vx_hat).real


def run_simulation(
    N=400,
    tEnd=1,
    dt=0.001,
    tOut=0.01,
    nu=0.001,
    plotRealTime=True,
    verbose=True,
    save_figure=True,
    show_figure=True,
    figure_path="navier-stokes-spectral.png",
    return_state=False,
):
    """Navier-Stokes simulation with Fourier-space time integration."""
    t = 0.0
    L = 1.0

    xlin = np.linspace(0.0, L, num=N + 1)
    xlin = xlin[:N]
    xx, yy = np.meshgrid(xlin, xlin)

    vx0 = -np.sin(2 * np.pi * yy)
    vy0 = np.sin(4 * np.pi * xx)

    ops = build_fourier_operators(N, L, dt, nu)
    kx = ops["kx"]
    ky = ops["ky"]
    ikx = ops["ikx"]
    iky = ops["iky"]
    kx_kSq_inv = ops["kx_kSq_inv"]
    ky_kSq_inv = ops["ky_kSq_inv"]
    diffusion_factor = ops["diffusion_factor"]
    dealias = ops["dealias"]

    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2

    vx_hat = fft2(vx0)
    vy_hat = fft2(vy0)
    vx_hat, vy_hat = project_incompressible(
        vx_hat, vy_hat, kx, ky, kx_kSq_inv, ky_kSq_inv
    )

    Nt = int(np.ceil(tEnd / dt))

    fig = None
    if plotRealTime or save_figure or show_figure:
        fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1
    wz = None

    for i in range(Nt):
        vx = ifft2(vx_hat).real
        vy = ifft2(vy_hat).real
        wz = compute_vorticity(vx_hat, vy_hat, ikx, iky, ifft2)

        rhs_hat_x = dealias * fft2(vy * wz)
        rhs_hat_y = dealias * fft2(-vx * wz)
        rhs_hat_x, rhs_hat_y = project_incompressible(
            rhs_hat_x, rhs_hat_y, kx, ky, kx_kSq_inv, ky_kSq_inv
        )

        vx_hat = (vx_hat + dt * rhs_hat_x) * diffusion_factor
        vy_hat = (vy_hat + dt * rhs_hat_y) * diffusion_factor

        t += dt
        if verbose:
            print(t)

        plotThisTurn = t + dt > outputCount * tOut
        should_plot = fig is not None and ((plotRealTime and plotThisTurn) or (i == Nt - 1))
        if should_plot:
            wz = compute_vorticity(vx_hat, vy_hat, ikx, iky, ifft2)
            plt.cla()
            plt.imshow(wz, cmap="RdBu")
            plt.clim(-20, 20)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            if plotRealTime and plotThisTurn:
                plt.pause(0.001)

        if plotThisTurn:
            outputCount += 1

    if fig is not None:
        if save_figure:
            plt.savefig(figure_path, dpi=240)
        if show_figure:
            plt.show()

    if return_state:
        wz = compute_vorticity(vx_hat, vy_hat, ikx, iky, ifft2)
        return {
            "t": t,
            "vx": ifft2(vx_hat).real,
            "vy": ifft2(vy_hat).real,
            "wz": wz,
        }

    return 0


def main():
    return run_simulation()


if __name__ == "__main__":
    main()
