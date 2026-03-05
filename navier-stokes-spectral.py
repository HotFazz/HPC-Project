import os

import matplotlib.pyplot as plt
import numpy as np

try:
    import pyfftw
except ImportError:
    pyfftw = None

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid)
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""


class NumpyFFTBackend:
    """NumPy FFT implementation used as a baseline and fallback."""

    def __init__(self, shape):
        self.shape = shape
        self.library = "numpy"
        self.threads = 1

    def empty_complex(self):
        return np.empty(self.shape, dtype=np.complex128)

    def fft2_into(self, src, dst):
        dst[...] = np.fft.fft2(src)

    def ifft2_into(self, src, dst):
        dst[...] = np.fft.ifft2(src)


class PyFFTWBackend:
    """pyFFTW backend with aligned arrays and reusable FFT plans."""

    def __init__(self, shape, threads=None, planner_effort="FFTW_MEASURE"):
        if pyfftw is None:
            raise RuntimeError("pyFFTW backend requested but pyfftw is not installed")

        self.shape = shape
        self.library = "pyfftw"
        self.threads = threads or max(1, os.cpu_count() or 1)

        forward_src = pyfftw.empty_aligned(shape, dtype="complex128")
        forward_dst = pyfftw.empty_aligned(shape, dtype="complex128")
        backward_src = pyfftw.empty_aligned(shape, dtype="complex128")
        backward_dst = pyfftw.empty_aligned(shape, dtype="complex128")
        flags = (planner_effort,)

        self._forward = pyfftw.FFTW(
            forward_src,
            forward_dst,
            axes=(0, 1),
            direction="FFTW_FORWARD",
            flags=flags,
            threads=self.threads,
        )
        self._backward = pyfftw.FFTW(
            backward_src,
            backward_dst,
            axes=(0, 1),
            direction="FFTW_BACKWARD",
            flags=flags,
            threads=self.threads,
            normalise_idft=True,
        )

    def empty_complex(self):
        return pyfftw.empty_aligned(self.shape, dtype="complex128")

    def fft2_into(self, src, dst):
        self._forward.update_arrays(src, dst)
        self._forward()

    def ifft2_into(self, src, dst):
        self._backward.update_arrays(src, dst)
        self._backward()


def build_fft_backend(shape, backend="auto", fftw_threads=None, planner_effort="FFTW_MEASURE"):
    """Create the requested FFT backend."""
    requested = backend.lower()
    if requested not in {"auto", "numpy", "pyfftw"}:
        raise ValueError(f"Unsupported FFT backend '{backend}'")

    if requested in {"auto", "pyfftw"} and pyfftw is not None:
        return PyFFTWBackend(shape, threads=fftw_threads, planner_effort=planner_effort)
    if requested == "pyfftw":
        raise RuntimeError("pyFFTW backend requested but pyfftw is not installed")
    return NumpyFFTBackend(shape)


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


def initialize_complex_field(dst, real_values):
    """Load a real-valued field into a complex work buffer."""
    dst.real[...] = real_values
    dst.imag.fill(0.0)


def compute_vorticity(vx_hat, vy_hat, ikx, iky, fft_backend, wz_hat, scratch_hat, wz):
    """Update the real-space vorticity work buffer from Fourier-space velocity."""
    np.multiply(ikx, vy_hat, out=wz_hat)
    np.multiply(iky, vx_hat, out=scratch_hat)
    np.subtract(wz_hat, scratch_hat, out=wz_hat)
    fft_backend.ifft2_into(wz_hat, wz)
    wz.imag.fill(0.0)
    return wz


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
    fft_backend="auto",
    fftw_threads=None,
    fftw_planner_effort="FFTW_MEASURE",
):
    """Navier-Stokes simulation with Fourier-space time integration."""
    t = 0.0
    L = 1.0
    shape = (N, N)

    xlin = np.linspace(0.0, L, num=N + 1)
    xlin = xlin[:N]
    xx, yy = np.meshgrid(xlin, xlin)

    ops = build_fourier_operators(N, L, dt, nu)
    kx = ops["kx"]
    ky = ops["ky"]
    ikx = ops["ikx"]
    iky = ops["iky"]
    kx_kSq_inv = ops["kx_kSq_inv"]
    ky_kSq_inv = ops["ky_kSq_inv"]
    diffusion_factor = ops["diffusion_factor"]
    dealias = ops["dealias"]

    fft = build_fft_backend(
        shape,
        backend=fft_backend,
        fftw_threads=fftw_threads,
        planner_effort=fftw_planner_effort,
    )

    vx = fft.empty_complex()
    vy = fft.empty_complex()
    wz = fft.empty_complex()
    nonlinear_x = fft.empty_complex()
    nonlinear_y = fft.empty_complex()
    vx_hat = fft.empty_complex()
    vy_hat = fft.empty_complex()
    rhs_hat_x = fft.empty_complex()
    rhs_hat_y = fft.empty_complex()
    wz_hat = fft.empty_complex()
    scratch_hat = fft.empty_complex()

    initialize_complex_field(vx, -np.sin(2 * np.pi * yy))
    initialize_complex_field(vy, np.sin(4 * np.pi * xx))
    fft.fft2_into(vx, vx_hat)
    fft.fft2_into(vy, vy_hat)
    vx_hat, vy_hat = project_incompressible(
        vx_hat, vy_hat, kx, ky, kx_kSq_inv, ky_kSq_inv
    )

    Nt = int(np.ceil(tEnd / dt))

    fig = None
    if plotRealTime or save_figure or show_figure:
        fig = plt.figure(figsize=(4, 4), dpi=80)
    outputCount = 1

    for i in range(Nt):
        fft.ifft2_into(vx_hat, vx)
        fft.ifft2_into(vy_hat, vy)
        vx.imag.fill(0.0)
        vy.imag.fill(0.0)
        compute_vorticity(vx_hat, vy_hat, ikx, iky, fft, wz_hat, scratch_hat, wz)

        np.multiply(vy, wz, out=nonlinear_x)
        fft.fft2_into(nonlinear_x, rhs_hat_x)
        rhs_hat_x *= dealias

        np.multiply(vx, wz, out=nonlinear_y)
        np.negative(nonlinear_y, out=nonlinear_y)
        fft.fft2_into(nonlinear_y, rhs_hat_y)
        rhs_hat_y *= dealias
        rhs_hat_x, rhs_hat_y = project_incompressible(
            rhs_hat_x, rhs_hat_y, kx, ky, kx_kSq_inv, ky_kSq_inv
        )

        np.multiply(rhs_hat_x, dt, out=rhs_hat_x)
        np.multiply(rhs_hat_y, dt, out=rhs_hat_y)
        np.add(vx_hat, rhs_hat_x, out=vx_hat)
        np.add(vy_hat, rhs_hat_y, out=vy_hat)
        np.multiply(vx_hat, diffusion_factor, out=vx_hat)
        np.multiply(vy_hat, diffusion_factor, out=vy_hat)

        t += dt
        if verbose:
            print(t)

        plotThisTurn = t + dt > outputCount * tOut
        should_plot = fig is not None and ((plotRealTime and plotThisTurn) or (i == Nt - 1))
        if should_plot:
            compute_vorticity(vx_hat, vy_hat, ikx, iky, fft, wz_hat, scratch_hat, wz)
            plt.cla()
            plt.imshow(wz.real, cmap="RdBu")
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
        fft.ifft2_into(vx_hat, vx)
        fft.ifft2_into(vy_hat, vy)
        vx.imag.fill(0.0)
        vy.imag.fill(0.0)
        compute_vorticity(vx_hat, vy_hat, ikx, iky, fft, wz_hat, scratch_hat, wz)
        return {
            "t": t,
            "vx": vx.real.copy(),
            "vy": vy.real.copy(),
            "wz": wz.real.copy(),
            "fft_backend": fft.library,
            "fftw_threads": fft.threads,
        }

    return 0


def main():
    fft_backend = os.getenv("NAVIER_STOKES_FFT_BACKEND", "auto")
    fftw_threads = os.getenv("NAVIER_STOKES_FFTW_THREADS")
    if fftw_threads is not None:
        fftw_threads = int(fftw_threads)
    return run_simulation(fft_backend=fft_backend, fftw_threads=fftw_threads)


if __name__ == "__main__":
    main()
