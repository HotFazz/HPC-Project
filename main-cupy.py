import argparse
from timeit import default_timer as timer

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cupyx.scipy import fft as cxfft  # better FFT backend on GPU

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""


def poisson_solve(rho, kSq_inv):
    V_hat = -(cxfft.fftn(rho)) * kSq_inv
    return cxfft.ifftn(V_hat).real


def diffusion_solve(v, dt, nu, kSq):
    v_hat = cxfft.fftn(v) / (1.0 + dt * nu * kSq)
    return cxfft.ifftn(v_hat).real


def grad(v, kx, ky):
    v_hat = cxfft.fftn(v)
    dvx = cxfft.ifftn(1j * kx * v_hat).real
    dvy = cxfft.ifftn(1j * ky * v_hat).real
    return dvx, dvy


def div(vx, vy, kx, ky):
    dvx_x = cxfft.ifftn(1j * kx * cxfft.fftn(vx)).real
    dvy_y = cxfft.ifftn(1j * ky * cxfft.fftn(vy)).real
    return dvx_x + dvy_y


def curl(vx, vy, kx, ky):
    dvx_y = cxfft.ifftn(1j * ky * cxfft.fftn(vx)).real
    dvy_x = cxfft.ifftn(1j * kx * cxfft.fftn(vy)).real
    return dvy_x - dvx_y


def apply_dealias(f, dealias):
    f_hat = dealias * cxfft.fftn(f)
    return cxfft.ifftn(f_hat).real


def main():
    """Navier-Stokes Simulation"""

    parser = argparse.ArgumentParser(description="Navier-Stokes Spectral Simulation")
    parser.add_argument(
        "--bench",
        action="store_true",
        help="Benchmark mode (disable printing/plotting)",
    )
    args = parser.parse_args()

    benchMode = args.bench

    # Simulation parameters
    N = 400  # Spatial resolution
    t = 0  # current time of the simulation
    tEnd = 1  # time at which simulation ends
    dt = 0.001  # timestep
    tOut = 0.01  # draw frequency
    nu = 0.001  # viscosity
    plotRealTime = not benchMode

    # Domain [0,1] x [0,1]
    dtype = cp.float64
    cdtype = cp.complex64

    L = 1.0
    xlin = cp.linspace(0, L, num=N + 1, dtype=dtype)[:N]
    xx, yy = cp.meshgrid(xlin, xlin, indexing="xy")

    vx = -cp.sin(2 * cp.pi * yy).astype(dtype)
    vy = cp.sin(2 * cp.pi * xx * 2).astype(dtype)

    # Fourier Space Variables
    klin = (2.0 * cp.pi / L) * cp.arange(-N // 2, N // 2, dtype=dtype)
    kmax = cp.max(klin)
    kx, ky = cp.meshgrid(klin, klin, indexing="xy")
    kx = cp.fft.ifftshift(kx)
    ky = cp.fft.ifftshift(ky)

    kSq = kx * kx + ky * ky
    kSq_inv = cp.empty_like(kSq)
    kSq_inv[:] = 1.0 / kSq
    kSq_inv[kSq == 0] = 1

    # dealias with the 2/3 rule
    dealias = (
        (cp.abs(kx) < (2.0 / 3.0) * kmax) & (cp.abs(ky) < (2.0 / 3.0) * kmax)
    ).astype(dtype)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # prep figure
    outputCount = 1

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    cp.cuda.Stream.null.synchronize()
    t0 = timer()

    start.record()
    # Main Loop
    for i in range(Nt):
        # Advection: rhs = -(v.grad)v
        dvx_x, dvx_y = grad(vx, kx, ky)
        dvy_x, dvy_y = grad(vy, kx, ky)

        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)

        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)

        vx += dt * rhs_x
        vy += dt * rhs_y

        # Poisson solve for pressure
        div_rhs = div(rhs_x, rhs_y, kx, ky)
        P = poisson_solve(div_rhs, kSq_inv)
        dPx, dPy = grad(P, kx, ky)

        # Correction (to eliminate divergence component of velocity)
        vx += -dt * dPx
        vy += -dt * dPy

        # Diffusion solve (implicit)
        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)

        # vorticity (for plotting)
        wz = curl(vx, vy, kx, ky)

        # update time
        t += dt

        if not benchMode:
            print(t)

        # plot in real time
        plotThisTurn = False
        if t + dt > outputCount * tOut:
            plotThisTurn = True
        if plotRealTime and plotThisTurn:
            plt.cla()
            plt.imshow(wz, cmap="RdBu")
            plt.clim(-20, 20)
            ax = plt.gca()
            ax.invert_yaxis()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            ax.set_aspect("equal")
            plt.pause(0.001)
            outputCount += 1

    # Save figure
    if plotRealTime:
        plt.savefig("navier-stokes-spectral.png", dpi=240)
        plt.show()

    cp.cuda.Stream.null.synchronize()
    end.record()
    end.synchronize()
    print("elapsed (CUDA events):", cp.cuda.get_elapsed_time(start, end) / 1000, "s")
    t1 = timer()
    dev = cp.cuda.Device()
    print("active device:", dev.id)
    print("device name:", cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode())
    print("elapsed:", t1 - t0)

    return 0


if __name__ == "__main__":
    main()
