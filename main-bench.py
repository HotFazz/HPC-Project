from timeit import default_timer as timer

import numpy as np

"""
Create Your Own Navier-Stokes Spectral Method Simulation (With Python)
Philip Mocz (2023), @PMocz

Barebones version without any plotting or anything else

Simulate the Navier-Stokes equations (incompressible viscous fluid) 
with a Spectral method

v_t + (v.nabla) v = nu * nabla^2 v + nabla P
div(v) = 0

"""


def poisson_solve(rho, kSq_inv):
    """solve the Poisson equation, given source field rho"""
    V_hat = -(np.fft.fftn(rho)) * kSq_inv
    V = np.real(np.fft.ifftn(V_hat))
    return V


def diffusion_solve(v, dt, nu, kSq):
    """solve the diffusion equation over a timestep dt, given viscosity nu"""
    v_hat = (np.fft.fftn(v)) / (1.0 + dt * nu * kSq)
    v = np.real(np.fft.ifftn(v_hat))
    return v


def grad(v, kx, ky):
    """return gradient of v"""
    v_hat = np.fft.fftn(v)
    dvx = np.real(np.fft.ifftn(1j * kx * v_hat))
    dvy = np.real(np.fft.ifftn(1j * ky * v_hat))
    return dvx, dvy


def div(vx, vy, kx, ky):
    """return divergence of (vx,vy)"""
    dvx_x = np.real(np.fft.ifftn(1j * kx * np.fft.fftn(vx)))
    dvy_y = np.real(np.fft.ifftn(1j * ky * np.fft.fftn(vy)))
    return dvx_x + dvy_y


def curl(vx, vy, kx, ky):
    """return curl of (vx,vy)"""
    dvx_y = np.real(np.fft.ifftn(1j * ky * np.fft.fftn(vx)))
    dvy_x = np.real(np.fft.ifftn(1j * kx * np.fft.fftn(vy)))
    return dvy_x - dvx_y


def apply_dealias(f, dealias):
    """apply 2/3 rule dealias to field f"""
    f_hat = dealias * np.fft.fftn(f)
    return np.real(np.fft.ifftn(f_hat))


def simulate(N=400, tEnd=1.0, dt=0.001, nu=0.001):
    """Navier-Stokes Simulation"""

    # Simulation parameters not given in params
    # Domain [0,1] x [0,1]
    L = 1

    xlin = np.linspace(0, L, num=N + 1)  # Note: x=0 & x=1 are the same point!
    xlin = xlin[0:N]  # chop off periodic point
    xx, yy = np.meshgrid(xlin, xlin)

    # Initial Condition (vortex)
    vx = -np.sin(2 * np.pi * yy)
    vy = np.sin(2 * np.pi * xx * 2)

    # Fourier Space Variables
    klin = 2.0 * np.pi / L * np.arange(-N / 2, N / 2)
    kmax = np.max(klin)
    kx, ky = np.meshgrid(klin, klin)
    kx = np.fft.ifftshift(kx)
    ky = np.fft.ifftshift(ky)
    kSq = kx**2 + ky**2
    kSq_inv = 1.0 / kSq
    kSq_inv[kSq == 0] = 1

    # dealias with the 2/3 rule
    dealias = (np.abs(kx) < (2.0 / 3.0) * kmax) & (np.abs(ky) < (2.0 / 3.0) * kmax)

    # number of timesteps
    Nt = int(np.ceil(tEnd / dt))

    # Main Loop
    t = 0.0  # current time of the simulation
    for _ in range(Nt):
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

        # update time
        t += dt
        # print(t)

    return 0


def benchmark(runs, warmup, **sim_kwargs):
    for _ in range(warmup):
        print("Warming up")
        simulate(**sim_kwargs)

    times = np.empty(runs, dtype=np.float64)
    for i in range(runs):
        print(f"Iter {i+1}")
        t0 = timer()
        simulate(**sim_kwargs)
        t1 = timer()
        times[i] = t1 - t0

    avg = times.mean()
    mn = times.min()
    mx = times.max()
    std = times.std(ddof=1) if runs > 1 else 0.0
    return times, avg, mn, mx, std


def main():
    runs = 20
    warmup = 5
    N = 100
    tEnd = 1.0
    dt = 0.001
    nu = 0.005

    sim_kwargs = dict(N=N, tEnd=tEnd, dt=dt, nu=nu)

    times, avg, mn, mx, std = benchmark(runs, warmup, **sim_kwargs)
    print(f"runs={runs} warmup={warmup} N={N} tEnd={tEnd} dt={dt} nu={nu}")
    print(f"avg: {avg:.6f} s")
    print(f"min: {mn:.6f} s")
    print(f"max: {mx:.6f} s")
    print(f"std: {std:.6f} s")



if __name__ == "__main__":
    main()
