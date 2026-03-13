import argparse
from timeit import default_timer as timer

import cupy as cp
import matplotlib.pyplot as plt
import numpy as np
from cupyx.scipy import fft as cxfft


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


def simulate_gpu(N=400, tEnd=1.0, dt=0.001, nu=0.001):
    dtype = cp.float64
    L = 1.0

    xlin = cp.linspace(0, L, num=N + 1, dtype=dtype)[:N]
    xx, yy = cp.meshgrid(xlin, xlin, indexing="xy")

    vx = -cp.sin(2 * cp.pi * yy).astype(dtype)
    vy = cp.sin(2 * cp.pi * xx * 2).astype(dtype)

    klin = (2.0 * cp.pi / L) * cp.arange(-N // 2, N // 2, dtype=dtype)
    kmax = cp.max(klin)
    kx, ky = cp.meshgrid(klin, klin, indexing="xy")
    kx = cp.fft.ifftshift(kx)
    ky = cp.fft.ifftshift(ky)

    kSq = kx * kx + ky * ky
    kSq_inv = cp.empty_like(kSq)
    kSq_inv[:] = 1.0 / kSq
    kSq_inv[kSq == 0] = 1

    dealias = (
        (cp.abs(kx) < (2.0 / 3.0) * kmax) & (cp.abs(ky) < (2.0 / 3.0) * kmax)
    ).astype(dtype)

    Nt = int(np.ceil(tEnd / dt))
    t = 0.0

    start = cp.cuda.Event()
    end = cp.cuda.Event()

    cp.cuda.Stream.null.synchronize()
    wall_start = timer()

    start.record()

    for _ in range(Nt):
        dvx_x, dvx_y = grad(vx, kx, ky)
        dvy_x, dvy_y = grad(vy, kx, ky)

        rhs_x = -(vx * dvx_x + vy * dvx_y)
        rhs_y = -(vx * dvy_x + vy * dvy_y)

        rhs_x = apply_dealias(rhs_x, dealias)
        rhs_y = apply_dealias(rhs_y, dealias)

        vx += dt * rhs_x
        vy += dt * rhs_y

        div_rhs = div(rhs_x, rhs_y, kx, ky)
        P = poisson_solve(div_rhs, kSq_inv)
        dPx, dPy = grad(P, kx, ky)

        vx += -dt * dPx
        vy += -dt * dPy

        vx = diffusion_solve(vx, dt, nu, kSq)
        vy = diffusion_solve(vy, dt, nu, kSq)

        t += dt

    end.record()
    end.synchronize()
    cp.cuda.Stream.null.synchronize()

    wall_end = timer()

    gpu_time = cp.cuda.get_elapsed_time(start, end) / 1000.0
    wall_time = wall_end - wall_start
    return gpu_time, wall_time


def benchmark_dimensions(dimensions, runs=10, warmup=2, tEnd=1.0, dt=0.001, nu=0.001):
    results = []

    for N in dimensions:
        print(f"Benchmarking N={N}")

        for _ in range(warmup):
            simulate_gpu(N=N, tEnd=tEnd, dt=dt, nu=nu)

        gpu_times = []
        wall_times = []

        for r in range(runs):
            gpu_t, wall_t = simulate_gpu(N=N, tEnd=tEnd, dt=dt, nu=nu)
            gpu_times.append(gpu_t)
            wall_times.append(wall_t)
            print(f"  run {r+1}/{runs}: gpu={gpu_t:.4f}s wall={wall_t:.4f}s")

        gpu_times = np.array(gpu_times)
        wall_times = np.array(wall_times)

        results.append(
            {
                "N": N,
                "gpu_mean": gpu_times.mean(),
                "gpu_std": gpu_times.std(ddof=1) if runs > 1 else 0.0,
                "gpu_min": gpu_times.min(),
                "gpu_max": gpu_times.max(),
                "wall_mean": wall_times.mean(),
                "wall_std": wall_times.std(ddof=1) if runs > 1 else 0.0,
                "wall_min": wall_times.min(),
                "wall_max": wall_times.max(),
            }
        )

    return results


def plot_results(results, output="gpu_runtime_vs_dimension.png"):
    N_vals = [r["N"] for r in results]
    gpu_means = [r["gpu_mean"] for r in results]
    gpu_stds = [r["gpu_std"] for r in results]

    plt.figure(figsize=(8, 5))
    plt.errorbar(N_vals, gpu_means, yerr=gpu_stds, marker="o", capsize=4, label="CUDA event time")
    plt.xlabel("Grid dimension N")
    plt.ylabel("Runtime (s)")
    plt.title("GPU runtime as a function of grid dimension")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.show()


def print_results_table(results):
    print("\nSummary:")
    print(
        f"{'N':>6} | {'GPU mean':>10} | {'GPU std':>9} | {'GPU min':>9} | {'GPU max':>9} | "
        f"{'Wall mean':>10} | {'Wall std':>9}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['N']:6d} | "
            f"{r['gpu_mean']:10.4f} | {r['gpu_std']:9.4f} | {r['gpu_min']:9.4f} | {r['gpu_max']:9.4f} | "
            f"{r['wall_mean']:10.4f} | {r['wall_std']:9.4f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Benchmark CuPy Navier-Stokes runtime vs dimension")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--tEnd", type=float, default=1.0)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--nu", type=float, default=0.001)
    parser.add_argument(
        "--dims",
        type=int,
        nargs="+",
        default=[64, 128, 192, 256, 320, 400],
        help="Grid dimensions to benchmark",
    )
    parser.add_argument("--output", type=str, default="gpu_runtime_vs_dimension.png")
    args = parser.parse_args()

    dev = cp.cuda.Device()
    print("active device:", dev.id)
    print("device name:", cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode())

    results = benchmark_dimensions(
        dimensions=args.dims,
        runs=args.runs,
        warmup=args.warmup,
        tEnd=args.tEnd,
        dt=args.dt,
        nu=args.nu,
    )

    print_results_table(results)
    plot_results(results, output=args.output)


if __name__ == "__main__":
    main()
