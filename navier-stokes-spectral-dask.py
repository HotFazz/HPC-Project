import argparse
import functools
import importlib.util
import itertools
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import dask
import numpy as np
from dask import delayed

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl-cache")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

DEFAULT_SWEEP_RESOLUTIONS = (128, 256, 400, 512)
DEFAULT_SWEEP_VISCOSITIES = (0.001, 0.005, 0.01)


@dataclass(frozen=True)
class SimulationConfig:
    """Single parameter-sweep configuration for the spectral solver."""

    N: int
    nu: float
    dt: float = 0.001
    t_end: float = 1.0
    t_out: float = 0.01
    fft_backend: str = "numpy"
    fftw_threads: int | None = None

    @property
    def case_id(self):
        return f"N={self.N}, nu={self.nu:g}"


@functools.lru_cache(maxsize=1)
def load_solver_module():
    """Load the optimized single-run solver once per process."""
    module_path = Path(__file__).with_name("navier-stokes-spectral.py")
    spec = importlib.util.spec_from_file_location("navier_stokes_spectral_base", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load solver module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Dask workers never need interactive plotting.
    module.plt.pause = lambda *args, **kwargs: None
    module.plt.show = lambda *args, **kwargs: None
    module.plt.savefig = lambda *args, **kwargs: None
    return module


def summarize_state(state):
    """Compress the final flow field into small diagnostics for Dask transport."""
    vx = state["vx"]
    vy = state["vy"]
    wz = state["wz"]
    return {
        "kinetic_energy": float(0.5 * np.mean(vx * vx + vy * vy)),
        "enstrophy": float(0.5 * np.mean(wz * wz)),
        "max_vorticity": float(np.max(np.abs(wz))),
    }


def run_sweep_case(config):
    """Execute one simulation case and return lightweight diagnostics."""
    solver = load_solver_module()
    start = time.perf_counter()
    state = solver.run_simulation(
        N=config.N,
        tEnd=config.t_end,
        dt=config.dt,
        tOut=config.t_out,
        nu=config.nu,
        plotRealTime=False,
        verbose=False,
        save_figure=False,
        show_figure=False,
        return_state=True,
        fft_backend=config.fft_backend,
        fftw_threads=config.fftw_threads,
    )
    wall_time = time.perf_counter() - start

    result = asdict(config)
    result.update(
        case_id=config.case_id,
        wall_time=wall_time,
        fft_backend=state["fft_backend"],
        fftw_threads=state["fftw_threads"],
    )
    result.update(summarize_state(state))
    return result


def resolve_fftw_threads(fft_backend, fftw_threads, scheduler):
    """Avoid nested parallelism when Dask is already using worker processes."""
    if fft_backend not in {"pyfftw", "auto"}:
        return fftw_threads
    if fftw_threads is not None:
        return fftw_threads
    if scheduler == "processes":
        return 1
    return max(1, os.cpu_count() or 1)


def build_sweep_configs(
    resolutions,
    viscosities,
    dt,
    t_end,
    t_out,
    fft_backend,
    fftw_threads,
    scheduler,
):
    """Generate the parameter grid for the Dask sweep."""
    resolved_threads = resolve_fftw_threads(fft_backend, fftw_threads, scheduler)
    return [
        SimulationConfig(
            N=N,
            nu=nu,
            dt=dt,
            t_end=t_end,
            t_out=t_out,
            fft_backend=fft_backend,
            fftw_threads=resolved_threads,
        )
        for N, nu in itertools.product(resolutions, viscosities)
    ]


def run_parameter_sweep(configs, scheduler="processes", num_workers=None):
    """Run the sweep with Dask delayed tasks."""
    jobs = [delayed(run_sweep_case, pure=False)(config) for config in configs]
    compute_kwargs = {"scheduler": scheduler}
    if num_workers is not None:
        compute_kwargs["num_workers"] = num_workers
    return list(dask.compute(*jobs, **compute_kwargs))


def benchmark_parameter_sweep(configs, schedulers, num_workers=None):
    """Benchmark multiple Dask schedulers on the same sweep."""
    benchmarks = []
    for scheduler in schedulers:
        start = time.perf_counter()
        results = run_parameter_sweep(
            configs,
            scheduler=scheduler,
            num_workers=None if scheduler == "single-threaded" else num_workers,
        )
        total_wall_time = time.perf_counter() - start
        benchmarks.append(
            {
                "scheduler": scheduler,
                "total_wall_time": total_wall_time,
                "results": results,
            }
        )
    return benchmarks


def print_result_table(results):
    """Pretty-print the per-case diagnostics."""
    print(
        "case_id               wall_time   backend  fftw_threads  kinetic_energy  enstrophy  max_vorticity"
    )
    for result in results:
        print(
            f"{result['case_id']:<20} "
            f"{result['wall_time']:>9.3f} "
            f"{result['fft_backend']:<8} "
            f"{str(result['fftw_threads']):>12} "
            f"{result['kinetic_energy']:>15.6e} "
            f"{result['enstrophy']:>10.6e} "
            f"{result['max_vorticity']:>13.6e}"
        )


def print_benchmark_summary(benchmarks):
    """Print aggregate sweep timings and speedups."""
    print("scheduler         total_wall_time   speedup_vs_first")
    baseline = benchmarks[0]["total_wall_time"]
    for benchmark in benchmarks:
        speedup = baseline / benchmark["total_wall_time"]
        print(
            f"{benchmark['scheduler']:<17} "
            f"{benchmark['total_wall_time']:>15.3f} "
            f"{speedup:>17.3f}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Dask-based parameter sweeps for the spectral Navier-Stokes solver."
    )
    parser.add_argument(
        "--resolutions",
        type=int,
        nargs="+",
        default=list(DEFAULT_SWEEP_RESOLUTIONS),
        help="Spatial resolutions to sweep.",
    )
    parser.add_argument(
        "--viscosities",
        type=float,
        nargs="+",
        default=list(DEFAULT_SWEEP_VISCOSITIES),
        help="Viscosity values to sweep.",
    )
    parser.add_argument("--dt", type=float, default=0.001, help="Timestep for every case.")
    parser.add_argument(
        "--t-end",
        type=float,
        default=1.0,
        help="End time for every case.",
    )
    parser.add_argument(
        "--t-out",
        type=float,
        default=0.01,
        help="Output cadence passed to the underlying solver.",
    )
    parser.add_argument(
        "--scheduler",
        choices=("single-threaded", "threads", "processes"),
        default="processes",
        help="Dask scheduler to use for the main sweep.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers for threaded or process-based execution.",
    )
    parser.add_argument(
        "--fft-backend",
        choices=("numpy", "pyfftw", "auto"),
        default="numpy",
        help="FFT backend used inside each simulation worker.",
    )
    parser.add_argument(
        "--fftw-threads",
        type=int,
        default=None,
        help="Threads per simulation when using the pyFFTW backend.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark single-threaded execution against the selected scheduler.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of tables.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the JSON payload.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    configs = build_sweep_configs(
        resolutions=args.resolutions,
        viscosities=args.viscosities,
        dt=args.dt,
        t_end=args.t_end,
        t_out=args.t_out,
        fft_backend=args.fft_backend,
        fftw_threads=args.fftw_threads,
        scheduler=args.scheduler,
    )

    if args.benchmark:
        schedulers = ["single-threaded"]
        if args.scheduler != "single-threaded":
            schedulers.append(args.scheduler)
        payload = {
            "configs": [asdict(config) for config in configs],
            "benchmarks": benchmark_parameter_sweep(
                configs,
                schedulers=schedulers,
                num_workers=args.workers,
            ),
        }
    else:
        payload = {
            "configs": [asdict(config) for config in configs],
            "scheduler": args.scheduler,
            "workers": args.workers,
            "results": run_parameter_sweep(
                configs,
                scheduler=args.scheduler,
                num_workers=args.workers,
            ),
        }

    if args.output is not None:
        Path(args.output).write_text(json.dumps(payload, indent=2))

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    if args.benchmark:
        print_benchmark_summary(payload["benchmarks"])
        print()
        print_result_table(payload["benchmarks"][-1]["results"])
    else:
        print_result_table(payload["results"])

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
