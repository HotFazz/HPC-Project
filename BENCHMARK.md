# Navier-Stokes Spectral Solver — Benchmark Results

## Single-run performance: Main vs Optimized branch

Benchmark parameters: `tEnd=0.1`, `dt=0.001`, `nu=0.001`

| Grid (N) | Main branch (original) | Optimized (numpy backend) | Speedup |
|-----------|------------------------|---------------------------|---------|
| 128x128   | 0.527s                 | 0.141s                    | 3.74x   |
| 256x256   | 2.475s                 | 0.648s                    | 3.82x   |
| 400x400   | 5.386s                 | 1.461s                    | 3.69x   |

The original solver on `main` performs ~23 FFTs per timestep (transforms back and forth for every operation: grad, div, curl, dealias, diffusion). The refactored version on `avids-optiz` stays in Fourier space and only performs ~5 FFTs per timestep (IFFT for nonlinear terms, FFT back, plus vorticity). It also pre-allocates all buffers and uses in-place NumPy operations to avoid memory allocation overhead.

## Dask parallel parameter sweep

4 cases: 2 resolutions (128, 256) x 2 viscosities (0.001, 0.005), `tEnd=0.1`

| Scheduler       | Total wall time | Speedup |
|-----------------|-----------------|---------|
| single-threaded | 29.19s          | 1.0x    |
| processes       | 2.41s           | 12.1x   |

### Per-case results (process scheduler)

| Case             | Wall time | Kinetic Energy | Enstrophy    | Max Vorticity |
|------------------|-----------|----------------|--------------|---------------|
| N=128, nu=0.001  | 0.149s    | 4.912e-01      | 4.818e+01    | 1.862e+01     |
| N=128, nu=0.005  | 0.148s    | 4.546e-01      | 4.319e+01    | 1.774e+01     |
| N=256, nu=0.001  | 0.716s    | 4.912e-01      | 4.818e+01    | 1.862e+01     |
| N=256, nu=0.005  | 0.725s    | 4.546e-01      | 4.319e+01    | 1.774e+01     |

## Optimization chain summary

1. **Main branch** — baseline (~23 FFTs/step, real-space state, temporary allocations every step)
2. **Fourier-space refactor** — ~3.7x faster (5 FFTs/step, pre-allocated buffers, in-place ops)
3. **pyFFTW backend** — additional speedup via optimized FFTW plans and multithreaded FFTs (not benchmarked here; requires `pyfftw`)
4. **Dask parameter sweep** — 12x wall-clock speedup for multi-case runs via process-level parallelism
