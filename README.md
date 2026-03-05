# navier-stokes-spectral-python
Spectral Solver for Navier-Stokes Equations

## Create Your Own Navier-Stokes Spectral Method Simulation (With Python)

### Philip Mocz (2023) [@PMocz](https://twitter.com/PMocz)

### [📝 Read the Algorithm Write-up on Medium](https://philip-mocz.medium.com/create-your-own-navier-stokes-spectral-method-fluid-simulation-with-python-3f37405524f4)

Simulate the Navier-Stokes equations with a Spectral method


```
python navier-stokes-spectral.py
```

Optional `pyfftw` backend:

```
python3 -m pip install --user pyfftw
NAVIER_STOKES_FFT_BACKEND=numpy python3 navier-stokes-spectral.py
NAVIER_STOKES_FFT_BACKEND=pyfftw NAVIER_STOKES_FFTW_THREADS=10 python3 navier-stokes-spectral.py
```

Dask parameter sweep variant:

```
python3 navier-stokes-spectral-dask.py --resolutions 256 320 --viscosities 0.001 0.003 0.005 0.01 --t-end 0.2 --scheduler processes --workers 4 --benchmark
```

This variant parallelizes independent simulation cases with `dask.delayed`, which is a better fit for this solver than chunking the FFT-heavy inner loop.

![Simulation](./navier-stokes-spectral.png)
