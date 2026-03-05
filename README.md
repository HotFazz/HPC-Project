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

![Simulation](./navier-stokes-spectral.png)
