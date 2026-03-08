# Results

The cython results are minimal, however it seems that the move of the velocity function to cython did have a marginal impact. These are the benchmark results using the Cython optimization versus the default benchmark:

## Cython
```
runs=10 warmup=0 N=100 tEnd=1.0 dt=0.001 nu=0.001
avg: 1.360691 s
min: 1.358191 s
max: 1.369117 s
std: 0.003277 s
```

```
runs=20 warmup=5 N=100 tEnd=1.0 dt=0.001 nu=0.001
avg: 1.356055 s
min: 1.354098 s
max: 1.366008 s
std: 0.002588 s
```

```
runs=20 warmup=5 N=100 tEnd=1.0 dt=0.001 nu=0.005
avg: 1.361064 s
min: 1.359160 s
max: 1.363194 s
std: 0.001073 s
```

## Default
```
runs=10 warmup=0 N=100 tEnd=1.0 dt=0.001 nu=0.001
avg: 1.365565 s
min: 1.361838 s
max: 1.375791 s
std: 0.003968 s
```

```
runs=20 warmup=5 N=100 tEnd=1.0 dt=0.001 nu=0.001
avg: 1.368603 s
min: 1.362190 s
max: 1.437424 s
std: 0.016306 s
```

```
runs=20 warmup=5 N=100 tEnd=1.0 dt=0.001 nu=0.005
avg: 1.375095 s
min: 1.367315 s
max: 1.391676 s
std: 0.005592 s
```

## Notes

We see that the avg runtime was about 1/100th of a second lower for each run relative the baseline. 
