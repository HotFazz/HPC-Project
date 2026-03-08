# Notes

Prior to doing any cython we know that the majority of time is spent within FFTs. We can do nothing about these, however the assignment says to optimize anyways. There are a few things we can move to a different function, and then running it with Cython, however the results will be negligable. 

This bench will use the main-bench.py as its base. The modified version is cython_prof.py

## Step 1

We look for any part of the code that is ran a lot that might be extractable into a Cython function. All the extracted functions use FFT libraries essentially alone. There are no obvious loops or VM performance thiefs. We do find a few small arithmetic operations that can be done in Cython instead, such as the delta calculations:

```python
vx += dt * rhs_x
vy += dt * rhs_y
```

These will be moved to a Cython function. This function
