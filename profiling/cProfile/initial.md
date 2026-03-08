# Initial findings
## cProfile

While running cProfile and viewing it with snakeviz, we find that the fast fourier transforms are the main performance impactors. Essentially the entire runtime consists of FFTs, however they are difficult to improve on due to their already optimized nature. Perhaps we should remove as many calls as possible? Could they be ran on the GPU? 

It is difficult to use cython since the FFTs are already compiled. Optimizing away stuff like 
