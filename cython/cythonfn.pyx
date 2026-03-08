import cython

@cython.boundscheck(False)
@cython.wraparound(False)
def velocity_update(double[:, :] vx,
                    double[:, :] vy,
                    double[:, :] rhs_x,
                    double[:, :] rhs_y,
                    double dt):
    cdef unsigned int i, j
    cdef unsigned int nx = vx.shape[0]
    cdef unsigned int ny = vx.shape[1]

    for i in range(nx):
        for j in range(ny):
            vx[i, j] += dt * rhs_x[i, j]
            vy[i, j] += dt * rhs_y[i, j]

