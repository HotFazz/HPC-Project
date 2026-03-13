import cython

def velocity_update(vx, vy, rhs_x, rhs_y, dt):
    nx = vx.shape[0]
    ny = vx.shape[1]

    for i in range(nx):
        for j in range(ny):
            vx[i, j] += dt * rhs_x[i, j]
            vy[i, j] += dt * rhs_y[i, j]

