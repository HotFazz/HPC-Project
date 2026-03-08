import cython

def velocity_update(vx, vy, rhs_x, rhs_y, dt):
    vx += dt * rhs_x
    vy += dt * rhs_y

