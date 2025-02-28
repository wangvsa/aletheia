import numpy
from matplotlib import pyplot, cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

nx = 41
ny = 41
nt = 100
nit = 50
c = 1
dx = 2.0 / (nx - 1)
dy = 2.0 / (ny - 1)
x = numpy.linspace(0, 2, nx)
y = numpy.linspace(0, 2, ny)
X, Y = numpy.meshgrid(x, y)

rho = 1
nu = .1
dt = .001

u = numpy.zeros((ny, nx))
v = numpy.zeros((ny, nx))
p = numpy.zeros((ny, nx)) 
b = numpy.zeros((ny, nx))

def build_up_b(b, rho, dt, u, v, dx, dy):

	b[1:-1, 1:-1] = (rho * (1 / dt * 
		((u[1:-1, 2:] - u[1:-1, 0:-2]) / 
			(2 * dx) + (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
		((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx))**2 -
		2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
			(v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx))-
		((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy))**2))

	return b

def pressure_poisson(p, dx, dy, b):
	pn = numpy.empty_like(p)
	pn = p.copy()

	for q in range(nit):
		pn = p.copy()
		p[1:-1, 1:-1] = (((pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + 
			(pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) /
		(2 * (dx**2 + dy**2)) -
		dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * 
		b[1:-1,1:-1])

	p[:, -1] = p[:, -2] ##dp/dy = 0 at x = 2
	p[0, :] = p[1, :]  ##dp/dy = 0 at y = 0
	p[:, 0] = p[:, 1]    ##dp/dx = 0 at x = 0
	p[-1, :] = 0        ##p = 0 at y = 2
	return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu):
    err_x = 0
    err_y = 0
    un = numpy.empty_like(u)
    vn = numpy.empty_like(v)
    b = numpy.zeros((ny, nx))
    for n in range(nt):
        un = u.copy()
        vn = v.copy()

        b = build_up_b(b, rho, dt, u, v, dx, dy)
        p = pressure_poisson(p, dx, dy, b)

        u[1:-1, 1:-1] = (un[1:-1, 1:-1]-
			un[1:-1, 1:-1] * dt / dx *
			(un[1:-1, 1:-1] - un[1:-1, 0:-2]) -
			vn[1:-1, 1:-1] * dt / dy *
			(un[1:-1, 1:-1] - un[0:-2, 1:-1]) -
			dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
			nu * (dt / dx**2 *
				(un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) +
				dt / dy**2 *
				(un[2:, 1:-1] - 2 * un[1:-1, 1:-1] + un[0:-2, 1:-1])))

        v[1:-1,1:-1] = (vn[1:-1, 1:-1] -
			un[1:-1, 1:-1] * dt / dx *
			(vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) -
			vn[1:-1, 1:-1] * dt / dy *
			(vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) -
			dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
			nu * (dt / dx**2 *
				(vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) +
				dt / dy**2 *
				(vn[2:, 1:-1] - 2 * vn[1:-1, 1:-1] + vn[0:-2, 1:-1])))

        u[0, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[-1, :] = 1    #set velocity on cavity lid equal to 1
        v[0, :] = 0
        v[-1, :]=0
        v[:, 0] = 0
        v[:, -1] = 0

        if n == 10: # inser error
            err_x, err_y, error = insert_error()
            p[err_x, err_y] = error
        if n  == 20: # store the frame after 10 iterations
            return numpy.stack((p,u,v), axis=2), [err_x, err_y]

def insert_error():
    x = numpy.random.randint(0, nx)
    y = numpy.random.randint(0, ny)
    error = 1
    return x, y, error

# Insert one error at each frame
def get_with_error_data():
    N = 100
    MM = numpy.zeros((N, nx, ny, 3))   # 3 channels for u, v and p
    error_positions = numpy.zeros((len(MM), 2))    # error positions for every iteration
    for i in range(N):
        MM[i], error_positions[i] = cavity_flow(nt, u, v, dt, dx, dy, p, rho, nu)

    return MM, error_positions

'''
MM, error_positions = get_with_error_data()
print MM.shape
print error_positions.shape
'''
