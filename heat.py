#solotion of heat equation by Crank-Nicolson method in Similutaion
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.sparse import diags, kron, identity
from scipy.sparse import diags, kron, identity
from scipy.sparse.linalg import spsolve

Lx, Ly=1.0, 1.0          #It is the size of the metal conducter eg(1m*1m)
Nx, Ny=20, 20            #frames no in x&y axis
dx, dy = Lx/Nx, Ly/Ny    #gap between the frames
alpha= 0.01              #it is the constant of the conducter, proportional to the rate of flowing speed
dt= 0.001                #time interval
T= 0.1                   #similutaion time
Nt = int(T/dt)           #no of steps!

x=np.linspace(0,Lx,Nx+1)
y=np.linspace(0,Ly,Ny+1)
X,Y = np.meshgrid(x,y,indexing='ij') # this makes the visual plate in form of arrays

u= np.zeros((Nx+1,Ny+1))
u[Nx//2,Ny//2] =100  # we considering the heat is spread from the center since at time=0 the  temperature of thee middle is 100 dgr celcius

def flatten(u):
    return u[1:-1,1:-1].flatten()
def reshape(u_vec):
    return u_vec.reshape((Nx-1,Ny-1)) #reshaping the plate

rx = alpha * dt / (2 * dx**2)
ry = alpha * dt / (2 * dy**2) # this is the constant crank Nicolson Formula

Ix = identity(Nx-1)
Iy = identity(Ny-1)

ex = np.ones(Nx-1)
ey = np.ones(Ny-1)

Tx = diags([ex, -2*ex, ex], [-1, 0, 1], shape=(Nx-1, Nx-1))
Ty = diags([ey, -2*ey, ey], [-1, 0, 1], shape=(Ny-1, Ny-1)) #2nd ordeer derivative

L = kron(Iy, Tx) + kron(Ty, Ix)    #2d laplacian i.e using matrix multiplication

A = identity((Nx-1)*(Ny-1)) - rx * L
B = identity((Nx-1)*(Ny-1)) + rx * L

u_interior = flatten(u)

fig, ax = plt.subplots()
im = ax.imshow(u, cmap='hot', origin='lower', extent=[0, Lx, 0, Ly], vmin=0, vmax=100)
cbar = plt.colorbar(im)
ax.set_title("2D Heat Equation: t = 0.00")
ax.set_xlabel('x')
ax.set_ylabel('y')

def update(frame):
    global u_interior
    for _ in range(5):
        rhs = B @ u_interior
        u_interior = spsolve(A, rhs)
        u[1:-1, 1:-1] = reshape(u_interior)
    
    ax.clear()
    cp = ax.contourf(X, Y, u, cmap='hot', levels=50)
    ax.set_title(f"2D Heat Equation: t = {frame*5*dt:.3f} s")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    return cp.collections

anim = FuncAnimation(fig, update, frames=Nt//5, interval=50, blit=False)
plt.show()
anim.save("heat2d.mp4", fps=20, dpi=150)