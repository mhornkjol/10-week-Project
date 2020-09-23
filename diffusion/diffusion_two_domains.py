from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from mshr import *

# Set log level to only show warnings or worse
set_log_level(30)


T = 3*3600*24  # final time
num_steps = 100  # number of time steps
dt = T / num_steps  # time step size
D1 = 1.2*10**(-8) # diffusion coefficient in the SAS
D2 = 1.2*10**(-10) # diffusion coefficient in the brain

# Create mesh and define function space
meshSize = 64
domain = Rectangle(Point(0, 0), Point(0.1, 0.1))
aquaduct = Rectangle(Point(0.045, 0.01), Point(0.055, 0.045))
inside = Rectangle(Point(0.04, 0.045), Point(0.06, 0.055))
brain = Rectangle(Point(0.01, 0.01), Point(0.09, 0.09))
subdomain = brain - aquaduct - inside
domain.set_subdomain(1, subdomain)
mesh = generate_mesh(domain, meshSize)
V = FunctionSpace(mesh, 'P', 1)

# Subdomain stuff
markers = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())
dx = Measure('dx', domain=mesh, subdomain_data=markers)

# Define boundary condition
bc_bottom_wall = DirichletBC(V, Constant(1), 'near(x[1], 0)')
bcc = [bc_bottom_wall]

# Define initial value
u_ = Function(V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

# Lumping
mass_form = v*u*dx
mass_action_form = action(mass_form, Constant(1))
M_lumped = assemble(mass_form)
M_lumped.zero()
M_lumped.set_diagonal(assemble(mass_action_form))   

a = D1*dt*dot(grad(u), grad(v))*dx(0)+D2*dt*dot(grad(u), grad(v))*dx(1)
A = assemble(a)
A = M_lumped + A
L = u_*v*dx

# Define vtk file
vtkfile = File('Diffusion two domains plot/solution.pvd')

# Time-stepping
u = Function(V)
t = 0
flag = True

for n in range(num_steps):  
    # Turn of concentration BC after 10 timesteps
    if (t>=10*dt):
        bcc = []
        A = assemble(a)
        A = M_lumped + A
        flag = False
    # Update current time
    t += dt

    # Compute solution
    b = assemble(L)
    [bc.apply(A, b) for bc in bcc]
    solve(A, u.vector(), b)

    vtkfile << (u, t)

    # Update previous solution
    u_.assign(u)
