from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from mshr import *

# program for solving the diffusion equation on a square brain with SAS
# with different diffusion coefficient in the brain and SAS
# Creates folders for figures
# Author: Martin Hornkjøl

# Set log level to only show warnings or worse
set_log_level(30)


T = 30*3600*24  # final time
num_steps = 3000  # number of time steps
dt = T / num_steps  # time step size
D1 = 3.8*10**(-10) # diffusion coefficient in the SAS
D2 = 1.2*10**(-10) # diffusion coefficient in the brain

# Create mesh and define function space
meshSize = 64
CP = Rectangle(Point(0.04, 0.054), Point(0.06, 0.056))
domain = Rectangle(Point(0, 0), Point(0.1, 0.1)) - CP
aquaduct = Rectangle(Point(0.045, 0.01), Point(0.055, 0.045))
inside = Rectangle(Point(0.04, 0.045), Point(0.06, 0.056))
brain = Rectangle(Point(0.01, 0.01), Point(0.09, 0.09))
subdomain = brain - aquaduct - inside
domain.set_subdomain(1, subdomain)
mesh = generate_mesh(domain, meshSize)
V = FunctionSpace(mesh, 'P', 1)

# Define domains
markers = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())
dx = Measure('dx', domain=mesh, subdomain_data=markers)

# Define boundary condition
bc_bottom_wall = DirichletBC(V, Constant(1), 'near(x[1], 0)')
bc_ag = DirichletBC(V, Constant(0.0), "near(x[1], 0.1) && ((x[0] > 0.02 && 0.0271 > x[0]) || (x[0] > 0.065 && 0.0721 > x[0]))")

bcc = [bc_bottom_wall, bc_ag]

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

# Diffusion equation to be solved
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

# Tracer amount vectors and time vector
tracer_total = np.zeros(num_steps)
tracer_SAS = np.zeros(num_steps)
tracer_brain = np.zeros(num_steps)
time = np.zeros(num_steps)

# Solve the diffusion equation in timesteps
for i in range(num_steps):  
    # Turn of concentration BC after 10 timesteps
    if (t>=600 and flag):
        bcc = [bc_ag]
        A = assemble(a)
        A = M_lumped + A
        flag = False

    # Compute solution
    b = assemble(L)
    [bc.apply(A, b) for bc in bcc]
    solve(A, u.vector(), b)

    # Save solution
    vtkfile << (u, t)

    #total tracer amount at times vector
    tracer_total[i] = assemble(u_*dx)
    tracer_SAS[i] = assemble(u_*dx(0))
    tracer_brain[i] = assemble(u_*dx(1))
    time[i] = t

    # Update previous solution
    u_.assign(u)

    # Update current time
    t += dt

# Change the time to hours
# Scale the tracer amount to a max value of 1
time = time/3600
tracer_total_scaled = tracer_total/max(tracer_total)
tracer_SAS_scaled = tracer_SAS/max(tracer_total)
tracer_brain_scaled = tracer_brain/max(tracer_total)

# Plot the tracer amount over time in total, the brain and the SAS
plt.figure()
plt.xlabel("Time [hours]", fontsize=12)
plt.ylabel('Amount of tracer [arbitrary units]', fontsize=12)
plt.plot(time, tracer_total_scaled, label="Total tracer amount")
plt.plot(time, tracer_SAS_scaled, label="Tracer amount in the SAS")
plt.plot(time, tracer_brain_scaled, label="Tracer amount in the brain")
plt.legend()
plt.savefig("../Figures/tracer_amount_diffusion_plot.png", bbox_inches='tight')
plt.show()