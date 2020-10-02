from fenics import *
import matplotlib.pyplot as plt
from mshr import *
import numpy as np

# Set log level to only show warnings or worse
set_log_level(30)


T = 3*3600*24 # final time
num_steps = 3000 # number of time steps
dt = T / num_steps # time step size
D = 1.2*10**(-8) # Diffusion coefficient

# Create mesh and define function space
meshSize = 64
domain = Rectangle(Point(0, 0), Point(0.1, 0.1))
aquaduct = Rectangle(Point(0.045, 0.01), Point(0.055, 0.045))
inside = Rectangle(Point(0.04, 0.045), Point(0.06, 0.055))
brain = Rectangle(Point(0.01, 0.01), Point(0.09, 0.09))
hippocampus = Rectangle(Point(0.04, 0.015), Point(0.045, 0.04))
subdomain = brain - aquaduct - inside - hippocampus
domain.set_subdomain(1, subdomain)
domain.set_subdomain(2, hippocampus)
mesh = generate_mesh(domain, meshSize)
V = FunctionSpace(mesh, 'P', 1)
O = VectorFunctionSpace(mesh, 'CG', 2)

# Define subdomains and boundaries
markers = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())

boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundary_markers.set_all(9999)

class BoundaryCP(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],0.055) and ((0.04 < x[0] and x[0] < 0.06))

class BoundaryAG(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1], 0.1) and ((x[0] > 0.01 and 0.04 > x[0]) or (x[0] > 0.065 and 0.085 > x[0]))


bCP = BoundaryCP()
bCP.mark(boundary_markers, 0)
bAG = BoundaryAG()
bAG.mark(boundary_markers, 1)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
dx = Measure('dx', domain=mesh, subdomain_data=markers)

# Define boundary condition
bc_bottom_wall = DirichletBC(V, Constant(1.0), "near(x[1],0)")
bcs = [bc_bottom_wall]

# Load velcity from .xdmf file
u = Function(O)
with XDMFFile(MPI.comm_world, "./velocity_with_hippocampus.xdmf") as xdmf:
    xdmf.read_checkpoint(u, "velocity", 0)

# Define initial value
c_n = Function(V)

# Define variational problem
c = TrialFunction(V)
v = TestFunction(V)

# Normal vector
n = FacetNormal(mesh)

# Lumping
mass_form = v*c*dx
mass_action_form = action(mass_form, Constant(1))
M_lumped = assemble(mass_form)
M_lumped.zero()
M_lumped.set_diagonal(assemble(mass_action_form))

a1 = - dt*inner(c*u, grad(v))*dx + D*dt*inner(grad(c), grad(v))*dx + dt*inner(c*dot(u,n), v)*ds(0) + dt*inner(c*dot(u,n), v)*ds(1)
a2 = - dt*inner(c*u, grad(v))*dx + D*dt*inner(grad(c), grad(v))*dx + dt*inner(c*dot(u,n), v)*ds(1)
A = assemble(a1)
A = M_lumped + A
L = c_n*v*dx

# Define vtk file
vtkfile = File('diffusion-convection with hippocampus plots/solution.pvd')


# Time-stepping
c = Function(V)
t = 0
flag = True

y = np.zeros(num_steps)
x = np.zeros(num_steps)
for i in range(num_steps):
    if(t>=600 and flag==True):
        bcs = []
        A = assemble(a2)
        A = M_lumped + A
        flag = False

    # Compute solution
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]
    solve(A, c.vector(), b)

    # Plot solution
    vtkfile << (c, t)

    # Update previous solution
    c_n.assign(c)

    #Concentration vector
    y[i] = assemble(c_n*dx(2))
    x[i] = t

    t += dt

# Extract some usefull values
# Find start and end of flat section
# Find time of 1/2 left and 1/10 left of max tracer
flag1 = True
flag2 = True
flag3 = True
flag4 = True
max_tracer = max(y)
flat_start_value = 0
flat_start_time = 0
flat_end_value = 0
flat_end_time = 0
one_half_value = 0
one_half_time = 0
one_tenth_value = 0
one_tenth_time = 0
for i in range(len(x)):
    if((y[i]-y[i-1]) < DOLFIN_EPS and flag1):
        flat_start_value = y[i-1]
        flat_start_time = x[i-1]
        flag1 = False

    if((y[i]+DOLFIN_EPS)<y[i-1] and flag2):
        flat_end_value = y[i-1]
        flat_end_time = x[i-1]
        flag2 = False
    
    if(y[i]/max_tracer < 1/2 and flag3):
        one_half_value = y[i]
        one_half_time = x[i]
        flag3 = False

    if(y[i]/max_tracer < 1/10 and flag4):
        one_tenth_value = y[i]
        one_tenth_time = x[i]
        flag4 = False
    
# Save the usefull values
with open('values_with_hippocampus.txt', 'w') as f:
    f.write('Flat start value: {}\nFlat start time: {}\n\nFlat end value: {}\nFlat end time: {}\n\nOne half value: {}\nOne half time: {}\n\nOne tenth value: {}\nOne tenth time: {}'.format(flat_start_value, \
        flat_start_time,flat_end_value,flat_end_time, \
        one_half_value, one_half_time, one_tenth_value, one_tenth_time))



plt.figure()
plt.xlabel("Time [s]", fontsize=12)
plt.ylabel('Amount of tracer [arbitrary units]', fontsize=12)
plt.plot(x,y)
plt.savefig("../Figures/tracer_amount_in_hippocapmus_plot.png", bbox_inches='tight')
plt.show()