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
subdomain = brain - aquaduct - inside
domain.set_subdomain(1, subdomain)
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
bc_ag = DirichletBC(V, Constant(0.0), "near(x[1], 0.1) && ((x[0] > 0.01 && 0.04 > x[0]) || (x[0] > 0.065 && 0.085 > x[0]))")
bcs = [bc_bottom_wall, bc_ag]

# Load velcity from .xdmf file
u = Function(O)
with XDMFFile(MPI.comm_world, "./velocity.xdmf") as xdmf:
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
vtkfile = File('diffusion-convection ag bc plots/solution.pvd')


# Time-stepping
c = Function(V)
t = 0
flag = True

tracer_total = np.zeros(num_steps)
tracer_SAS = np.zeros(num_steps)
tracer_brain = np.zeros(num_steps)
time = np.zeros(num_steps)
for i in range(num_steps):
    # Turn of tracer injection after 10 min
    if(t>=600 and flag==True):
        bcs = [bc_ag]
        A = assemble(a2)
        A = M_lumped + A
        flag = False

    # Compute solution
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]
    solve(A, c.vector(), b)

    # Save solution
    vtkfile << (c, t)

    # Update previous solution
    c_n.assign(c)

    #total tracer amount at times vector
    tracer_total[i] = assemble(c_n*dx)
    tracer_SAS[i] = assemble(c_n*dx(0))
    tracer_brain[i] = assemble(c_n*dx(1))
    time[i] = t

    t += dt

# Extract some usefull values and save then in txt files
# Find time of 1/2 left and 1/10 left of max tracer for total, SAS and brain

tracer_domains = [tracer_total, tracer_SAS, tracer_brain]
tracer_domains_names = ["tracer_total", "tracer_SAS", "tracer_brain"]
j = 0
for tracer in tracer_domains:
    flag1 = True
    flag2 = True

    one_half_value = 0
    one_half_time = 0
    one_tenth_value = 0
    one_tenth_time = 0
    for i in range(len(time)):
        if(tracer[i]/max(tracer) < 1/2 and flag1 and time[i]>600):
            one_half_value = tracer[i]/max(tracer)
            one_half_time = time[i]/3600
            flag1 = False

        if(tracer[i]/max(tracer) < 1/10 and flag2 and time[i]>600):
            one_half_value = tracer[i]/max(tracer)
            one_half_time = time[i]/3600
            flag2 = False
        
    with open('values_ag_bc_{}.txt'.format(tracer_domains_names[j]), 'w') as f:
        f.write('One half value: {}\nOne half time: {}\n\nOne tenth value: {}\nOne tenth time: {}'.format(one_half_value, \
            one_half_time, one_tenth_value, one_tenth_time))
    
    j += 1


# Plot the tracer amount for the total domain, the SAS and the brain
# Change time units to hours and normalize so max tracer amount is 1
time = time/3600
tracer_total_scaled = tracer_total/max(tracer_total)
tracer_SAS_scaled = tracer_SAS/max(tracer_total)
tracer_brain_scaled = tracer_brain/max(tracer_total)

plt.figure()
plt.xlabel("Time [hours]", fontsize=12)
plt.ylabel('Amount of tracer [arbitrary units]', fontsize=12)
plt.plot(time, tracer_total_scaled, label="Total tracer amount")
plt.plot(time, tracer_SAS_scaled, label="Tracer amount in the SAS")
plt.plot(time, tracer_brain_scaled, label="Tracer amount in the brain")
plt.legend()
plt.savefig("../Figures/tracer_amount_ag_bc_plot.png", bbox_inches='tight')
plt.show()