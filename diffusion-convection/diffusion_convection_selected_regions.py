from fenics import *
import matplotlib.pyplot as plt
from mshr import *
import numpy as np

# Program for solving the diffusion-convection equation in a square brain with SAS
# with different diffusion constant inside the brain and SAS.
# Looking specifically at certain domains which must be written into the code
# Requires velocity from the program stokes.py to run.
# Creates folders for figures and the saving of values.
# Author: Martin Hornkjøl

# Set log level to only show warnings or worse
set_log_level(30)


#Turn on and off JCI comparing
JCI_on = False


# Define numerical and physical constants
T = 3*3600*24 # final time
num_steps = 3000 # number of time steps
dt = T / num_steps # time step size
D1 = 3.8*10**(-10) # diffusion coefficient in the SAS
D2 = 1.2*10**(-10) # diffusion coefficient in the brain


# Select domain(s) to look at. Options: tracer_hippocampus, tracer_top_SAS, tracer_white_matter, tracer_grey_matter
# For tracer_domain_names remove tracer_
#! Warning: ordering is important
#! Warning: overlapping domains might cause issues
tracer_total = np.zeros(num_steps)
tracer_hippocampus = np.zeros(num_steps)
tracer_top_SAS = np.zeros(num_steps)
tracer_white_matter = np.zeros(num_steps)
tracer_grey_matter = np.zeros(num_steps)

time = np.zeros(num_steps)

tracer_domains = [tracer_hippocampus]
tracer_domains_names = ["hippocampus"]


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

# Build vectorfunction and functionspace
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
        return near(x[1], 0.1) and ((x[0] > 0.02 and 0.0271 > x[0]) or (x[0] > 0.065 and 0.0721 > x[0]))

class DomainHC(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] > 0.04 and x[0] < 0.045) and (x[1] > 0.015 and x[1] < 0.04)

class DomainWM(SubDomain):
    def inside(self, x, on_boundary):
        return (x[0] > 0.015 and x[0] < 0.085 and x[1] > 0.015 and x[1] < 0.085) \
            and ((x[0] < 0.04 or x[0] > 0.06) or x[1] > 0.045) and ((x[0] < 0.035 or x[0] > 0.065) or (x[1] < 0.04 or x[1]> 0.06))

class DomainGM(SubDomain):
    def inside(self, x, on_boundary):
        return not ((x[0] > 0.015 and x[0] < 0.085 and x[1] > 0.015 and x[1] < 0.085) \
            and ((x[0] < 0.04 or x[0] > 0.06) or x[1] > 0.045) and ((x[0] < 0.035 or x[0] > 0.065) or (x[1] < 0.04 or x[1]> 0.06))) \
            and (x[0] > 0.01 and x[0] < 0.09 and x[1] > 0.01 and x[1] < 0.09) and ((x[0] < 0.045 or x[0] > 0.055) or x[1] > 0.045) \
            and ((x[0] < 0.04 or x[0] > 0.06) or (x[1] < 0.045 or x[1] > 0.055))

class DomainTS(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] > 0.05 and (x[0] < 0.01 or x[0] > 0.09)) or (x[1] > 0.09)

bCP = BoundaryCP()
bCP.mark(boundary_markers, 0)
bAG = BoundaryAG()
bAG.mark(boundary_markers, 1)

for i in range(len(tracer_domains)):
    if(tracer_domains_names[i]=="hippocampus"):
        HC_mark = i+2
        dHC = DomainHC()
        dHC.mark(markers, HC_mark)
    if(tracer_domains_names[i]=="top_SAS"):
        TS_mark = i+2
        dTS = DomainTS()
        dTS.mark(markers, TS_mark)
    if(tracer_domains_names[i]=="white_matter"):
        WM_mark = i+2
        dWM = DomainWM()
        dWM.mark(markers, WM_mark)
    if(tracer_domains_names[i]=="grey_matter"):
        GM_mark = i+2
        dGM = DomainGM()
        dGM.mark(markers, GM_mark)
    
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
dx = Measure('dx', domain=mesh, subdomain_data=markers)

# Define boundary condition
bc_bottom_wall = DirichletBC(V, Constant(1.0), "near(x[1],0)")
bcs = [bc_bottom_wall]

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

a1 = - dt*inner(c*u, grad(v))*dx + D1*dt*inner(grad(c), grad(v))*dx(0) + D2*dt*inner(grad(c), grad(v))*dx(1)\
     + dt*inner(c*dot(u,n), v)*ds(0) + dt*inner(c*dot(u,n), v)*ds(1)
a2 = - dt*inner(c*u, grad(v))*dx + D1*dt*inner(grad(c), grad(v))*dx(0) +D2*dt*inner(grad(c), grad(v))*dx(1)\
     + dt*inner(c*dot(u,n), v)*ds(1)
A = assemble(a1)
A = M_lumped + A
L = c_n*v*dx

# Define vtk file
vtkfile = File('diffusion-convection selected regions/solution.pvd')


# Time-stepping
c = Function(V)
t = 0
flag = True

# JCI definitions
j = 0
JCI_steps = [0,1,2,3,4]
JCI = np.zeros(len(JCI_steps))

for i in range(num_steps):
    # Turn of tracer injection after 10 min
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

    # tracer amount at different times vector
    tracer_total[i] = assemble(c_n*dx)
    for k in range(len(tracer_domains)):
        if(tracer_domains_names[k]=="hippocampus"):
            tracer_hippocampus[i] = assemble(c_n*dx(HC_mark))
        if(tracer_domains_names[k]=="top_SAS"):
            tracer_top_SAS[i] = assemble(c_n*dx(TS_mark))
        if(tracer_domains_names[k]=="white_matter"):
            tracer_white_matter[i] = assemble(c_n*dx(WM_mark))
        if(tracer_domains_names[k]=="grey_matter"):
            tracer_grey_matter[i] = assemble(c_n*dx(GM_mark))
    
    time[i] = t

    #Calculate the tracer amount at JCI times
    #! Warning: dependent on time_steps = 3000 and T=3*3600*24
    if(JCI_on):
        if(i==94 or i == 188 or i==375 or i==1500 or i==3000): # 1.504h, 3.008h, 6h, 24h, 48h
            JCI[j] = assemble(c_n*dx(2))
            j+= 1

    t += dt

# Extract some usefull values and save then in txt files
# Find time of 1/2 left and 1/10 left of max tracer for hippocampus
j = 0
for tracer in tracer_domains:
    # Flags used to only choose the first value
    flag1 = True
    flag2 = True

    # Set default values in case 1/2 or 1/10 isn't reached
    one_half_value = 0
    one_half_time = 0
    one_tenth_value = 0
    one_tenth_time = 0

    # Loop through time and choose the time and value og 1/2 and 1/10 tracer
    for i in range(1, len(time)):
        if(tracer[i]/max(tracer) < 1/2 and flag1 and tracer[i]<tracer[i-1]):
            one_half_value = tracer[i]/max(tracer)
            one_half_time = time[i]/3600
            flag1 = False

        if(tracer[i]/max(tracer) < 1/10 and flag2 and tracer[i]<tracer[i-1]):
            one_tenth_value = tracer[i]/max(tracer)
            one_tenth_time = time[i]/3600
            flag2 = False
        
    # Save data in a .txt document
    with open('Reduction values/reduction_values_{}.txt'.format(tracer_domains_names[j]), 'w') as f:
        f.write('One half value: {}\nOne half time: {}\n\nOne tenth value: {}\nOne tenth time: {}'.format(one_half_value, \
            one_half_time, one_tenth_value, one_tenth_time))
    
    j += 1


# Plot the tracer amount for the total domain and hippocampus
# Change time units to hours and normalize so max tracer amount is 1
time = time/3600

i = 0
for tracer in tracer_domains:
    # tracer_total_scaled = tracer_total/max(tracer_total)
    tracer_scaled = tracer/max(tracer_total)
    plt.figure(i)
    plt.xlabel("Time [hours]", fontsize=12)
    plt.ylabel('Amount of tracer [arbitrary units]', fontsize=12)
    # plt.plot(time, tracer_total_scaled)
    plt.plot(time, tracer_scaled)
    plt.savefig("../Figures/tracer_amount_{}_plot.png".format(tracer_domains_names[i]), bbox_inches='tight')
    plt.show()
    i += 1

if(JCI_on):
    JCI_scaled = JCI/max(tracer_total)
    plt.figure(i)
    plt.ylabel('Amount of tracer [arbitrary units]', fontsize=12)
    plt.plot(JCI_steps, JCI_scaled)
    plt.savefig("../Figures/JCI_compare_{}.png".format(tracer_domains_names[0]), bbox_inches='tight')
    plt.show()