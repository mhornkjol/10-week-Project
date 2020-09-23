from fenics import *
import matplotlib.pyplot as plt
from mshr import *

# Set log level to only show warnings or worse
set_log_level(30)


T = 3600*24*3 # final time
num_steps = 200  # number of time steps
dt = T / num_steps  # time step size
D = 1.2*10**(-8) # Diffusion coefficient

# Create mesh and define function space
meshSize = 128
domain = Rectangle(Point(0, 0), Point(0.1, 0.1))
aquaduct = Rectangle(Point(0.045, 0.01), Point(0.055, 0.045))
inside = Rectangle(Point(0.04, 0.045), Point(0.06, 0.055))
brain = Rectangle(Point(0.01, 0.01), Point(0.09, 0.09))
hypocampus = Rectangle(Point(0.040,0.01), Point(0.045, 0.045))
subdomain = brain - aquaduct - inside - hypocampus
domain.set_subdomain(1, subdomain)
domain.set_subdomain(2, hypocampus)
mesh = generate_mesh(domain, meshSize)
V = FunctionSpace(mesh, 'P', 1)
O = VectorFunctionSpace(mesh, 'CG', 2)

# Subdomain stuff
markers = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())

boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)
boundary_markers.set_all(9999)

class BoundaryY0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 1)

by0 = BoundaryY0()
by0.mark(boundary_markers, 0)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
dx = Measure('dx', domain=mesh, subdomain_data=markers)

# Define boundary condition
bc_bottom_wall = DirichletBC(V, Constant(1.0), "near(x[1],0)")
bcs = [bc_bottom_wall]

u = Function(O)

with XDMFFile(MPI.comm_world, "./velocity.xdmf") as xdmf:
    xdmf.read_checkpoint(u, "velocity", 0)

# Define initial value
c_n = Function(V)

# Define variational problem
c = TrialFunction(V)
v = TestFunction(V)

# Lumping
mass_form = v*c*dx
mass_action_form = action(mass_form, Constant(1))
M_lumped = assemble(mass_form)
M_lumped.zero()
M_lumped.set_diagonal(assemble(mass_action_form))

a = D*dt*dot(grad(c), grad(v))*dx + dot(u, grad(c))*v*dt*dx
A = assemble(a)
A = M_lumped + A
L = c_n*v*dx

# Define vtk file
vtkfile = File('Plot/solution.pvd')

normal = FacetNormal(mesh)

# Time-stepping
c = Function(V)
t = 0
flag = True

for n in range(num_steps):
    if(t>=10*dt and flag==True):
        bcs = []
        A = assemble(a)
        A = M_lumped + A
        flag = False

    # Update current time
    t += dt
    Total_consentration = assemble(c*dx)
    # Grad_on_Bound = assemble(D*dot(grad(c), normal)*ds(0))
    print('{}: {}'.format(n, Total_consentration))

    # Compute solution
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]
    solve(A, c.vector(), b)

    # Plot solution
    vtkfile << (c, t)

    # Update previous solution
    c_n.assign(c)