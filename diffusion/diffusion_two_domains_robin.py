from fenics import *
import numpy as np
import matplotlib.pyplot as plt
from mshr import *

# Set log level to only show warnings or worse
set_log_level(30)


# Diffusion
class Diffusion(UserExpression):
    def __init__(self, markers, **kwargs):
        super().__init__(**kwargs)
        self.markers = markers

    def eval_cell(self, values, x, cell):
        if(self.markers[cell.index] == 0):
            values[0] = 1.2*10**(-8)
        elif(self.markers[cell.index] == 1):
            values[0] = 1.2*10**(-10)


T = 3600*24*20  # final time
num_steps = 100  # number of time steps
dt = T / num_steps  # time step size
f = Constant(0.0)  # rhs is 0 for diffusion
u_D = 1  # Dirichlet bc
r = 0.001  # Robin bc
s = 0  # Robin bc

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

# Define boundary condition
tol = 1E-14


class BoundaryX0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0, tol)


class BoundaryX1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], 0.1, tol)


class BoundaryY0(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0, tol)


class BoundaryY1(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[1], 0.1, tol)


domain_markers = MeshFunction('size_t', mesh, mesh.topology().dim(),
                              mesh.domains())
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1)

boundary_markers.set_all(9999)

bx0 = BoundaryX0()
bx1 = BoundaryX1()
by0 = BoundaryY0()
by1 = BoundaryY1()

bx0.mark(boundary_markers, 0)
bx1.mark(boundary_markers, 1)
by0.mark(boundary_markers, 2)
by1.mark(boundary_markers, 3)

boundary_conditions = {0: {'Robin': (r, s)},
                       1: {'Robin': (r, s)},
                       2: {'Dirichlet': u_D},
                       3: {'Robin': (r, s)}}

bcs = []
for i in boundary_conditions:
    if 'Dirichlet' in boundary_conditions[i]:
        bc = DirichletBC(V, boundary_conditions[i]['Dirichlet'],
                         boundary_markers, i)
        bcs.append(bc)

ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)
dx = Measure('dx', domain=mesh, subdomain_data=domain_markers)

D = Diffusion(domain_markers, degree=0)  # Diffusion constant

# Define initial value
u_n = Function(V)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)

integrals_R = []
for i in boundary_conditions:
    if 'Robin' in boundary_conditions[i]:
        r, s = boundary_conditions[i]['Robin']
        integrals_R.append(r*(u - s)*v*ds(i))

# Lumping
mass_form = v*u*dx
mass_action_form = action(mass_form, Constant(1))
M_lumped = assemble(mass_form)
M_lumped.zero()
M_lumped.set_diagonal(assemble(mass_action_form))

a = D*dt*dot(grad(u), grad(v))*dx + sum(integrals_R)
A = assemble(a)
A = M_lumped + A
L = (u_n + dt*f)*v*dx

# Define vtk file
vtkfile = File('Diffusion two domains robin plot/solution.pvd')

# Time-stepping
u = Function(V)
t = 0

for n in range(num_steps):

    # Update current time
    t += dt

    # Compute solution
    b = assemble(L)
    [bc.apply(A, b) for bc in bcs]
    solve(A, u.vector(), b)

    # Plot solution
    vtkfile << (u, t)

    # Update previous solution
    u_n.assign(u)
