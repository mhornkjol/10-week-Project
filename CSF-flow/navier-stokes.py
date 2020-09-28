from dolfin import *
from fenics import *
from mshr import *

# Constants
T = 5 # 3600*24 # final time
num_steps = 500  # number of time steps
dt = T / num_steps  # time step size
rho = 1000

# Load mesh
meshSize = 64
domain = Rectangle(Point(0, 0), Point(0.1, 0.1))
aquaduct = Rectangle(Point(0.045, 0.01), Point(0.055, 0.045))
inside = Rectangle(Point(0.04, 0.045), Point(0.06, 0.055))
brain = Rectangle(Point(0.01, 0.01), Point(0.09, 0.09))
subdomain = brain - aquaduct - inside
domain.set_subdomain(1, subdomain)
mesh = generate_mesh(domain, meshSize)

# Build function space
V = VectorFunctionSpace(mesh, "Lagrange", 2)
Q = FunctionSpace(mesh, "Lagrange", 1)

# Viscosity inside and outside brain
mu1 = 8.9*10**(-4)
mu2 = 1000*mu1

# No-slip boundary condition for velocity on sides and bottom
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(V, noslip, "near(x[0], 0.1) || near(x[0], 0) || near(x[1], 0)")

# Top boundary condition excluding AG
bc1 = DirichletBC(V, noslip, "near(x[1], 0.1) && (x[0] < 0.01 || (0.03 < x[0] && x[0] < 0.04) || (0.06 < x[0] && x[0] < 0.07) || 0.09 < x[0])")

# Inflow from ventricle
inflow = Constant((0.0, -0.000003)) # 3 micrometer/s corresponding to 0.5 L/day flow
bc2 = DirichletBC(V, inflow, "near(x[1], 0.055) && (0.045 < x[0] && x[0] < 0.055)")

# Collect boundary conditions
bcu = [bc0, bc1, bc2]
bcp = []

domain_markers = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())

dx = Measure('dx', domain=mesh, subdomain_data=domain_markers)

# Define variational problem
u = TrialFunction(V)
p = TrialFunction(Q)
v = TestFunction(V)
q = TestFunction(Q)


# Create functions
u_ = Function(V)
u1 = Function(V)
p1 = Function(Q)

# Tentative velocity step
F1 = rho*inner(u - u_, v)*(1/dt)*dx + rho*inner(grad(u_)*u_, v)*dx + mu1*inner(grad(u), grad(v))*dx(0) + mu2*inner(grad(u), grad(v))*dx(1)
a1 = lhs(F1)
L1 = rhs(F1)

# Pressure update
a2 = inner(grad(p), grad(q))*dx
L2 = -(1/dt)*div(u1)*q*dx

# Velocity update
a3 = inner(u, v)*dx
L3 = inner(u1, v)*dx - dt/rho*inner(grad(p1), v)*dx

# Assemble matrices
A1 = assemble(a1)
A2 = assemble(a2)
A3 = assemble(a3)


# Create files for storing solution
ufile = File("Navier-Stokes plots/velocity.pvd")
pfile = File("Navier-Stokes plots/pressure.pvd")

# Time-stepping
t = dt
i = 0

while t < T + DOLFIN_EPS:
    print("%e: %e" %(dt, u_.vector().max()*mesh.hmax()))
    # print("step %d of %d" %(i, num_steps))
    i += 1
    
    # Update pressure boundary condition
    # p_in.t = t

    # Compute tentative velocity step
    b1 = assemble(L1)
    [bc.apply(A1, b1) for bc in bcu]
    solve(A1, u1.vector(), b1, "bicgstab", "hypre_amg")

    # Pressure correction
    b2 = assemble(L2)
    [bc.apply(A2, b2) for bc in bcp]
    solve(A2, p1.vector(), b2, "bicgstab", "hypre_amg")

    # Velocity correction
    b3 = assemble(L3)
    solve(A3, u1.vector(), b3, "cg", "sor")

    # Save to file
    ufile << u1
    pfile << p1

    # Move to next time step
    u_.assign(u1)
    t += dt

# # Save velocity
# with XDMFFile(MPI.comm_world, '../Diffusion-Convection/velocity.xdmf') as xdmf:
#     xdmf.write_checkpoint(u, "velocity", 0, append=True)