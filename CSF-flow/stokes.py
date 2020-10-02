from dolfin import *
from fenics import *
from mshr import *
import matplotlib.pyplot as plt

# Test for PETSc or Tpetra
if not has_linear_algebra_backend("PETSc") and not has_linear_algebra_backend("Tpetra"):
    info("DOLFIN has not been configured with Trilinos or PETSc. Exiting.")
    exit()

if not has_krylov_solver_preconditioner("amg"):
    info("Sorry, this demo is only available when DOLFIN is compiled with AMG "
         "preconditioner, Hypre or ML.")
    exit()

if has_krylov_solver_method("minres"):
    krylov_method = "minres"
elif has_krylov_solver_method("tfqmr"):
    krylov_method = "tfqmr"
else:
    info("Default linear algebra backend was not compiled with MINRES or TFQMR "
         "Krylov subspace method. Terminating.")
    exit()

# Create mesh
meshSize = 64
domain = Rectangle(Point(0, 0), Point(0.1, 0.1))
aquaduct = Rectangle(Point(0.045, 0.01), Point(0.055, 0.045))
inside = Rectangle(Point(0.04, 0.045), Point(0.06, 0.055))
brain = Rectangle(Point(0.01, 0.01), Point(0.09, 0.09))
subdomain = brain - aquaduct - inside
domain.set_subdomain(1, subdomain)
mesh = generate_mesh(domain, meshSize)
plot(mesh)
plt.savefig("../Figures/mesh.png", bbox_inches='tight')

# Build function space
P2 = VectorElement("Lagrange", mesh.ufl_cell(), 2)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
TH = P2 * P1
W = FunctionSpace(mesh, TH)

# Viscosity inside and outside brain
mu1 = 7.9*10**(-4)
mu2 = 1000*mu1

# No-slip boundary condition for velocity on sides and bottom
noslip = Constant((0.0, 0.0))
bc0 = DirichletBC(W.sub(0), noslip, "near(x[0], 0.1) || near(x[0], 0) || near(x[1], 0)")

# Top boundary condition excluding AG
bc1 = DirichletBC(W.sub(0), noslip, "near(x[1], 0.1) && (x[0] < 0.01 || (0.04 < x[0] && x[0] < 0.065) || 0.085 < x[0])")

# Inflow from Choroid Plexus
inflow = Constant((0.0, -0.000003)) # 3 micrometer/s corresponding to 0.5 L/day flow
bc2 = DirichletBC(W.sub(0), inflow, "near(x[1], 0.055) && (0.04 < x[0] && x[0] < 0.06)")

# Collect boundary conditions
bcs = [bc0, bc1, bc2]

domain_markers = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())

dx = Measure('dx', domain=mesh, subdomain_data=domain_markers)

n = FacetNormal(mesh)

# Define variational problem
(u, p) = TrialFunctions(W)
(v, q) = TestFunctions(W)
f = Constant((0.0, 0.0))
a = -mu1*inner(grad(u), grad(v))*dx(0) - mu2*inner(grad(u), grad(v))*dx(1) - mu2*inner(grad(u), grad(v))*dx(2) + div(v)*p*dx + q*div(u)*dx
L = inner(f, v)*dx

# Form for use in constructing preconditioner matrix
b = inner(grad(u), grad(v))*dx + p*q*dx

# Assemble system
A, bb = assemble_system(a, L, bcs)

# Assemble preconditioner system
P, btmp = assemble_system(b, L, bcs)

# Create Krylov solver and AMG preconditioner
solver = KrylovSolver(krylov_method, "amg")

# Associate operator (A) and preconditioner matrix (P)
solver.set_operators(A, P)

# Solve
U = Function(W)
solver.solve(U.vector(), bb)


# Get sub-functions
u, p = U.split(deepcopy=True)
# Save solution in VTK format
ufile_pvd = File("Stokes plots/velocity.pvd")
ufile_pvd << u
pfile_pvd = File("Stokes plots/pressure.pvd")
pfile_pvd << p

# Save velocity
with XDMFFile(MPI.comm_world, '../diffusion-convection/velocity.xdmf') as xdmf:
    xdmf.write_checkpoint(u, "velocity", 0, append=True)