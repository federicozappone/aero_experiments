import trimesh
import numpy as np
import scipy.sparse as sp
import scipy.solve as slv

# Load the 3D mesh using trimesh
mesh = trimesh.creation.uv_sphere()

# Define the fluid properties
rho = 1.225 # density of air [kg/m^3]
mu = 1.8e-5 # viscosity of air [Pa*s]
velocity = np.array([100.0, 0.0, 0.0]) # velocity of the fluid [m/s]

# Define the geometry of the mesh
surface_area = mesh.area # surface area of the mesh [m^2]
unit_normals = mesh.face_normals # unit normal vectors at each mesh vertex [m]

# Discretize the domain
elements = mesh.triangles # list of triangular elements
num_nodes = mesh.vertices.shape[0] # number of nodes in the mesh
num_elements = elements.shape[0] # number of elements in the mesh

# Define the shape functions
def shape_functions(xi, eta, zeta):
    """
    Compute the shape functions for a tetrahedron element.
    
    Parameters
    ----------
    xi : float
        Natural coordinate in the xi direction.
    eta : float
        Natural coordinate in the eta direction.
    zeta : float
        Natural coordinate in the zeta direction.
    
    Returns
    -------
    N : numpy array
        Shape functions for the element.
    """
    # Compute the shape functions
    N = np.array([1 - xi - eta - zeta, xi, eta, zeta])
    
    return N


def element_stiffness_matrix(e):
    """
    Compute the element stiffness matrix for element e.
    
    Parameters
    ----------
    e : int
        Index of the element.
    
    Returns
    -------
    Ae : numpy array
        Element stiffness matrix for element e.
    """
    # Get the element vertices and their coordinates
    x1, y1, z1 = mesh.vertices[elements[e,0],:]
    x2, y2, z2 = mesh.vertices[elements[e,1],:]
    x3, y3, z3 = mesh.vertices[elements[e,2],:]
    x4, y4, z4 = mesh.vertices[elements[e,2],:]
    
    # Define the test functions
    def v(xi, eta, zeta):
        return shape_functions(xi, eta, zeta)
    def w(xi, eta, zeta):
        return np.array([xi, eta, zeta])

    # Define the Gauss quadrature weights and points
    w1 = 1/24
    xi1 = 1/4
    eta1 = 1/4
    zeta1 = 1/4

    w2 = 1/24
    xi2 = 1/4
    eta2 = 1/4
    zeta2 = 1/4

    w3 = 1/24
    xi3 = 1/4
    eta3 = 1/4
    zeta3 = 1/4

    w4 = 1/24
    xi4 = 1/4
    eta4 = 1/4
    zeta4 = 1/4
    
    # Compute the element stiffness matrix using the Gauss quadrature
    Ae = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            # Compute the Jacobian of the transformation from natural to Cartesian coordinates
            J = np.array([[x2 - x1, x3 - x1, x4 - x1],
                          [y2 - y1, y3 - y1, y4 - y1],
                          [z2 - z1, z3 - z1, z4 - z1]])

            # Compute the determinant of the Jacobian
            detJ = np.linalg.det(J)
            
            # Compute the element stiffness matrix using the Gauss quadrature
            Ae[i,j] = w1*(np.dot(v(xi1, eta1, zeta1), np.dot(velocity, w(xi1, eta1, zeta1)))*detJ
                     - np.dot(shape_functions(xi1, eta1, zeta1), pressure)*detJ
                     + np.dot(shape_functions(xi1, eta1, zeta1), np.dot(2*mu*strain_rate_tensor, w(xi1, eta1, zeta1)))*detJ
                     + np.dot(shape_functions(xi1, eta1, zeta1), body_force)*detJ)

            Ae[i,j] += w2*(np.dot(v(xi2, eta2, zeta2), np.dot(velocity, w(xi2, eta2, zeta2)))*detJ
                     - np.dot(shape_functions(xi2, eta2, zeta2), pressure)*detJ
                     + np.dot(shape_functions(xi2, eta2, zeta2), np.dot(2*mu*strain_rate_tensor, w(xi2, eta2, zeta2)))*detJ
                     + np.dot(shape_functions(xi2, eta2, zeta2), body_force)*detJ)

            Ae[i,j] += w3*(np.dot(v(xi3, eta3, zeta3), np.dot(velocity, w(xi3, eta3, zeta3)))*detJ
                     - np.dot(shape_functions(xi3, eta3, zeta3), pressure)*detJ
                     + np.dot(shape_functions(xi3, eta3, zeta3), np.dot(2*mu*strain_rate_tensor, w(xi3, eta3, zeta3)))*detJ
                     + np.dot(shape_functions(xi3, eta3, zeta3), body_force)*detJ)

            Ae[i,j] += w4*(np.dot(v(xi4, eta4, zeta4), np.dot(velocity, w(xi4, eta4, zeta4)))*detJ
                     - np.dot(shape_functions(xi4, eta4, zeta4), pressure)*detJ
                     + np.dot(shape_functions(xi4, eta4, zeta4), np.dot(2*mu*strain_rate_tensor, w(xi4, eta4, zeta4)))*detJ
                     + np.dot(shape_functions(xi4, eta4, zeta4), body_force)*detJ)

    return Ae



# Assemble the global stiffness matrix
A = sp.lil_matrix((num_nodes, num_nodes))
for e in range(num_elements):
    # Evaluate the element stiffness matrix for element e
    Ae = element_stiffness_matrix(e)
    # Add the element stiffness matrix to the global stiffness matrix
    for i in range(3):
        for j in range(3):
            A[elements[e,i], elements[e,j]] += Ae[i,j]

# Solve the linear system using the conjugate gradient method
b = np.zeros(num_nodes) # RHS vector
x, info = slv.cg(A, b) # solution vector

# Extract the velocity and pressure from the solution vector
velocity = x[:num_nodes] # velocity at each node
pressure = x[num_nodes:] # pressure at each node
