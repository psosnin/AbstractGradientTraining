"""
Helper functions for adding functional constraints or their relaxations to a gurobi model.
"""

from functools import wraps
import torch
import gurobipy as gp
import numpy as np

# pyright: reportOperatorIssue=false, reportArgumentType=false


def tensor_to_numpy(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Convert any Tensor args to numpy arrays
        new_args = [arg.cpu().numpy() if isinstance(arg, torch.Tensor) else arg for arg in args]
        # Convert any Tensor kwargs to numpy arrays
        new_kwargs = {k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    return wrapper


@tensor_to_numpy
def add_relu_bigm(
    model: gp.Model,
    x: gp.MVar | gp.MLinExpr,
    l: np.ndarray,
    u: np.ndarray,
    relax: bool = False,
    triangle: bool = False,
) -> tuple[gp.MVar, gp.MVar | None]:
    """
    Add the constraints defining the function y = ReLU(x) to the gurobi model.
    If relax is True, then relax the binary variables.
    Returns the MVar for y and optionally the binaries z.

    Args:
        model (gp.Model): Gurobi model to add the constraints to.
        x (gp.MVar | gp.MLinExpr): [n x 1] Gurobi MVar for the input variable.
        l (np.ndarray): [n x 1] Array of lower bounds for the input variable x.
        u (np.ndarray): [n x 1] Array of upper bounds for the input variable x.

    Returns:
        y (gp.MVar): [n x 1] MVar for the output of the ReLU
        z (gp.MVar | None): MVar for the activation set of the ReLU
    """
    # Check input shape
    assert x.shape == l.shape == u.shape, f"Shapes of x, l, and u must match, got {x.shape}, {l.shape}, {u.shape}"

    # Define output variable
    y = model.addMVar(x.shape, lb=np.maximum(l, 0), ub=np.maximum(u, 0))

    # Add big-M constraints
    model.addConstr(y >= x)
    model.addConstr(y >= 0)
    model.addConstr(x <= u)
    model.addConstr(x >= l)

    vtype = gp.GRB.CONTINUOUS if relax else gp.GRB.BINARY

    # Use either the big-M or the triangle relaxation. Using the triangle relaxation directly is slightly faster than
    # using the big-M relaxation with continuous z variables.
    if triangle:
        l = np.minimum(l, 0)
        u = np.maximum(u, 0)
        z = None
        model.addConstr((u - l) * y <= u * (x - l))
    else:
        z = model.addMVar(x.shape, lb=0.0, ub=1.0, vtype=vtype)
        model.addConstr(y <= x - l * (1 - z))
        model.addConstr(y <= u * z)

    return y, z


@tensor_to_numpy
def add_bilinear_matmul(
    model: gp.Model,
    W: gp.MVar,
    h: gp.MVar | gp.MLinExpr | gp.MQuadExpr,
    W_l: np.ndarray | torch.Tensor,
    W_u: np.ndarray | torch.Tensor,
    h_l: np.ndarray | torch.Tensor,
    h_u: np.ndarray | torch.Tensor,
    relax: bool = False,
) -> gp.MQuadExpr | gp.MLinExpr:
    """
    Add the bilinear term s = W @ h to the gurobi model. If relax is True, then the bilinear term is replaced with its
    linear envelope.

    Args:
        model (gp.Model): Gurobi model
        W (gp.MVar): [m x n] Gurobi MVar for the weight matrix
        h (gp.MVar | gp.MLinExpr): [n] Gurobi MVar for the input vector
        W_l (np.ndarray): [m x n] Lower bounds on the weight matrix
        W_u (np.ndarray): [m x n] Upper bounds on the weight matrix
        h_l (np.ndarray): [n] Lower bounds on the input vector
        h_u (np.ndarray): [n] Upper bounds on the input vector
        relax (bool, optional): If True, use the linear envelope of the bilinear term. Defaults to False.

    Returns:
        gp.MQuadExpr | gp.MLinExpr: [m] MVar representing the bilinear variable s.
    """
    # validate shapes of input
    m, n = W.shape
    assert W.shape == W_l.shape == W_u.shape
    assert h.shape == h_l.shape == h_u.shape == (n,), f"got shapes {h.shape}, {h_l.shape}, {h_u.shape}, expected {(n,)}"

    # use bilinear term
    if not relax:
        # we can return it directly, avoiding the need to create more variables
        # this significantly speeds up the QCQP bounds
        return h @ W.T

    # use linear envelope
    # expand to be shape (n, 1)
    h = h[:, np.newaxis]
    h_l = h_l[:, np.newaxis]
    h_u = h_u[:, np.newaxis]
    # matrix of bilinear terms (W.T * h)
    S = model.addMVar(W.T.shape, lb=-np.inf, ub=np.inf)
    # lower bounds
    model.addConstr(S >= W_l.T * h + W.T * h_l - W_l.T * h_l)
    model.addConstr(S >= W_u.T * h + W.T * h_u - W_u.T * h_u)
    # upper bounds
    model.addConstr(S <= W_u.T * h + W.T * h_l - W_u.T * h_l)
    model.addConstr(S <= W.T * h_u + W_l.T * h - W_l.T * h_u)
    # sum along the rows to obtain the matrix - vector product
    return S.sum(0)


@tensor_to_numpy
def add_bilinear_elementwise(
    model: gp.Model,
    a: gp.MVar,
    b: gp.MVar,
    a_l: np.ndarray,
    a_u: np.ndarray,
    b_l: np.ndarray,
    b_u: np.ndarray,
    relax: bool = False,
) -> gp.MVar | gp.MQuadExpr:
    """
    Add the bilinear term s = a * b to the gurobi model. If relax is True, then the bilinear term is replaced with its
    linear envelope.

    Args:
        model (gp.Model): Gurobi model
        a (gp.MVar): [n x 1] Gurobi MVar for the input vector
        b (gp.MVar): [n x 1] Gurobi MVar for the input vector
        a_l (np.ndarray): [n x 1] Lower bounds on the vector a
        a_u (np.ndarray): [n x 1] Upper bounds on the vector a
        b_l (np.ndarray): [n x 1] Lower bounds on the vector b
        b_u (np.ndarray): [n x 1] Upper bounds on the vector b
        relax (bool, optional): If True, use the linear envelope of the bilinear term. Defaults to False.

    Returns:
        gp.MVar | gp.MQuadExpr: [m x 1] MVar representing the bilinear variable s.
    """
    # validate shapes of input
    assert a.shape == a_l.shape == a_u.shape
    assert b.shape == b_l.shape == b_u.shape

    # use bilinear term
    if not relax:
        return a * b

    s = model.addMVar((a * b).shape, lb=-np.inf)
    # use linear envelope
    # lower bounds
    model.addConstr(s >= a_l * b + a * b_l - a_l * b_l)
    model.addConstr(s >= a_u * b + a * b_u - a_u * b_u)
    # upper bounds
    model.addConstr(s <= a_u * b + a * b_l - a_u * b_l)
    model.addConstr(s <= a * b_u + a_l * b - a_l * b_u)

    return s


@tensor_to_numpy
def add_heaviside(m: gp.Model, x: gp.MVar, l: np.ndarray, u: np.ndarray, relax_binaries: bool) -> gp.MVar:
    """
    Add the term z = Heaviside(x) to the gurobi model. The Heaviside function is defined as
        z = 1 if x > 0
        z = 0 if x <= 0
    If relax_binaries is True, then we'll use the linear relaxation.

    Args:
        model (gp.Model): Gurobi model to add the constraints to.
        x (gp.MVar): [n x 1] Gurobi MVar for the input variable.
        l (np.ndarray): [n x 1] Array of lower bounds for the input variable x.
        u (np.ndarray): [n x 1] Array of upper bounds for the input variable x.
        relax_binaries (bool): If True, use the linear relaxation of the Heaviside function.

    Returns:
        z (gp.MVar): [n x 1] MVar for the output of the Heaviside function.
    """
    vtype = gp.GRB.CONTINUOUS if relax_binaries else gp.GRB.BINARY
    z = m.addMVar(shape=x.shape, lb=0, ub=1, vtype=vtype)
    m.addConstr(x <= z * u)
    m.addConstr(x >= (1 - z) * l)
    m.addConstr(np.heaviside(l, 0) <= z)
    m.addConstr(np.heaviside(u, 0) >= z)
    return z


@tensor_to_numpy
def add_softmax(m: gp.Model, x: gp.MVar | gp.MLinExpr) -> gp.MVar:
    """
    Add the constraints defining the softmax function y = softmax(x) to the gurobi model.

    Args:
        m (gp.Model): Gurobi model
        x (gp.MVar): Input MVar to the softmax
    """
    # add intermediate exp and output variable
    e = m.addMVar(x.shape, lb=0)
    y = m.addMVar(x.shape, lb=0, ub=1)

    if isinstance(x, gp.MLinExpr):
        z = m.addMVar(x.shape, lb=-np.inf)
        m.addConstr(z == x)
        x = z

    # add exponential constraints
    for xi, ei in zip(x, e):
        m.addGenConstrExp(xi, ei)

    # add softmax constraint
    m.addConstr(e.sum() * y == e)

    return y


if __name__ == "__main__":
    # Example usage of relu
    m = gp.Model()
    x = m.addMVar(2, lb=-1, ub=1)
    l = np.array([-1, -1])
    u = np.array([1, 1])
    y, z = add_relu_bigm(m, x, l, u, relax=False)
    m.optimize()
    print(np.maximum(x.X, 0))
    print(y.X)

    # Example usage of bilinear matmul
    m = gp.Model()
    W = m.addMVar((3, 2), lb=-1, ub=1)
    h = m.addMVar(2, lb=-1, ub=1)
    W_l = np.array([[-1, -1], [-1, -1], [-1, -1]])
    W_u = np.array([[1, 1], [1, 1], [1, 1]])
    h_l = np.array([-1, -1])
    h_u = np.array([1, 1])
    s = add_bilinear_matmul(m, W, h, W_l, W_u, h_l, h_u, relax=True)
    m.optimize()
    print(W.X @ h.X)
    print(s.getValue())  # type: ignore

    # Example usage of bilinear elementwise
    m = gp.Model()
    a = m.addMVar(2, lb=-1, ub=1)
    b = m.addMVar(2, lb=-1, ub=1)
    a_l = np.array([-1, -1])
    a_u = np.array([1, 1])
    b_l = np.array([-1, -1])
    b_u = np.array([1, 1])
    s = add_bilinear_elementwise(m, a, b, a_l, a_u, b_l, b_u, relax=True)
    m.optimize()
    print(a.X * b.X)
    print(s.X)  # type: ignore
