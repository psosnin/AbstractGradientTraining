"""
Helper functions for adding functional constraints to a gurobi model.
"""

import gurobipy as gp
import numpy as np

# pyright: reportOperatorIssue=false, reportArgumentType=false


def add_relu_bigm(
    model: gp.Model,
    x: gp.MVar | gp.MLinExpr | gp.MQuadExpr,
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
        x (gp.MVar | gp.MLinExpr | gp.MQuadExpr): [n x 1] Gurobi MVar for the input variable.
        l (np.ndarray): [n x 1] Array of lower bounds for the input variable x.
        u (np.ndarray): [n x 1] Array of upper bounds for the input variable x.

    Returns:
        y (gp.MVar): [n x 1] MVar for the output of the ReLU
        z (gp.MVar | None): MVar for the activation set of the ReLU
    """
    # Check input shape
    assert x.shape == l.shape == u.shape

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
        model.addConstr(y <= u * (x - l) / (u - l))
    else:
        z = model.addMVar(x.shape, vtype=vtype)
        model.addConstr(y <= x - l * (1 - z))
        model.addConstr(y <= u * z)

    return y, z


def add_bilinear_matmul(
    model: gp.Model,
    W: gp.MVar,
    h: gp.MVar | gp.MLinExpr | gp.MQuadExpr,
    W_l: np.ndarray,
    W_u: np.ndarray,
    h_l: np.ndarray,
    h_u: np.ndarray,
    relax: bool = False,
) -> gp.MVar | gp.MQuadExpr:
    """
    Add the bilinear term s = W @ h to the gurobi model. If relax is True, then the bilinear term is replaced with its
    linear envelope.

    Args:
        model (gp.Model): Gurobi model
        W (gp.MVar): [m x n] Gurobi MVar for the weight matrix
        h (gp.MVar | gp.MLinExpr | gp.MQuadExpr): [n x 1] Gurobi MVar for the input vector
        W_l (np.ndarray): [m x n] Lower bounds on the weight matrix
        W_u (np.ndarray): [m x n] Upper bounds on the weight matrix
        h_l (np.ndarray): [n x 1] Lower bounds on the input vector
        h_u (np.ndarray): [n x 1] Upper bounds on the input vector
        relax (bool, optional): If True, use the linear envelope of the bilinear term. Defaults to False.

    Returns:
        gp.MVar | gp.MQuadExpr: [m x 1] MVar or MQuadExpr representing the bilinear variable s.
    """
    # validate shapes of input
    m, n = W.shape
    assert W.shape == W_l.shape == W_u.shape
    assert h.shape == h_l.shape == h_u.shape == (n, 1)

    # use bilinear term
    if not relax:
        return W @ h

    # use linear envelope
    s = model.addMVar((m, 1), lb=-np.inf)
    # matrix of bilinear terms (W.T * h)
    S = model.addMVar(W.T.shape, lb=-np.inf, ub=np.inf)
    # lower bounds
    model.addConstr(S >= W_l.T * h + W.T * h_l - W_l.T * h_l)
    model.addConstr(S >= W_u.T * h + W.T * h_u - W_u.T * h_u)
    # upper bounds
    model.addConstr(S <= W_u.T * h + W.T * h_l - W_u.T * h_l)
    model.addConstr(S <= W.T * h_u + W_l.T * h - W_l.T * h_u)
    # sum along the rows to obtain the matrix - vector product
    model.addConstr(s == S.sum(0)[:, None])
    return s


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
        gp.MVar | gp.MQuadExpr: [m x 1] MVar or MQuadExpr representing the bilinear variable s.
    """
    # validate shapes of input
    assert a.shape == a_l.shape == a_u.shape
    assert b.shape == b_l.shape == b_u.shape

    # use bilinear term
    if not relax:
        return a * b

    # use linear envelope
    s = model.addMVar((a * b).shape, lb=-np.inf)
    # lower bounds
    model.addConstr(s >= a_l * b + a * b_l - a_l * b_l)
    model.addConstr(s >= a_u * b + a * b_u - a_u * b_u)
    # upper bounds
    model.addConstr(s <= a_u * b + a * b_l - a_u * b_l)
    model.addConstr(s <= a * b_u + a_l * b - a_l * b_u)

    return s


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


def add_softmax(m: gp.Model, x: gp.MVar | gp.MLinExpr | gp.MQuadExpr) -> gp.MVar:
    """
    Add the constraints defining the softmax function y = softmax(x) to the gurobi model.

    Args:
        m (gp.Model): Gurobi model
        x (gp.MVar): Input MVar to the softmax
    """
    # add intermediate exp and output variable
    e = m.addMVar(x.shape, lb=0)
    y = m.addMVar(x.shape, lb=0, ub=1)

    if isinstance(x, (gp.MLinExpr, gp.MQuadExpr)):
        z = m.addMVar(x.shape, lb=-np.inf)
        m.addConstr(z == x)
        x = z

    # add exponential constraints
    for xi, ei in zip(x, e):
        m.addGenConstrExp(xi, ei)

    # add softmax constraint
    m.addConstr(e.sum() * y == e)

    return y


def add_loss_gradient(m: gp.Model, logits: gp.MVar, label: np.ndarray, loss: str) -> gp.MLinExpr:
    """
    Add the gradient of the loss function with respect to the logits to the gurobi model.

    Args:
        m (gp.Model): Gurobi model to add the constraints to.
        logits (gp.MVar): [n x 1] Gurobi MVar for the logits.
        label (np.ndarray): [1] True label
        loss (str): Name of the loss function.

    Returns:
        dL (gp.MLinExpr): [n x 1] MLinExpr for the gradient of the loss function with respect to the logits.
    """

    if loss == "cross_entropy":
        # get one-hot encoding of labels
        y_t = np.zeros(logits.shape)
        y_t[label] = 1
        y = add_softmax(m, logits)
        loss_val = y - y_t
    else:
        raise NotImplementedError(f"Loss fn {loss} not implemented.")

    return loss_val
