"""
Provides helper functions for defining and working with Gurobi models. 
"""

import gurobipy as gp
import numpy as np


def bound_objective(model: gp.Model, objective: gp.MVar | gp.MLinExpr | gp.MQuadExpr) -> tuple[np.ndarray, np.ndarray]:
    """
    Given a gurobi model and a vector / matrix of objectives, compute the elementwise minimum and maximum value of the
    objective over the model.

    Args:
        model (gp.Model): Gurobi model
        objective (gp.MVar | gp.MLinExpr | gp.MQuadExpr): Objective to minimize/maximize over, either a gurobi variable
            or expression.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple of two numpy arrays, the first containing the minimum of each objective
                                       and the second containing the maximum value.
    """
    L, U = np.zeros(objective.size), np.zeros(objective.size)
    for i in range(objective.size):
        obj = objective[np.unravel_index(i, objective.shape)]  # type: ignore
        # lower bound
        model.setObjective(obj, gp.GRB.MINIMIZE)
        model.reset()
        model.optimize()
        if model.status == gp.GRB.OPTIMAL:  # if model is solved, store the objective value
            L[i] = model.objVal
        elif model.status == gp.GRB.INFEASIBLE:  # if model is infeasible, store infinity
            L[i] = -np.inf
        else:  # otherwise use the best bound
            L[i] = getattr(model, "objBound", -np.inf)
        # upper bound
        model.setObjective(obj, gp.GRB.MAXIMIZE)
        model.reset()
        model.optimize()
        if model.status == gp.GRB.OPTIMAL:
            U[i] = model.objVal
        elif model.status == gp.GRB.INFEASIBLE:
            U[i] = np.inf
        else:
            U[i] = getattr(model, "objBound", np.inf)
    return L.reshape(objective.shape), U.reshape(objective.shape)


def init_gurobi_model(name: str, quiet: bool = True) -> gp.Model:
    """
    Initialise a blank Gurobi model. Setting quiet = True will suppress all output from the model.
    """
    env = gp.Env(empty=True)
    env.setParam("LogToConsole", 0)
    env.start()
    m = gp.Model(name=name, env=env) if quiet else gp.Model(name=name)
    return m


def get_gurobi_model_stats(model: gp.Model) -> str:
    """
    Return a string with statistics about the Gurobi model.

    Args:
        model (gp.Model):Gurobi model

    Returns:
        str: Model statistics in a human-readable format.
    """
    return (
        f"Statistics for model {model.ModelName}:\n"
        f"  {'Linear constraint matrix':<30}: {model.NumConstrs} Constrs, {model.NumVars} Vars, {model.NumNZs} NZs\n"
        f"  {'Quadratic constraints':<30}: {model.NumQConstrs} QConstrs, {model.NumQCNZs} QNZs\n"
        f"  {'SOS constraints':<30}: {model.NumSOS} SOS\n"
        f"  {'General constraints':<30}: {model.NumGenConstrs} GenConstrs\n"
        f"  {'Quadratic objective':<30}: {model.NumQNZs} NZs, {model.NumPWLObjVars} PWL \n"
        f"  {'Integer variables':<30}: {model.NumIntVars} IntVars, {model.NumBinVars} BinVars\n"
        f"  {'Solve time':<30}: {model.Runtime:.2f}s\n"
        f"  {'Status':<30}: {model.Status}\n"
    )
