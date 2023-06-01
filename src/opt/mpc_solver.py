# based on https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_modeling/mpc_modeling.py

import numpy as np
import cvxpy



def use_modeling_tool(A, B, N, Q, R, P, x0, 
                      umax=None, umin=None, 
                      xmin=None, xmax=None,
                      x_star_in=None):
    """
    solve MPC with modeling tool for test
    """
    (nx, nu) = B.shape

    # mpc calculation
    x = cvxpy.Variable((nx, N + 1))
    u = cvxpy.Variable((nu, N))

    costlist = 0.0
    constrlist = []

    for t in range(N):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            costlist += 0.5 * cvxpy.quad_form(x[:, t] - x_star, Q)
        else:
            costlist += 0.5 * cvxpy.quad_form(x[:, t], Q)
        costlist += 0.5 * cvxpy.quad_form(u[:, t], R)

        constrlist += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]

        if xmin is not None:
            constrlist += [x[:, t] >= xmin[:, 0]]
        if xmax is not None:
            constrlist += [x[:, t] <= xmax[:, 0]]

    costlist += 0.5 * cvxpy.quad_form(x[:, N], P)  # terminal cost
    if xmin is not None:
        constrlist += [x[:, N] >= xmin[:, 0]]
    if xmax is not None:
        constrlist += [x[:, N] <= xmax[:, 0]]

    if umax is not None:
        constrlist += [u <= umax]  # input constraints
    if umin is not None:
        constrlist += [u >= umin]  # input constraints

    constrlist += [x[:, 0] == x0]  # inital state constraints

    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)
    prob.solve(verbose=False)

    return x.value, u.value

