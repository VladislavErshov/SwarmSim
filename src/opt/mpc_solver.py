# based on https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_modeling/mpc_modeling.py

import numpy as np
import cvxpy
from cvxpy.atoms.affine.wraps import psd_wrap



def use_modeling_tool(A, B, N, Q, R, P, x0,
                      umax=None, umin=None, 
                      xmin=None, xmax=None,
                      x_star_in=None):
    """
    Solve a simple conventional MPC problem 
    """
    (nx, nu) = B.shape
    Q = psd_wrap(Q)
    R = psd_wrap(R)
    P = psd_wrap(P)

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
    cost_val = prob.value

    return x.value, u.value, cost_val


def mesocoupling_solve(A_mes, B_mes, N_mes, Q, R_mes, P, x0_mes,
                       A_mic, B_mic, N_mic, R_mic, x0_mic, L=None, L_lambda=1., 
                       umax_mes=None, umin_mes=None, umax_mic=None, umin_mic=None,
                       xmin_mes=None, xmax_mes=None, xmin_mic=None, xmax_mic=None,
                       x_star_in=None):
    """
    Solve a meso-micro-scale problem 
    """
    (nx_mes, nu_mes) = B_mes.shape
    (nx_mic, nu_mic) = B_mic.shape
    
    Q = psd_wrap(Q)
    R_mes = psd_wrap(R_mes)
    R_mic = psd_wrap(R_mic)
    P = psd_wrap(P)

    # mpc calculation: meso- and micro-scale wariables
    x_mes = cvxpy.Variable((nx_mes, N_mes + 1))
    x_mic = cvxpy.Variable((nx_mic, N_mic + 1))
    u_mes = cvxpy.Variable((nu_mes, N_mes))
    u_mic = cvxpy.Variable((nu_mic, N_mic))

    costlist = 0.0
    constrlist = []

    # Slow-time meso-scale problem
    for t in range(N_mes):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            costlist += 0.5 * cvxpy.quad_form(x_mes[:, t] - x_star, Q)
        else:
            costlist += 0.5 * cvxpy.quad_form(x_mes[:, t], Q)
        costlist += 0.5 * cvxpy.quad_form(u_mes[:, t], R_mes)

        constrlist += [x_mes[:, t + 1] == A_mes * x_mes[:, t] + B_mes * u_mes[:, t]]

        if xmin_mes is not None:
            constrlist += [x_mes[:, t] >= xmin_mes[:, 0]]
        if xmax_mes is not None:
            constrlist += [x_mes[:, t] <= xmax_mes[:, 0]]

    costlist += 0.5 * cvxpy.quad_form(x_mes[:, N_mes], P)  # terminal cost
    if xmin_mes is not None:
        constrlist += [x_mes[:, N_mes] >= xmin_mes[:, 0]]
    if xmax_mes is not None:
        constrlist += [x_mes[:, N_mes] <= xmax_mes[:, 0]]

    if umax_mes is not None:
        constrlist += [u_mes <= umax_mes]  # input constraints
    if umin_mes is not None:
        constrlist += [u_mes >= umin_mes]  # input constraints

    constrlist += [x_mes[:, 0] == x0_mes]  # inital state constraints

    # Fast-time micro-scale problem
    if L is not None:
        L = psd_wrap(L)
        for t in range(N_mic):
            costlist += 0.5 * L_lambda * cvxpy.quad_form(x_mic[:, t], L)
            costlist += 0.5 * cvxpy.quad_form(u_mic[:, t], R_mic)

            constrlist += [x_mic[:, t + 1] == A_mic * x_mic[:, t] + B_mic * u_mic[:, t]]

            if xmin_mic is not None:
                constrlist += [x_mic[:, t] >= xmin_mic[:, 0]]
            if xmax_mic is not None:
                constrlist += [x_mic[:, t] <= xmax_mic[:, 0]]

        if umax_mic is not None:
            constrlist += [u_mic <= umax_mic]  # input constraints
        if umin_mic is not None:
            constrlist += [u_mic >= umin_mic]  # input constraints

        constrlist += [x_mic[:, 0] == x0_mic]  # inital state constraints    

    # Solve 
    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)
    prob.solve(verbose=False)
    cost_val = prob.value

    return x_mes.value, x_mic.value, u_mes.value, u_mic.value, cost_val

