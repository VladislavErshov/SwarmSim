# based on https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_modeling/mpc_modeling.py

import numpy as np
import cvxpy
from cvxpy.atoms.affine.wraps import psd_wrap



def conventional_solve(A, B, N, Q, R, P, x0, adim,
                       umax=None, umin=None, 
                       xmin=None, xmax=None,
                       x_star_in=None, coll_d=None):
    """
    Solve a multi-agent MPC problem [TODO: with collision avoidance]
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

        # TODO: make convex
        if coll_d is not None:
            for idx in range(nx//adim):
                for jdx in range(idx+1, nx//adim):
                    constrlist += [cvxpy.norm1(x[idx * adim : (idx+1) * adim, t] - x[jdx * adim : (jdx+1) * adim, t]) >= coll_d]

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


def microcoupling_solve(A, B, N_mic, Q, R, P, x0, adim, N_cpl,
                        L=None, L_lambda=1., 
                        umax=None, umin=None,
                        xmin=None, xmax=None,
                        x_star_in=None, coll_d=None):
    """
    Solve a micro-scale problem with coupling 
    """
    (nx, nu) = B.shape
    
    Q = psd_wrap(Q)
    R = psd_wrap(R)
    P = psd_wrap(P)

    # mpc calculation: micro-scale wariables
    x = cvxpy.Variable((nx, N_mic + 1))
    u = cvxpy.Variable((nu, N_mic))

    costlist = 0.0
    constrlist = []

    # Slow-time meso-scale problem
    for t in range(N_mic):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            costlist += 0.5 * cvxpy.quad_form(x[:, t] - x_star, Q)
        else:
            costlist += 0.5 * cvxpy.quad_form(x[:, t], Q)
        #costlist += 0.5 * cvxpy.quad_form(u[:, t], R)

        constrlist += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]

        if xmin is not None:
            constrlist += [x[:, t] >= xmin[:, 0]]
        if xmax is not None:
            constrlist += [x[:, t] <= xmax[:, 0]]

        # TODO: make convex
        if coll_d is not None:
            for idx in range(nx//adim):
                for jdx in range(idx+1, nx//adim - 1):
                    constrlist += [cvxpy.norm1(x[idx * adim : (idx+1) * adim, t] - x[jdx * adim : (jdx+1) * adim, t]) >= coll_d]

    costlist += 0.5 * cvxpy.quad_form(x[:, N_mic], P)  # terminal cost
    if xmin is not None:
        constrlist += [x[:, N_mic] >= xmin[:, 0]]
    if xmax is not None:
        constrlist += [x[:, N_mic] <= xmax[:, 0]]

    if umax is not None:
        constrlist += [u <= umax]  # input constraints
    if umin is not None:
        constrlist += [u >= umin]  # input constraints

    constrlist += [x[:, 0] == x0]  # inital state constraints

    # Coupling problem
    if L is not None:
        L = psd_wrap(L)
        for t in range(N_cpl):
            costlist += 0.5 * L_lambda * cvxpy.quad_form(x[:, t], L)

    # Solve 
    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)
    prob.solve(verbose=False)
    cost_val = prob.value

    return x.value, u.value, cost_val


def mesocoupling_solve(A_mes, B_mes, N_mes, Q, R_mes, P, x0_mes, adim,
                       A_cpl, B_cpl, N_cpl, R_cpl, x0_cpl, L=None, L_lambda=1., 
                       umax_mes=None, umin_mes=None, umax_cpl=None, umin_cpl=None,
                       xmin_mes=None, xmax_mes=None, xmin_cpl=None, xmax_cpl=None,
                       x_star_in=None, coll_d=None):
    """
    Solve a meso-scale problem with coupling
    """
    (nx_mes, nu_mes) = B_mes.shape
    (nx_cpl, nu_cpl) = B_cpl.shape
    
    Q = psd_wrap(Q)
    R_mes = psd_wrap(R_mes)
    R_cpl = psd_wrap(R_cpl)
    P = psd_wrap(P)

    # mpc calculation: meso- and micro-scale wariables
    x_mes = cvxpy.Variable((nx_mes, N_mes + 1))
    x_cpl = cvxpy.Variable((nx_cpl, N_cpl + 1))
    u_mes = cvxpy.Variable((nu_mes, N_mes))
    u_cpl = cvxpy.Variable((nu_cpl, N_cpl))

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
        for t in range(N_cpl):
            costlist += 0.5 * L_lambda * cvxpy.quad_form(x_cpl[:, t], L)
            #costlist += 0.5 * cvxpy.quad_form(u_cpl[:, t], R_cpl)

            constrlist += [x_cpl[:, t + 1] == A_cpl * x_cpl[:, t] + B_cpl * u_cpl[:, t]]

            if xmin_cpl is not None:
                constrlist += [x_cpl[:, t] >= xmin_cpl[:, 0]]
            if xmax_cpl is not None:
                constrlist += [x_cpl[:, t] <= xmax_cpl[:, 0]]

            # TODO: make convex
            if coll_d is not None:
                for idx in range(nx_cpl//adim):
                    for jdx in range(idx+1, nx_cpl//adim - 1):
                        constrlist += [cvxpy.norm1(x_cpl[idx * adim : (idx+1) * adim, t] - x_cpl[jdx * adim : (jdx+1) * adim, t]) >= coll_d]

        if umax_cpl is not None:
            constrlist += [u_cpl <= umax_cpl]  # input constraints
        if umin_cpl is not None:
            constrlist += [u_cpl >= umin_cpl]  # input constraints

        constrlist += [x_cpl[:, 0] == x0_cpl]  # inital state constraints    

    # Solve 
    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constrlist)
    prob.solve(verbose=False)
    cost_val = prob.value

    return x_mes.value, x_cpl.value, u_mes.value, u_cpl.value, cost_val

