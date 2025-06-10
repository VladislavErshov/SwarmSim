# based on https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_modeling/mpc_modeling.py

import numpy as np
import cvxpy
from cvxpy.atoms.affine.wraps import psd_wrap


def conventional_solve(
    A, B, N, Q, R, P, x0, adim,
    umax=None, umin=None,
    xmin=None, xmax=None,
    x_star_in=None, coll_d=None
):
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
    constraints = []

    for t in range(N):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            costlist += 0.5 * cvxpy.quad_form(x[:, t] - x_star, Q)
        else:
            costlist += 0.5 * cvxpy.quad_form(x[:, t], Q)
        costlist += 0.5 * cvxpy.quad_form(u[:, t], R)

        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]

        if xmin is not None:
            constraints += [x[:, t] >= xmin[:, 0]]
        if xmax is not None:
            constraints += [x[:, t] <= xmax[:, 0]]

        # TODO: make convex
        if coll_d is not None:
            num_agents = nx // adim
            for idx in range(num_agents):
                for jdx in range(idx + 1, num_agents):
                    slice1 = x[idx * adim : (idx + 1) * adim, t]
                    slice2 = x[jdx * adim : (jdx + 1) * adim, t]
                    min_distance_constraint = cvxpy.norm1(slice1 - slice2) >= coll_d
                    constraints.append(min_distance_constraint)

    if x_star_in is not None:
        costlist += 0.5 * cvxpy.quad_form(x[:, N] - x_star, P)  # terminal cost
    else:
        costlist += 0.5 * cvxpy.quad_form(x[:, N], P)
    if xmin is not None:
        constraints += [x[:, N] >= xmin[:, 0]]
    if xmax is not None:
        constraints += [x[:, N] <= xmax[:, 0]]

    if umax is not None:
        input_constraint = u <= umax
        constraints.append(input_constraint)
    if umin is not None:
        input_constraint = u >= umin
        constraints.append(input_constraint)

    initial_state_constraint = x[:, 0] == x0
    constraints.append(initial_state_constraint)

    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constraints)
    prob.solve(verbose=False)
    cost_val = prob.value

    return x.value, u.value, cost_val


def microcoupling_solve(
    A, B, N_mic, Q, R, P, x0, adim, N_cpl,
    L=None, L_lambda=1.,
    umax=None, umin=None,
    xmin=None, xmax=None,
    x_star_in=None, coll_d=None
):
    """
    Solve a micro-scale problem with coupling 
    """
    (nx, nu) = B.shape
    
    Q = psd_wrap(Q)
    R = psd_wrap(R)
    P = psd_wrap(P)

    # mpc calculation: micro-scale variables
    x = cvxpy.Variable((nx, N_mic + 1))
    u = cvxpy.Variable((nu, N_mic))

    costlist = 0.0
    constraints = []

    # Slow-time meso-scale problem
    for t in range(N_mic):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            costlist += 0.5 * cvxpy.quad_form(x[:, t] - x_star, Q)
        else:
            costlist += 0.5 * cvxpy.quad_form(x[:, t], Q)
        #costlist += 0.5 * cvxpy.quad_form(u[:, t], R)

        constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t]]

        if xmin is not None:
            constraints += [x[:, t] >= xmin[:, 0]]
        if xmax is not None:
            constraints += [x[:, t] <= xmax[:, 0]]

        # TODO: make convex
        if coll_d is not None:
            num_agents = nx // adim
            for idx in range(num_agents):
                for jdx in range(idx + 1, num_agents):
                    slice1 = x[idx * adim : (idx + 1) * adim, t]
                    slice2 = x[jdx * adim : (jdx + 1) * adim, t]
                    min_distance_constraint = cvxpy.norm1(slice1 - slice2) >= coll_d
                    constraints.append(min_distance_constraint)

    if x_star_in is not None:
        costlist += 0.5 * cvxpy.quad_form(x[:, N_mic] - x_star, P)  # terminal cost
    else:
        costlist += 0.5 * cvxpy.quad_form(x[:, N_mic], P)
    if xmin is not None:
        constraints += [x[:, N_mic] >= xmin[:, 0]]
    if xmax is not None:
        constraints += [x[:, N_mic] <= xmax[:, 0]]

    if umax is not None:
        input_constraint = u <= umax
        constraints.append(input_constraint)
    if umin is not None:
        input_constraint = u >= umin
        constraints.append(input_constraint)

    initial_state_constraint = x[:, 0] == x0
    constraints.append(initial_state_constraint)

    # Coupling problem
    if L is not None:
        L = psd_wrap(L)
        for t in range(N_cpl):
            costlist += 0.5 * L_lambda * cvxpy.quad_form(x[:, t], L)

    # Solve 
    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constraints)
    prob.solve(verbose=False)
    cost_val = prob.value

    return x.value, u.value, cost_val


def mesocoupling_solve(
    A_mes, B_mes, N_mes, Q, R, P, x0_mes, adim,
    A_cpl, B_cpl, N_cpl, x0_cpl, L=None, L_lambda=1.,
    umax_mes=None, umin_mes=None, umax_cpl=None, umin_cpl=None,
    xmin_mes=None, xmax_mes=None, xmin_cpl=None, xmax_cpl=None,
    x_star_in=None, coll_d=None
):
    """
    Solve a meso-scale problem with coupling
    """
    (nx_mes, nu_mes) = B_mes.shape
    (nx_cpl, nu_cpl) = B_cpl.shape
    
    Q = psd_wrap(Q)
    R = psd_wrap(R)
    P = psd_wrap(P)

    # mpc calculation: meso- and micro-scale variables
    x_mes = cvxpy.Variable((nx_mes, N_mes + 1))
    x_cpl = cvxpy.Variable((nx_cpl, N_cpl + 1))
    u_mes = cvxpy.Variable((nu_mes, N_mes))
    u_cpl = cvxpy.Variable((nu_cpl, N_cpl))

    costlist = 0.0
    constraints = []

    # Slow-time meso-scale problem
    for t in range(N_mes):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            costlist += 0.5 * cvxpy.quad_form(x_mes[:, t] - x_star, Q)
        else:
            costlist += 0.5 * cvxpy.quad_form(x_mes[:, t], Q)
        costlist += 0.5 * cvxpy.quad_form(u_mes[:, t], R)

        constraints += [x_mes[:, t + 1] == A_mes * x_mes[:, t] + B_mes * u_mes[:, t]]

        if xmin_mes is not None:
            constraints += [x_mes[:, t] >= xmin_mes[:, 0]]
        if xmax_mes is not None:
            constraints += [x_mes[:, t] <= xmax_mes[:, 0]]

    if x_star_in is not None:
        costlist += 0.5 * cvxpy.quad_form(x_mes[:, N_mes] - x_star, P)  # terminal cost
    else:
        costlist += 0.5 * cvxpy.quad_form(x_mes[:, N_mes], P) 
    if xmin_mes is not None:
        constraints += [x_mes[:, N_mes] >= xmin_mes[:, 0]]
    if xmax_mes is not None:
        constraints += [x_mes[:, N_mes] <= xmax_mes[:, 0]]

    if umax_mes is not None:
        input_constraint = u_mes <= umax_mes
        constraints.append(input_constraint)
    if umin_mes is not None:
        input_constraint = u_mes >= umin_mes
        constraints.append(input_constraint)

    initial_state_constraint = x_mes[:, 0] == x0_mes
    constraints.append(initial_state_constraint)

    # Fast-time micro-scale problem
    if L is not None:
        L = psd_wrap(L)
        for t in range(N_cpl):
            costlist += 0.5 * L_lambda * cvxpy.quad_form(x_cpl[:, t], L)

            constraint = x_cpl[:, t + 1] == A_cpl * x_cpl[:, t] + B_cpl * u_cpl[:, t]
            constraints.append(constraint)

            if xmin_cpl is not None:
                constraint = x_cpl[:, t] >= xmin_cpl[:, 0]
                constraints.append(constraint)
            if xmax_cpl is not None:
                constraint = x_cpl[:, t] <= xmax_cpl[:, 0]
                constraints.append(constraint)

            # TODO: make convex
            if coll_d is not None:
                num_agents = nx_cpl // adim
                for idx in range(num_agents):
                    for jdx in range(idx + 1, num_agents):
                        slice1 = x_cpl[idx * adim : (idx + 1) * adim, t]
                        slice2 = x_cpl[jdx * adim : (jdx + 1) * adim, t]
                        min_distance_constraint = cvxpy.norm1(slice1 - slice2) >= coll_d
                        constraints.append(min_distance_constraint)

        if umax_cpl is not None:
            input_constraint = u_cpl <= umax_cpl
            constraints.append(input_constraint)
        if umin_cpl is not None:
            input_constraint = u_cpl >= umin_cpl
            constraints.append(input_constraint)

        initial_state_constraint = x_cpl[:, 0] == x0_cpl
        constraints.append(initial_state_constraint)

    # Solve
    prob = cvxpy.Problem(cvxpy.Minimize(costlist), constraints)
    prob.solve(verbose=False)
    cost_val = prob.value

    return x_mes.value, x_cpl.value, u_mes.value, u_cpl.value, cost_val
