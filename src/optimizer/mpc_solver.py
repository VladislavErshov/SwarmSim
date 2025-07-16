# based on https://github.com/AtsushiSakai/PyAdvancedControl/blob/master/mpc_modeling/mpc_modeling.py

import numpy as np
import cvxpy
from cvxpy.atoms.affine.wraps import psd_wrap


def conventional_solve(
    A, B, N, Q, R, P, x0, a_dim,
    u_max=None, u_min=None,
    x_min=None, x_max=None,
    x_star_in=None, coll_d=None,
    obstacles=None
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

    cost = 0.0
    constraints = []

    for t in range(N):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            cost += 0.5 * cvxpy.quad_form(x[:, t] - x_star, Q)
        else:
            cost += 0.5 * cvxpy.quad_form(x[:, t], Q)
        cost += 0.5 * cvxpy.quad_form(u[:, t], R)

        constraints += create_state_update_equation_constraint(A, B, t, u, x)
        constraints += create_max_min_state_constraints(t, x, x_max, x_min)
        constraints += create_min_distance_constraints(a_dim, coll_d, nx, t, x)
        constraints += create_avoid_obstacle_constraints(a_dim, coll_d, nx, t, x, obstacles)

    if x_star_in is not None:
        cost += 0.5 * cvxpy.quad_form(x[:, N] - x_star, P)  # terminal cost
    else:
        cost += 0.5 * cvxpy.quad_form(x[:, N], P)

    constraints += create_max_min_state_constraints(N, x, x_max, x_min)
    constraints += create_input_constraints(u, u_max, u_min)
    constraints += create_initial_state_constraint(x, x0)

    problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    problem.solve(verbose=False)
    cost_val = problem.value

    return x.value, u.value, cost_val


def microcoupling_solve(
    A, B, N_mic, Q, R, P, x0, a_dim, N_cpl,
    L=None, L_lambda=1.0,
    u_max=None, u_min=None,
    x_min=None, x_max=None,
    x_star_in=None, coll_d=None,
    obstacles=None
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

    cost = 0.0
    constraints = []

    # Slow-time meso-scale problem
    for t in range(N_mic):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            cost += 0.5 * cvxpy.quad_form(x[:, t] - x_star, Q)
        else:
            cost += 0.5 * cvxpy.quad_form(x[:, t], Q)
        #cost += 0.5 * cvxpy.quad_form(u[:, t], R)

        constraints += create_state_update_equation_constraint(A, B, t, u, x)
        constraints += create_max_min_state_constraints(t, x, x_max, x_min)
        constraints += create_min_distance_constraints(a_dim, coll_d, nx, t, x)
        constraints += create_avoid_obstacle_constraints(a_dim, coll_d, nx, t, x, obstacles)

    if x_star_in is not None:
        cost += 0.5 * cvxpy.quad_form(x[:, N_mic] - x_star, P)  # terminal cost
    else:
        cost += 0.5 * cvxpy.quad_form(x[:, N_mic], P)

    constraints += create_max_min_state_constraints(N_mic, x, x_max, x_min)
    constraints += create_input_constraints(u, u_max, u_min)
    constraints += create_initial_state_constraint(x, x0)

    # Coupling problem
    if L is not None:
        L = psd_wrap(L)
        for t in range(N_cpl):
            cost += 0.5 * L_lambda * cvxpy.quad_form(x[:, t], L)

    # Solve 
    problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    problem.solve(verbose=False)
    cost_val = problem.value

    return x.value, u.value, cost_val


def mesocoupling_solve(
    A_mes, B_mes, N_mes, Q, R, P, x0_mes, a_dim,
    A_cpl, B_cpl, N_cpl, x0_cpl, L=None, L_lambda=1.,
    u_max_mes=None, u_min_mes=None, u_max_cpl=None, u_min_cpl=None,
    x_min_mes=None, x_max_mes=None, x_min_cpl=None, x_max_cpl=None,
    x_star_in=None, coll_d=None,
    obstacles=None
):
    """Solve a meso-scale problem with coupling."""
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

    cost = 0.0
    constraints = []

    # Slow-time meso-scale problem
    for t in range(N_mes):
        if x_star_in is not None:
            x_star = cvxpy.Constant(x_star_in)
            cost += 0.5 * cvxpy.quad_form(x_mes[:, t] - x_star, Q)
        else:
            cost += 0.5 * cvxpy.quad_form(x_mes[:, t], Q)
        cost += 0.5 * cvxpy.quad_form(u_mes[:, t], R)

        constraints += create_state_update_equation_constraint(A_mes, B_mes, t, u_mes, x_mes)
        constraints += create_max_min_state_constraints(t, x_mes, x_max_mes, x_min_mes)

    if x_star_in is not None:
        cost += 0.5 * cvxpy.quad_form(x_mes[:, N_mes] - x_star, P)  # terminal cost
    else:
        cost += 0.5 * cvxpy.quad_form(x_mes[:, N_mes], P)

    constraints += create_max_min_state_constraints(N_mes, x_mes, x_max_mes, x_min_mes)
    constraints += create_input_constraints(u_mes, u_max_mes, u_min_mes)
    constraints += create_initial_state_constraint(x_mes, x0_mes)

    # Fast-time micro-scale problem
    if L is not None:
        L = psd_wrap(L)
        for t in range(N_cpl):
            cost += 0.5 * L_lambda * cvxpy.quad_form(x_cpl[:, t], L)

            constraints += create_state_update_equation_constraint(A_cpl, B_cpl, t, u_cpl, x_cpl)
            constraints += create_max_min_state_constraints(t, x_cpl, x_max_cpl, x_min_cpl)
            constraints += create_min_distance_constraints(a_dim, coll_d, nx_cpl, t, x_cpl)
            constraints += create_avoid_obstacle_constraints(a_dim, coll_d, nx_cpl, t, x_cpl, obstacles)

        constraints += create_input_constraints(u_cpl, u_max_cpl, u_min_cpl)
        constraints += create_initial_state_constraint(x_cpl, x0_cpl)

    # Solve
    problem = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    problem.solve(verbose=False)
    cost_val = problem.value

    return x_mes.value, x_cpl.value, u_mes.value, u_cpl.value, cost_val


def create_state_update_equation_constraint(A, B, t, u, x):
    return [x[:, t + 1] == A * x[:, t] + B * u[:, t]]


def create_max_min_state_constraints(t, x, x_max, x_min):
    max_min_state_constraints = []
    if x_min is not None:
        min_state_constraint = x[:, t] >= x_min[:, 0]
        max_min_state_constraints.append(min_state_constraint)
    if x_max is not None:
        max_state_constraint = x[:, t] <= x_max[:, 0]
        max_min_state_constraints.append(max_state_constraint)

    return max_min_state_constraints


def create_initial_state_constraint(x, x0):
    return [x[:, 0] == x0]


def create_input_constraints(u, u_max, u_min):
    input_constraints = []
    if u_max is not None:
        max_constraint = u <= u_max
        input_constraints.append(max_constraint)
    if u_min is not None:
        min_constraint = u >= u_min
        input_constraints.append(min_constraint)

    return input_constraints


def create_min_distance_constraints(a_dim, coll_d, nx, t, x):
    min_distance_constraints = []
    if coll_d is None:
        return []

    # TODO: make convex
    num_agents = nx // a_dim
    for idx in range(num_agents):
        for jdx in range(idx + 1, num_agents):
            agent_i_center = x[idx * a_dim : (idx + 1) * a_dim, t]
            agent_j_center = x[jdx * a_dim : (jdx + 1) * a_dim, t]
            min_distance_constraint = cvxpy.norm1(agent_i_center - agent_j_center) >= coll_d
            min_distance_constraints.append(min_distance_constraint)

    return min_distance_constraints[:0]


def create_avoid_obstacle_constraints(a_dim, coll_d, nx, t, x, obstacles):
    avoid_obstacle_constraints = []
    if coll_d is None or obstacles is None:
        return []

    # TODO: make convex
    num_agents = nx // a_dim
    for idx in range(num_agents):
        for obstacle in obstacles:
            agent_i_center = x[idx * a_dim : (idx + 1) * a_dim, t]
            avoid_obstacle_constraint = cvxpy.norm1(agent_i_center - obstacle['center']) >= coll_d + obstacle['radius']
            # avoid_obstacle_constraint = cvxpy.norm1(agent_i_center - obstacle['center']) >= coll_d + obstacle['radius']
            avoid_obstacle_constraints.append(avoid_obstacle_constraint)

    return avoid_obstacle_constraints
