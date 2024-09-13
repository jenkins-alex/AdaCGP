def wolfe_line_search(objective_fn, grad_fn, update_fn, param, step_init, args=(), max_iter=10, c1=1e-4, c2=0.9, beta=0.1):
    """Wolfe line search algorithm

    Args:
        objective_fn (function): objective function to be minimised
        grad_fn (function): gradient of the objective function
        update_fn (function): function to update parameters
        param (torch.tensor): parameters to be updated
        step_init (float): initial step size
        args (tuple, optional): arguments for objective, gradient and update functions. Defaults to ().
        max_iter (int, optional): max iterations. Defaults to 10.
        c1 (float, optional): wolfe parameter 1. Defaults to 1e-4.
        c2 (float, optional): wolfe parameter 2. Defaults to 0.9.
        beta (float, optional): wolfe parameter 3. Defaults to 0.1.

    Returns:
        float: new step size
    """
    # Set initial parameters for line search
    step = step_init
    f_0 = objective_fn(param, *args)
    G = grad_fn(param, *args)
    p = -G
    f_prime_0 = G.flatten() @ p.flatten()
 
    for iter in range(max_iter):
        # get new values of parameters for given step update
        new_param = update_fn(param, G, step, *args)
 
        # evaluate objective function with new parameters
        f_new = objective_fn(new_param, *args)
 
        # evaluate gradient at new parameters
        new_G = grad_fn(new_param, *args)
        new_f_prime = new_G.flatten() @ p.flatten()
 
        # Armijo-Goldstein condition (sufficient decrease condition)
        if f_new > (f_0 + c1 * step * f_prime_0):
            step *= beta
        else:
            break
 
        # Curvature condition
        if new_f_prime < c2 * f_prime_0:
            step *= beta
        else:
            break
    return step