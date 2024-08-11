def f(x):
    """
    Define the objective function to be minimized.
    Example: f(x) = x^2 + 4x + 4
    """
    return x**2 + 4*x + 4

# test comment

def gradient(x, h=1e-5):
    """
    Compute the first derivative (gradient) using finite differences.
    
    Parameters:
    x (float): Point at which the gradient is calculated.
    h (float): Step size for finite difference approximation.
    
    Returns:
    float: Approximate gradient at x.
    """
    return (f(x + h) - f(x - h)) / (2 * h)

def hessian(x, h=1e-5):
    """
    Compute the second derivative (Hessian) using finite differences.
    
    Parameters:
    x (float): Point at which the Hessian is calculated.
    h (float): Step size for finite difference approximation.
    
    Returns:
    float: Approximate Hessian at x.
    """
    grad_at_x = gradient(x, h)
    grad_at_x_plus_h = gradient(x + h, h)
    return (grad_at_x_plus_h - grad_at_x) / h

def newton_method(f, x0, tol=1e-6, max_iter=100):
    """
    Perform Newton's method to find the minimum of the function f.
    
    Parameters:
    f (function): The function to minimize.
    x0 (float): Initial guess for the minimizer.
    tol (float): Tolerance for convergence.
    max_iter (int): Maximum number of iterations.
    
    Returns:
    float: Approximate minimizer of the function f.
    """
    x = x0
    for i in range(max_iter):
        grad = gradient(x)
        hess = hessian(x)
        
        if hess == 0:  # Prevent division by zero
            raise ValueError("Hessian is zero, can't perform Newton's step.")
        
        # Update step
        x_new = x - grad / hess
        
        # Check convergence
        if abs(x_new - x) < tol:
            print(f"Converged after {i+1} iterations.")
            return x_new
        
        x = x_new
    
    raise ValueError("Newton's method did not converge.")

# Example usage
initial_guess = 0.0
minimum = newton_method(f, initial_guess)
print(f"The minimum is approximately at x = {minimum}")
