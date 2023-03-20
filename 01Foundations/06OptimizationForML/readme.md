Оновчлол нь бүх машин сургалтын алгоритмуудын гол цөм юм. Бид машин сургалтын загварыг сургахдаа өгөгдсөн өгөгдлийн багцаар оновчлол хийж байна.

# 1. Discover what optimization is 

# 2. Discover the Optimization Algorithms

## Optimization with SciPY

Scalar Optimization: Optimization of a convex single variable function.
Local Search: Optimization of a unimodal multiple variable function.
Global Search: Optimization of a multimodal multiple variable function.
Least Squares: Solve linear and non-linear least squares problems.
Curve Fitting: Fit a curve to a data sample.
Root Finding: Find the root (input that gives an output of zero) of a function.
Linear Programming: Linear optimization subject to constraints.


### Local Search With SciPy

```py
# minimize an objective function
result = minimize(objective, point)
```

The example below demonstrates how to solve a two-dimensional convex function using the L-BFGS-B local search algorithm.

```py
# l-bfgs-b algorithm local optimization of a convex function
from scipy.optimize import minimize
from numpy.random import rand

# objective function
def objective(x):
	return x[0]**2.0 + x[1]**2.0

# define range for input
r_min, r_max = -5.0, 5.0
# define the starting point as a random sample from the domain
pt = r_min + rand(2) * (r_max - r_min)
# perform the l-bfgs-b algorithm search
result = minimize(objective, pt, method='L-BFGS-B')
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```

### Global Search With Scipy

The library also provides the shgo() function for sequence optimization and the brute() for grid search optimization.

Each algorithm returns an OptimizeResult object that summarizes the success or failure of the search and the details of the solution if found.

The example below demonstrates how to solve a two-dimensional multimodal function using simulated annealing.

```py
# simulated annealing global optimization for a multimodal objective function
from scipy.optimize import dual_annealing

# objective function
def objective(v):
	x, y = v
	return (x**2 + y - 11)**2 + (x + y**2 -7)**2

# define range for input
r_min, r_max = -5.0, 5.0
# define the bounds on the search
bounds = [[r_min, r_max], [r_min, r_max]]
# perform the simulated annealing search
result = dual_annealing(objective, bounds)
# summarize the result
print('Status : %s' % result['message'])
print('Total Evaluations: %d' % result['nfev'])
# evaluate solution
solution = result['x']
evaluation = objective(solution)
print('Solution: f(%s) = %.5f' % (solution, evaluation))
```



# 3. Dive the Optimization topics
