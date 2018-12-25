# Optimization
Implementation of steepest descent and conjugate gradient algorithms in numpy
#### Note
These algorithm work only when the quadratic form of the equation is paraboloid and has no saddle points.
## Usage
```python
from optimizers import SteepestDescent, ConjugateDirection
optim_sd = SteepestDescent(A, b)
optim_cd = ConjugateDirection(A, b)
optim_sd.converge(iterations=10)
optim_cd.converge() # Gets solved in exactly n iterations where n=dimension of A
# Visualize convergence (A bit buggy right now and works only for 2 dimensional A)
optim_sd.plot_convergence()
optim_cd.plot_convergence()
```
![](download\ (1).png)
