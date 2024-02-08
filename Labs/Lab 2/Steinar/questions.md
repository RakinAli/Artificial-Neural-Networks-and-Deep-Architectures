# 3.1
## Vary number of units to obtain the absolute residual error below 0.1, 0.01, and 0.001
To obtain an error below 0.1 it is enough to use 4 units, one unit placed at each minimum and maximum.

To obtain an error below 0.01 it is enough to use 6 units, one unit placed at each minimum and maximum and one 

## simply transform the output of your RBF network to reduce error to 0 for the square (2x) problem
By using the output activation sgn(y) which is 1 when y > 0 and -1 when y < 0, we can simply transform the output of the rbf network to obtain error 0. We still need 4 units, one for each minimum and maximum of the sine wave.

# 3.2

## Compare the effect of the number of RBF units and their width for the two learning approaches
Gives similar results residual error for model 1: 0.1074 residual error for model 2: 0.0154 better results when using batch learning with sigma np.pi / 4 and 7 nodes.
When using sigma = 2 the residual error is  0.0739 for model 1 and 0.0278 for model 2

## What can you say about the rate of convergence and its dependence on the learning rate, eta for the on-line learning scheme?
The speed of convergence depends on the learning rate, higher learning rate => higher speed of convergance. If the learning rate is too high the gradients might explode and then the algorithm will not converge.

## How important is the positioning of the RBF nodes in the input space?, what strategy did you choose? Is it better than random positioning of the RBF nodes? Please support your conclusions with quantitative evidence (e.g.) error comparison.
When weights are shifted 1 to the right the error is not too bad for the normal method but the sequential method gets worse. ARE for model 1: 0.5811 ARE for model 2: 0.06867

With random weights the ARE is 0.3864 for model 1 and 0.08187 for model 2
Seems to work worse for model 2 with random weights 

## Please compare your optimal RBF network trained in batch mode witha single-hidden-layer perceptron trained with backprop (also in batchmode), which you implemented in the first lab assignment. Please use the same number of hidden units as in the RBF network. The comparison should be made for both functions: sin(2x) and square(2x), only for the noisy case. Please remember that generalisation performance and training time are of greatest interest.
The RBF network does a much better job, we need much more hidden neurons to be able to approximate the sin_2x function using the single-hidden-layer-perceptron trained with backprop

# 4.1
# Animals.dat
Similar animals seem to appear at the same place. That is flies and insects appear close to each other and bigger animals appear closer together.