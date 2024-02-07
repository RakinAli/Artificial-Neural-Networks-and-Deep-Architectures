# 3.1
## Vary number of units to obtain the absolute residual error below 0.1, 0.01, and 0.001
To obtain an error below 0.1 it is enough to use 4 units, one unit placed at each minimum and maximum.

To obtain an error below 0.01 it is enough to use 6 units, one unit placed at each minimum and maximum and one 

## simply transform the output of your RBF network to reduce error to 0 for the square (2x) problem
By using the output activation sgn(y) which is 1 when y > 0 and -1 when y < 0, we can simply transform the output of the rbf network to obtain error 0. We still need 4 units, one for each minimum and maximum of the sine wave.

# 3.2
