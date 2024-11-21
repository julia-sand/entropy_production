# entropy_production

To compute the overdamped problem (an optimal transport problem)

srun python entropy_production/sinkhorn.py epsilon T h_step g n

where:

epsilon is the multiscale expansion parameter. 

T is the final time in the underdamped problem

h_step is the size of the time mesh

g is the coupling constant in the dynamics 

n is the number of samples in the pointclouds for solving the optimal transport problem

To compute the corrections for the underdamped distributions and drift, use

srun python3 distributionanddrift.py epsilon T h_step g

The overdamped data is fetched in the datafetch.py script. The "results" csv is overwritten to add the underdamped columns. This can be amended with the filename of the output. 



