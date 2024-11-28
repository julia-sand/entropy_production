# entropy_production

To compute the overdamped problem (an optimal transport problem)
```
srun python entropy_production/sinkhorn.py --epsilon= --T= --hstep=
```

where:

epsilon is the multiscale expansion parameter. 

Tf is the final time in the underdamped problem

hstep is the size of the time mesh

g is the coupling constant in the dynamics 

and more arguments can be found by using the help "-h" option.

Outputs: the csv containing the results of the overdamped calculation and a text file containing the metaparameters of the problem

To compute the corrections for the underdamped distributions and drift, use

srun python3 distributionanddrift.py --args

The overdamped data is fetched in the datafetch.py script based on the input file name. The new underdamped columns are appended to the "results" csv. This behviour can be customised by changing the output file for at the end of the distributionanddrift.py file. Similarly, use plotdrift.py to get the plot of the drifts.



