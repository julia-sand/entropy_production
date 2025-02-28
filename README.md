# entropy_production

To compute the overdamped problem (an optimal transport problem)
```
srun python entropy_production/sinkhorn.py --args**
```

For all parsable parameters, info can be found using help option. 

Outputs: the csv containing the results of the overdamped calculation and a text file containing the metaparameters of the run

To compute the corrections for the underdamped distributions and drift, use

```
srun python entropy_production/distributionanddrift.py --args**
```

The overdamped data is fetched in the datafetch.py script based on the input file name. The new underdamped columns are appended to the "results" csv. This behaviour can be customised by changing the output file for at the end of the distributionanddrift.py file. Similarly, use plotdrift.py to get the plot of the drifts.



