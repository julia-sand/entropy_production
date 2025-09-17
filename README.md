# entropy_production

Accompanying code for [Minimal work protocols for inertial particles in nonharmonic traps](https://doi.org/10.1103/PhysRevE.111.034127)

A user guide of the procedure can be found in the Minimal Work Protocols notebook. This is an overview of the steps. 

To get better results, use the scripts as follows:

To compute the overdamped problem (an optimal transport problem)
```
run python entropy_production/sinkhorn.py --args**
```

For all parsable parameters, info can be found using help option. 

Outputs: the csv containing the results of the overdamped calculation and a text file containing the metaparameters of the run

To compute the corrections for the underdamped distributions and drift from the overdamped solution, use

```
run python entropy_production/distributionanddrift.py --args**
```

The overdamped data is fetched in the datafetch.py script based on the input file name. The new underdamped columns are appended to the "results" csv. This behaviour can be customised by changing the output file for at the end of the distributionanddrift.py file. 

Plots can be made by running the following scripts 
- For the cumulants use cumulantsplot.py
- For drift and distribution: distributionanddrift.py
- For plots of the joint distribution: first girsanovjoint.py (to compute and save the values); plot with girsanovplot.py
- Sample histograms of the final distribution: histograms.py
- Momentum variance at different g: momvarplot.py
- Entropy Production at different times totalcosts.py (to compute and save values); plot with totalcostplot.py


