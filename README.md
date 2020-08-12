# EntropyDiscretization
Performs binary entropy discretization based on the paper by Fayyad and Irani

### Features
Uses entropy to base a decision on whether or not to create a cut point at which the discretization will occur; performs only binary discretization as outlined in the first portion of the afformentioned paper.

Performs the heuristic provided in the paper by Fayyad and Irani in which only boundary points are checked. This reduces the number of checks performed but should not affect performance. 

Uses numpy functions to ensure operations are quick.

Follows loosely the guidelines for preprocessing classes of sklearn: includes functions: transform, fit and fit_transform so it should work in a pipeline. It does not, however, check whether the inputted data is a numpy array or a pandas DataFrame, 
so you will need to convert the data into a numpy array before processing it through this fit, fit_transform, or transform functions.


