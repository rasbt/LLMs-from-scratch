# Optimizing Hyperparameters for Pretraining

The [hparam_search.py](hparam_search.py) script, based on the extended training function in [Appendix D: Adding Bells and Whistles to the Training Loop](../../appendix-D/01_main-chapter-code/appendix-D.ipynb), is designed to find optimal hyperparameters via grid search.

>[!NOTE]
This script will take a long time to run. You may want to reduce the number of hyperparameter configurations explored in the `HPARAM_GRID` dictionary at the top.