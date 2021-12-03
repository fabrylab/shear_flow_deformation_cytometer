# Shear Flow Deformation Cytometer

The Shear Flow Deformation Cytometer is a method to measure the visco-elastic properties of living cells in suspension. 
Cells are suspended in a high-viscosity (0.5-5 Pa s) fluid (typically alginate with concentrations 1-2.5%)
and exposed to fluid shear during their passage through large (200 µm diameter) microfluidic channels.

## Installation
The software uses Python, so you need to have a Python installation, e.g. Anaconda.

To install the software package, download the repository from GitHub, open a command line in the downloaded folder (the one containing "setup.py") and execute

  `pip install .`
  
## Usage

To record image data, use the script:

  `recording.py`
  
To evaluate the resulting data (.tif files) use the script:

  `evaluate.py`
  
## Analysis

To analyse the evaluated data, this script example can serve as a starting point.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from deformationcytometer.evaluation.helper_functions import load_all_data_new, get2Dhist_k_alpha_err

# load all the data in the given folder
# you can also add a list of folders instead
# filenames can also include * for wildcard matches
data, config = load_all_data_new(r"\\path_to_folder_containing_the_data")

# print all the columns, these are the ones from the results.csv and from the meta data files
print(data.columns)

# you can now group by some of the columns
# for example just by the filename to get an evaluation per file
for filename, d in data.groupby("filename"):
    # calculate the k and alpha and their bootstrapped errors according to the 2D mode
    k, k_err, alpha, alpha_err = get2Dhist_k_alpha_err(d)
    print(filename, k, k_err, alpha, alpha_err)

# or directly apply it on the grouped dataframe to get a new dataframe
aggregated_data = data.groupby("filename").apply(get2Dhist_k_alpha_err)
print(aggregated_data)
```
