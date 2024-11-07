======================================================
Python for Data Analysis and Scientific Computing
Final Project
- - - - - - - - - - - - - - - - -
Di Wu
======================================================
Project Overview
----------------
Analyze a diabetes health dataset to uncover insights into how various health conditions contribute to the diagnosis of diabetes.

Requirements
------------
- Python Version: 3.8
- Environment: Jupyter Notebook

Packages and Modules Used
-------------------------
- Standard Libraries:
  - Itertools
  - Collections
  - Textwrap
- Third-Party Libraries:
  - Jupyter
  - NumPy
  - Pandas
  - IPython
  - Matplotlib
  - Seaborn
  - SciPy
  - Statsmodels
  - Scikit-learn

Sequence
--------
1. Download the necessary files to the same directory:
   - `utilities.py`
   - `FinalProject.ipynb`
   - `diabetes_prediction_dataset.csv`
   - `environment.yml` (for Conda)
   - `requirements.txt` (for pip)

2. Setup
  - using Conda:
    - Create conda environment:
      conda env create -f environment.yml
    - Activate:
      conda activate diabetes-analysis
    - Start Jupyter Notebook:
      jupyter notebook
    - Navigate to `FinalProject.ipynb` and run the notebook.
  - using pip:
    - Create a virtual environment:
      python -m venv .venv (for Windows, change the command for other OS)
    - Activate the virtual environment
      .venv\Scripts\activate (for Windows, change the command for other OS)
    - Install the required packages:
      pip install -r requirements.txt
    - Start Jupyter Notebook:
      jupyter notebook
    - Navigate to `FinalProject.ipynb` and run the notebook.

Project Structure
-----------------
- Data Loading and Preprocessing:
  - Initial data inspection.

- Exploratory Data Analysis:
  - Summary statistics and visualizations for numerical and categorical variables.

- Normalization and PCA:
  - Handling missing values and do proper encoding.
  - Normalizing the data and performing Principal Component Analysis (PCA).
  
- Machine Learning Models:
  - Model training and evaluation.


Classes and Functions in `utilities.py`
---------------------------------------
- Classes:
  - ProjectDataFrame:
    - Represents a project-specific DataFrame with methods for data manipulation, feature extraction, and visualization. 
      This class allows users to initialize an instance with a DataFrame or a file path, set and get the underlying data, 
      set the features and target variables, and perform various data analysis and visualization tasks.

  - ProjNormalizedDF:
	  - Inherits from ProjectDataFrame and represents a normalized project DataFrame. 
      It includes additional methods for performing Principal Component Analysis (PCA).

  - ProjPrincipalComponentsDF:
	  - Inherits from ProjectDataFrame and represents a project DataFrame with principal components as features. 
    - This class is used to handle data that has already been transformed through PCA.

  - PCA:
	  - Implements Principal Component Analysis for dimensionality reduction and visualization. 
      It includes methods for performing PCA, retrieving loading scores, and plotting the results.

  - LogisticModel:
	  - Represents a logistic regression model with methods for training, evaluating, 
      and displaying the logistic regression equation. It handles the splitting of data into training and test sets, 
      training the model, and evaluating its performance.
  
- Other top-level functions:
  - Various helper functions supporting the classes above, with a few for both helper and standalone use.
