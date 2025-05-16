# Publication-related Materials

## US-RSE 2025

* https://us-rse.org/usrse25/

### [Notebook](https://us-rse.org/usrse25/participate/#notebooks) (Conference)

* [`notebook`](./usrse_2025/notebook/)

#### Purpose

Demonstating Synthetic Population Ensembles for Small-Area Estimation

#### Abstract

Synthetic populations are realistic virtual representations of people and households that enable fuller understanding of the social composition of localities to support tasks like planning and civil engineering. To produce cross-sectional measures of individual or residential characteristics at high spatial resolution (e.g., sub-populations that could experience greater transportation or energy-related affordability challenges than others), synthetic population data can be aggregated into small-area estimates (SAEs).    

Synthetic population ensembles consisting of multiple plausible synthetic populations for an area of interest extend the SAE process to uncertainty quantification, providing estimates bounded by confidence intervals. The  Likeness Python stack offers capabilities for producing SAEs via the `livelike` package, which provides a comprehensive and reproducible methodology for generating synthetic populations based on the American Community Survey (ACS).  
 
This notebook demonstrates the application of `livelike` to produce SAEs related to individual and household energy affordability within the Knoxville, Tennessee metropolitan area based on the ACS 2019 – 2023 5-Year Estimates. We follow a procedure for specifying and solving population synthesis models, inspecting intermediary results, and generating and visualizing the SAEs. We explore this process through two scenarios at the household (energy affordability) and person (transportation affordability) levels. Given the wealth of demographic, economic, housing, and mobility subjects covered by the ACS, this approach may be modified to generate synthetic populations and SAEs tailored to many other research areas in human dynamics. 

#### Instructions

  * **Confirmed* functional on Linux & macOS*
  * **Potentially* functional on Windows*

1. Ensure a working version of [Miniforge](https://github.com/conda-forge/miniforge) is installed.
2. Create environment from [`py312_livelike_usrse_2025_env.yaml`](./usrse_2025/notebook/py312_livelike_usrse_2025_env.yaml).
   ```
   conda env create --file usrse_2025/notebook/py312_livelike_usrse_2025_env.yaml
   ```
3. Activate environment.
   ```
   conda activate py312_livelike_usrse_2025_env
   ```
4. Launch Jupyterlab
   ```
   jupyter lab
   ```
5. Open and run notebook.
