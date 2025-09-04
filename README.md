# NPPCast: Grouped Time-Series Casting for Global Ocean NPP Forecasts

**Paper:** 
**Code:** https://github.com/wubizhi/NPPCast  
**Data (Zenodo):** https://doi.org/10.5281/zenodo.17008458

## Overview
NPPCast reformulates large-scale spatiotemporal forecasting as a set of grouped multivariate time-series problems. After masking non-ocean cells and partitioning the ocean grid into G groups, a TimesNet-based regional forecaster is trained per group, and outputs are fused into global maps. The repo reproduces all tables/figures in the paper.


## How to Run **NPPCast**

### 1) Prerequisites

* **Python 3.9+** (recommended via Anaconda/Miniconda)
* Core libraries: `numpy`, `xarray`, `xskillscore`, `scipy`, `scikit-learn`, `matplotlib`, `cartopy`, `netCDF4`, `h5py`, `pillow`, `tqdm`
* (Optional, for SSIM) `scikit-image`

Example conda setup:

```bash
conda create -n nppcast python=3.10 -y
conda activate nppcast
conda install -c conda-forge numpy xarray xskillscore scipy scikit-learn matplotlib cartopy netcdf4 h5py pillow tqdm -y
pip install scikit-image
```

---

### 2) Data setup

Download the Google Drive bundle **`NPP_running_data`** and place the files into the following subfolders (create them if they don’t exist):

```
<repo-root>/
├─ Common_data/
├─ FOSI2/
├─ NPP_Data/
└─ NPP_MAT/
```

Move the files as follows:

1. **FOSI (forcing)**

   * Put `CESM2_FOSI_NPP_surface.nc` → `FOSI2/`

2. **CESM simulations (GIAF & RCP)**

   * Put `g.e21.GIAF_JRA.TL319_g17.spinup-cycle5.pop.h.photoC_TOT.195801-201812.nc` → `NPP_Data/`

3. **MODIS products (.mat)**

   * Put the following → `NPP_MAT/`

     * `CbPM_month_intp.mat`
     * `EVGPM_month_intp.mat`
     * `SVGPM_month_intp.mat`
     * `NPP_month_mean_intp.mat`
     * `NPP_annual_mean_intp.mat`

4. **Grid & masks**

   * Put `CESM_Parameter.nc` → `Common_data/`

> After this step, your layout should look like:

```
Common_data/CESM_Parameter.nc
FOSI2/CESM2_FOSI_NPP_surface.nc
NPP_Data/g.e21.GIAF_JRA.TL319_g17.spinup-cycle5.pop.h.photoC_TOT.195801-201812.nc
NPP_MAT/CbPM_month_intp.mat
NPP_MAT/EVGPM_month_intp.mat
NPP_MAT/SVGPM_month_intp.mat
NPP_MAT/NPP_month_mean_intp.mat
NPP_MAT/NPP_annual_mean_intp.mat
```

---

### 3) Run the experiments

From the repository root:

```bash
conda activate nppcast
bash run_exp09-14.sh
```

This script will:

* load pretraining data (FOSI / GIAF),
* fine-tune on the MODIS products,
* produce metrics (RMSE, MAE, ACC, NSE, SSIM) and figures under the project’s output directories.

---

### 4) Notes

* If you encounter projection or mapping issues in plotting steps, ensure `cartopy` has access to its shapefiles (or disable coastlines).
* If `.mat` reading is needed in your local scripts, make sure `scipy` is installed (`from scipy.io import loadmat`).
* For reproducibility, set `PYTHONHASHSEED`, random seeds in your training script, and document GPU/CPU details if benchmarking inference time.

