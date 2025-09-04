# NPPCast: Grouped Time-Series Casting for Global Ocean NPP Forecasts

**Paper:** 
**Code:** https://github.com/wubizhi/NPPCast  
**Data (Zenodo):** https://doi.org/10.5281/zenodo.17008458

## Overview
NPPCast reformulates large-scale spatiotemporal forecasting as a set of grouped multivariate time-series problems. After masking non-ocean cells and partitioning the ocean grid into G groups, a TimesNet-based regional forecaster is trained per group, and outputs are fused into global maps. The repo reproduces all tables/figures in the paper.

## Repository Layout
