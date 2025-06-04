# Dynamic Model Averaging for Financial Conditions Index (DMA-FCI)

This repository contains MATLAB code to replicate the empirical results from Koop and Korobilis (2014) "A new index of financial conditions," published in *European Economic Review*.

## Paper Information

**Citation**: Koop, G. and Korobilis, D. (2014). "A new index of financial conditions," *European Economic Review*, 71, 101-116.

**Abstract**: This paper develops a new Financial Conditions Index (FCI) using Dynamic Model Averaging (DMA) applied to Time-Varying Parameter Factor-Augmented Vector Autoregressions (TVP-FAVAR). The approach allows for automatic variable selection and model combination in a time-varying framework.

## Repository Structure

```
DMA_FCI/
├── README.md                           # This file
├── data/                              # Data files
│   ├── xdata.dat                      # Financial variables (paper dataset)
│   ├── xdata_all.dat                  # Extended dataset (81 variables)
│   ├── ydata.dat                      # Macroeconomic variables
│   ├── other_FCIs.dat                 # Competing FCIs from Fed banks
│   └── xnames.mat                     # Variable names
├── functions/                         # Utility functions
│   ├── mlag2.m                       # Create VAR lags
│   ├── Minn_prior_KOOP.m             # Minnesota-type prior
│   └── [other utility functions]
├── DEMONSTRATION/
│   └── TVP-FAVAR.m                   # Single TVP-FAVAR estimation
├── FORECASTING/
│   ├── Competing_FCIs.m              # Forecasts from existing FCIs
│   ├── DMA_TVP_FAVAR.m               # DMA/DMS forecasts (main)
│   ├── DMA_TVP_FAVAR_TS.m            # DMA with training sample prior
│   └── FAVAR_PC_DOZ.m                # Homoskedastic FAVAR benchmark
└── FULL-SAMPLE/
    └── DMA_probabilities.m           # Generate DMA probabilities & plots
```

## Code Categories

### 1. Demonstration Code
**`TVP-FAVAR.m`**
- Estimates a single TVP-FAVAR model
- **Purpose**: Educational/demonstration only
- **Use**: Starting point to understand estimation before DMA
- **Options**: Can load either main dataset (20 vars) or extended dataset (81 vars)

### 2. Forecasting Code

**`Competing_FCIs.m`**
- Generates forecasts from 4 existing FCIs collected from Federal Reserve Banks
- **Replicates**: Benchmark comparisons in the paper

**`DMA_TVP_FAVAR.m`** ⭐ **Main Code**
- Dynamic Model Averaging/Selection (DMA/DMS) with TVP-FAVAR
- Uses relatively non-informative prior
- **Replicates**: Main forecasting results

**`DMA_TVP_FAVAR_TS.m`**
- DMA/DMS with training sample prior
- **Purpose**: Sensitivity analysis (Online Appendix only)

**`FAVAR_PC_DOZ.m`**
- Homoskedastic FAVAR with principal components
- Uses Doz et al. (2011) factor estimation
- **Purpose**: Additional benchmark

### 3. Full-Sample Analysis

**`DMA_probabilities.m`**
- Plots time-varying DMA probabilities
- Shows expected number of variables over time
- Generates FCI implied by DMA
- **Replicates**: Figures 4 & 5 in the paper

## Model Specifications

The code implements several model variants controlled by forgetting/decay factors:

| Model | λ₁ | λ₂ | λ₃ | λ₄ | Description |
|-------|----|----|----|----|-------------|
| **TVP-FAVAR** | 0.96 | 0.96 | 0.96 | 0.99 | Time-varying parameters & factors |
| **FA-TVP-VAR** | 0.96 | 0.96 | 1.00 | 0.99 | Fixed factors, time-varying VAR |
| **Heteroskedastic FAVAR** | 0.96 | 0.96 | 1.00 | 1.00 | Fixed parameters, time-varying volatility |
| **Homoskedastic FAVAR** | 1.00 | 1.00 | 1.00 | 1.00 | All parameters fixed |

*Note: In code, forgetting factors are named `l_1`, `l_2`, `l_3`, `l_4` corresponding to κ₁, κ₂, κ₃, κ₄ in the paper.*

## Quick Start Guide

### Requirements
- MATLAB R2014b or later
- Statistics and Machine Learning Toolbox
- **Computational Requirements**: Very high for full DMA (see warnings below)

### Basic Usage

1. **Start with demonstration**:
   ```matlab
   run TVP-FAVAR.m  % Understand single model estimation
   ```

2. **Generate main results**:
   ```matlab
   run DMA_probabilities.m      % Figures 4 & 5
   run DMA_TVP_FAVAR.m         % Main forecasting results
   ```

3. **Extract forecasting metrics**:
   ```matlab
   % Mean MSFE across all variables
   squeeze(mean(MSFE_DMA(1:end-1,:,1),1))'  % h=1 steps ahead
   squeeze(mean(MSFE_DMA(1:end-2,:,2),1))'  % h=2 steps ahead
   squeeze(mean(MSFE_DMA(1:end-3,:,3),1))'  % h=3 steps ahead
   squeeze(mean(MSFE_DMA(1:end-4,:,4),1))'  % h=4 steps ahead
   ```

### User Input Parameters

Each main file contains a **"USER INPUT"** section with key parameters:

```matlab
nlag = 4;           % Number of lags in FAVAR (paper default)
l_1 = 0.96;         % Forgetting factor for VAR coefficients  
l_2 = 0.96;         % Forgetting factor for factor loadings
l_3 = 0.96;         % Forgetting factor for factors
l_4 = 0.99;         % Forgetting factor for volatilities
```

### Controlling DMA Complexity

**`var_no_dma`** parameter controls which variables are always included:

```matlab
var_no_dma = 1;           % Include var 1 (S&P500), DMA over 2^19 models
var_no_dma = [1 3 5];     % Include vars 1,3,5, DMA over 2^17 models  
var_no_dma = 1:20;        % Include all vars, no DMA (single model)
```

## Computational Warnings ⚠️

**Full DMA is computationally intensive!**

- **2^19 = 524,288 models** for complete DMA over 19 variables
- **Requires**: High-performance computing or cluster computing
- **Recommendation**: Start with subset of variables using `var_no_dma`

**Before running full DMA**:
1. Test single model estimation timing
2. Start with smaller model spaces (e.g., `var_no_dma = [1 3 5 9 12 15]`)
3. Consider using MATLAB Parallel Computing Toolbox

## Data Information

### Main Dataset (Paper)
- **`xdata.dat`**: 20 financial variables
- **`ydata.dat`**: Macroeconomic variables  
- **`other_FCIs.dat`**: Competing FCIs from Federal Reserve Banks

### Extended Dataset (Demonstration)
- **`xdata_all.dat`**: 81 financial variables
- **`xnames.mat`**: Variable names and descriptions

*Variable names and ordering available in `xnames.mat`*

## Output and Results

### Forecasting Results
- **MSFE arrays**: `MSFE_DMA`, `MSFE_competing`, etc.
- **Manual processing**: Results saved in arrays for manual Excel analysis
- **No automatic LaTeX tables**: Reduces programming complexity/errors

### Figures
- **Figure 4**: DMA probabilities over time
- **Figure 5**: Expected number of variables and FCI evolution
- Generated by `DMA_probabilities.m`

## Method Overview

**Dynamic Model Averaging (DMA)** automatically:
- Selects relevant financial variables over time
- Combines forecasts from multiple TVP-FAVAR models
- Accounts for model uncertainty and structural breaks
- Generates time-varying Financial Conditions Index

**Key Innovation**: Combines the flexibility of TVP-FAVAR with the robustness of model averaging for improved forecasting and FCI construction.

## Technical Notes

- **Minnesota-type prior**: Implemented in `Minn_prior_KOOP.m`
- **Forgetting factors**: Control different sources of time variation
- **Kalman filtering**: Used for state estimation in TVP framework
- **Recursive forecasting**: Out-of-sample evaluation methodology

## Citation

If you use this code in your research, please cite:

```
Koop, G. and Korobilis, D. (2014). A new index of financial conditions. 
European Economic Review, 71, 101-116.
```

## Important Disclaimers

⚠️ **This code is not ideal for complete novices**

✅ **Suitable for**: PhD students and researchers with MATLAB experience who study the paper carefully

❌ **No technical support provided**

⚠️ **Computational intensity**: Full DMA requires substantial computing resources

## License

This code is provided for academic research purposes. Please cite the original paper when using this code.

## Contact

**Dimitris Korobilis**  
University of Glasgow  
Email: dimitris.korobilis@glasgow.ac.uk

---

*Last updated: [Current Date]  
For questions about methodology, please refer to the original paper.*
