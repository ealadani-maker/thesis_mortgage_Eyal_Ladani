# Predicting Mortgage Default Over Time with Machine Learning
 
Master thesis, Data Science and Society, Tilburg University.
Supervisor: Dr. Brighton.
 
This repository holds the code for my thesis on predicting five-year mortgage default in the Freddie Mac Single-Family Loan-Level Dataset (SFLLD). It compares logistic regression, random forest, XGBoost and LightGBM, and checks how the models behave across borrower risk groups and across origination years from 2000 to 2020.
 
## Research questions
 
| | Question |
|---|---|
| Main RQ | Do machine learning models produce accurate and interpretable mortgage default predictions across borrower-risk segments and over time, using Freddie Mac loan-level data? |
| RQ1 | How do logistic regression, random forest, XGBoost, and LightGBM compare in predicting mortgage default using the Freddie Mac dataset? |
| RQ2 | How do predictive performance and false-negative rates differ across credit score, loan-to-value (LTV), and debt-to-income (DTI), looked at one at a time and in pairs (credit score and LTV, LTV and DTI, credit score and DTI)? |
| RQ3 | How do accuracy and calibration change for loans originated from 2000 to 2020, and do these changes match major events such as the 2008 crisis, the 2014 Ability-to-Repay/Qualified Mortgage rule, and the 2020 pandemic? |
| RQ4 | Which features contribute most to the predictions of the best-performing model, and how do these contributions differ across the credit score, LTV, and DTI segments from SQ2? |
 
## Design in short
 
- **Data:** 22 samples of 50,000 loans each (origination years 1999 to 2020), 1,100,000 loans in total. 1,098,678 remain after removing loans without records and loans whose 60-month window runs past March 2026.
- **Label:** a loan defaults if, within its first 60 months, it becomes 90 or more days past due (or is taken back as REO), or ends through a credit event (zero balance codes 02, 03 or 09). The overall rate is about 4%.
- **Out-of-time evaluation:** for each test year T from 2000 to 2020, the models are developed on loans originated in year T−1 (split 80/20 into training and validation) and tested on all loans originated in year T. That gives 21 test years × 4 models = 84 experiments. The test year is never used for tuning, early stopping, calibration or the cutoff.
- **Preprocessing** (learned on the training data only): median imputation plus a missing flag for numbers, standardisation for logistic regression only, one-hot encoding with a "Missing" category.
- **Imbalance:** class weights (`class_weight="balanced"`, or `scale_pos_weight` for XGBoost). Probabilities are recalibrated with isotonic regression on the validation data, and the cutoff maximises F2 on the validation data.
- **Tuning:** grid search on validation ROC-AUC (5 + 8 + 12 + 24 = 49 fits per test year).
- **Metrics:** ROC-AUC, PR-AUC, Brier score, expected calibration error (10 equal-size bins), recall, precision, false-negative and false-positive rate, plus a 95% bootstrap interval (200 resamples) for ROC-AUC.
- **Tests:** Friedman test and pairwise Wilcoxon signed-rank tests with Holm correction (Demšar, 2006).
- **Explanations:** SHAP (exact for logistic regression, TreeSHAP for XGBoost and LightGBM) on 1,000 test loans per year.
- **Robustness:** other random splits, longer training windows, stricter cutoffs, and logistic regression without class weights.
## Main findings
 
- All four models rank loans almost equally well. Mean test ROC-AUC is 0.806 for logistic regression, 0.804 for LightGBM, 0.803 for XGBoost and 0.799 for random forest. The year-to-year spread (SD about 0.046) is far larger than the gaps between models.
- Random forest overfits most: its train minus validation AUC is 0.085, against 0.006 for logistic regression.
- Ranking is hardest for the riskiest borrowers, for example within-group AUC of 0.67 for credit scores below 660 against 0.77 for scores of 780 and above.
- Calibration lags one year behind: the gap follows the jump in the default rate between the calibration year and the test year (underestimation in 2005, overestimation in 2008 and 2020).
- Ranking power falls after 2014 for the models and for credit score alone, which points to a change in the borrower population and not a model going stale.
- Credit score is the strongest driver in both explained models, followed by LTV, number of borrowers, interest rate and DTI.
The exact numbers and the thesis tables and figures they feed are printed by the notebook. Each section of the notebook names the thesis table or figure it belongs to.
 
## Repository contents
 
```
.
├── Thesis_EDA_Models.ipynb   the whole analysis: data, EDA, models, RQ1 to RQ4, robustness
├── README.md
└── (not in the repository) data/ and figures/, see below
```
 
The notebook is organised in three parts:
 
| Part | Sections |
|---|---|
| 1. Data | 1.1 load files and build the label, 1.2 missing values, first look (EDA), 1.3 file checks, 1.4 outliers, 1.5 correlations, 1.6 leakage, 1.7 the default label, 1.8 the modelling sample |
| 2. Models | 2.1 preprocessing and models, 2.2 metrics, 2.3 risk groups, 2.4 SHAP, 2.5 one experiment (`run()`), 2.6 main run |
| 3. Answers | RQ1, RQ2, RQ3, RQ4 and the robustness checks |
 
## Data
 
The data are **not included**. Freddie Mac asks users to register and accept its terms before downloading, so please get the files yourself:
 
1. Register at the Freddie Mac Single-Family Loan-Level Dataset page and download the sample files (Release 47 was used; the last performance month is March 2026).
2. Keep one ZIP per origination year, named `sample_YYYY.zip` for 1999 to 2020. Each ZIP holds an origination file and a performance file (pipe-separated, no header row).
3. Put them in `data/Zip/`.
## How to run
 
The notebook was written for Google Colab with the data on Google Drive.
 
1. Copy the project folder to Drive so that the paths in the first cell match, or change `BASE_DIR` in the first cell:
```
MyDrive/Colab Notebooks/Thesis/
├── data/
│   ├── Zip/                         sample_1999.zip ... sample_2020.zip
│   └── results/
│       ├── merged_loans_checkpoint.csv       created on the first run
│       └── eda/
│           ├── raw_file_checks.csv           created on the first run
│           └── first_default_month.csv       created on the first run
└── figures/                         created by the notebook (300 dpi PNG files)
```
 
2. Open `Thesis_EDA_Models.ipynb` in Colab and choose **Runtime, Run all**.
3. On the first run, the notebook builds the loan table from the ZIP files. The merged loan table and the two small check files are saved as CSV files, so later runs read them directly and skip the slow step. Everything else stays in memory, apart from the figures.
To run it locally, run the notebook in Jupyter and set `BASE_DIR` to a local folder with the same layout. The Google Drive cell does nothing outside Colab.
 
**Running time:** the main run (84 experiments with tuning) and the robustness checks take several hours. A full run took about 4 hours before the robustness section was extended to nine extra random splits, and takes longer now. The last cell prints the total running time. All random steps use fixed seeds (42 for the main run), so results are repeatable.
 
## Requirements
 
Python 3.13 with these versions (the versions used for the thesis):
 
| Package | Version |
|---|---|
| numpy | 2.1.3 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| xgboost | 3.4.1 |
| lightgbm | 4.6.0 |
| scipy | 1.16.3 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
 
```
pip install numpy==2.1.3 pandas==2.2.3 scikit-learn==1.6.1 xgboost==3.4.1 lightgbm==4.6.0 scipy==1.16.3 matplotlib==3.10.0 seaborn==0.13.2
```
 
## Figures written by the notebook
 
Saved in `figures/` at 300 dpi: `missing_data`, `feature_distributions`, `correlation_map`, `default_percentage_per_year`, `default_timing`, `confusion_matrix_lr`, `calibration_confidence`, `rq3_auc`, `rq3_case_mix` and one `rq4_dti_<model>` file per explained model. The tables in the thesis are printed in the notebook and copied into LaTeX by hand.
 
## Notes and limitations
 
- The sample is 50,000 loans per origination year, not the full Freddie Mac population.
- Neighbouring test years share economic conditions, so the p-values describe how consistent the ranking is across years, not formal proof.
- The year effects in RQ3 are associations in time, not causal effects.
- Random forest is not explained with SHAP; only logistic regression, XGBoost and LightGBM are.
## References
 
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data sets. *Journal of Machine Learning Research, 7*, 1–30.
- Lundberg, S. M., & Lee, S.-I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems, 30*.
- Lundberg, S. M., et al. (2020). From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence, 2*, 56–67.
## Contact
 
Eyal Ladani, Tilburg University
 
