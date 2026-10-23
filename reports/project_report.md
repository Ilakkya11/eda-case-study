# EDA Case Study Report

Student: Ilakkya

Roll number and submission date: Fill in before submission.

## Abstract
This case study applies exploratory data analysis, visualization, preprocessing, and a multilayer perceptron (MLP) to the Wisconsin Breast Cancer Diagnostic dataset. It contains 569 observations and 30 numeric predictors. The workflow checks data quality, examines distributions and correlations, uses a stratified train/validation/test split, and standardizes predictors using training data only. Three MLP configurations are compared using validation F1-score. The selected model achieved 95.35% test accuracy and 0.990 ROC-AUC in the recorded run. This is an educational demonstration and must not be used for clinical diagnosis.

## Dataset
Source: UCI Machine Learning Repository, Wisconsin Breast Cancer Diagnostic dataset: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic

There are 569 rows and 30 numeric predictors. Target mapping: 0 = malignant, 1 = benign. Class counts: 357 benign and 212 malignant.

## EDA and preprocessing
There were 0 missing cells and 0 duplicate rows. The IQR rule flagged 608 feature cells across 29 features; these values were retained rather than automatically removed because outliers may be valid observations. All predictors are numeric, so categorical encoding was not required. Stratified split sizes were 398 training, 85 validation, and 86 test samples. StandardScaler was fitted on training data only.

Mean radius averages 17.46 for malignant samples and 12.15 for benign samples. Mean area averages 978.38 for malignant samples and 462.79 for benign samples. Mean radius and mean perimeter have Pearson correlation 0.998.

## Seven stages of visualization
The assignment requires seven stages but does not specify a framework. This report uses Ben Fry's framework: Acquire, Parse, Filter, Mine, Represent, Refine, Interact. The script generates six EDA visuals: radius distribution, area boxplot, class distribution, radius-perimeter scatterplot, correlation heatmap, and selected-feature pairplot. Five additional figures cover training/validation loss, training/validation accuracy, confusion matrix, ROC curve, and permutation feature importance. The figures are static; the notebook can be rerun and modified.

## Deep learning model and results
A feed-forward MLP neural network was trained using standardized features. Three configurations were compared on validation F1-score. Selected hidden layers: [64, 32], alpha 0.0001, learning rate 0.001; 100 epochs and batch size 32.

| Metric | Value |
|---|---:|
| Test accuracy | 0.9535 |
| Precision (benign) | 0.9464 |
| Recall (benign) | 0.9815 |
| F1-score (benign) | 0.9636 |
| ROC-AUC (benign) | 0.9902 |

Top permutation importance features were worst texture, radius error, worst smoothness, mean concavity, and mean texture. Permutation importance reflects model reliance, not causality.

## Conclusion and future work
The visualizations show differences in size-related measurements by diagnosis and a strong correlation between mean radius and mean perimeter. The model achieved 95.35% accuracy and 0.990 ROC-AUC on one held-out split. The dataset is relatively small and may not represent other populations. Future work includes repeated cross-validation, comparison with other classifiers, calibration, subgroup evaluation, and domain-expert review.

## References
1. Wolberg, W. H., Street, W. N., & Mangasarian, O. L. (1995). Breast cancer Wisconsin (Diagnostic) Data Set. UCI Machine Learning Repository. https://doi.org/10.24432/C5DW2B
2. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825–2830.
3. McKinney, W. (2010). Data structures for statistical computing in Python. Proceedings of the 9th Python in Science Conference, 56–61.
4. Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90–95.
5. Fry, B. (2008). Visualizing Data. O'Reilly Media.

## Running the project
Use Python 3.10+; install dependencies from requirements.txt and run python src/main.py. Educational use only; not a clinical diagnostic system.