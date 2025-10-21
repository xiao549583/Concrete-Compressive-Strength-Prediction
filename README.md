#Project Name
Research on predicting compressive strength of concrete based on machine learning ensemble model

#License
This project is released under the MIT License.2025. LiRuibin

# Code Documentation for Concrete Strength Prediction Model

## Code Overview
This model is capable of predicting the compressive strength of concrete with high accuracy.
It significantly improves prediction accuracy compared to other single machine learning models.
It is suitable for predicting the compressive strength of concrete using machine learning.

## Operating Environment
The following Python libraries need to be installed (compatible versions recommended):
pip install -r requirements.yml
or   python>=3.8
       pandas>=1.0.0
       numpy>=1.18.0
       matplotlib>=3.3.0
       scikit-learn>=0.23.0
       xgboost>=1.0.0
       lightgbm>=3.0.0
       scipy>=1.5.0
       openpyxl>=3.0.0
       pip
       pip:
   shap>=0.39.0
   bayesian-optimization>=1.2.0

##Version Reference
The code includes version printing for key libraries, which will display when run:
SHAP version
LightGBM version
XGBoost version

##Core Functional Modules

###Data Processing and Feature Engineering
Data Loading: Reads Excel-formatted concrete dataset from a specified path (C:/Users/1/Desktop/Concrete_Data.xls), 
Data Cleaning: Handles spaces in column names and checks for the presence of the target column (Concrete compressive strength).
Feature Engineering:
Derived features: Cement-water ratio (Cement_Water_Ratio), aggregate-to-cement ratio (Aggregate_to_Cement), total binder content (Binder_Content).
Feature selection: Filters important features using mutual information (mutual_info_regression).
Preprocessing: Handles missing/infinite values (imputed with mean) and standardizes features (StandardScaler).

## Model Definition and Training
###Single Models
Includes 5 base regression models, with both simple and Bayesian-optimized configurations:
Simple Configuration Models:
XGBoost (XGBoost_Simple)
LightGBM (LightGBM_Simple)
Support Vector Machine (SVM, RBF kernel)
Random Forest (RandomForest)
Artificial Neural Network (ANN, MLPRegressor with two hidden layers)
Optimized Models:
XGBoost and LightGBM are optimized using Bayesian optimization (BayesianOptimization) to search for optimal hyperparameters, with 5-fold cross-validation R² as the objective.
### Ensemble Models
Based on optimized XGBoost and LightGBM as base models, 5 ensemble strategies are implemented:
Weighted Average Ensemble: Assigns weights based on base models' R² scores.
Stacking Ensemble: Uses MLP as the meta-model to stack base models' predictions.
Super Learner: Generates out-of-fold predictions via cross-validation, then optimizes weights by minimizing MSE.
Residual Boosting Ensemble: Takes the best single model as the base, trains a residual prediction model, and combines results.
Bayesian Model Averaging: Fuses predictions by assigning weights based on all models' performance.
###Model Evaluation and Selection
Evaluation Metrics: Computes R², RMSE, MSE, and MAE for training and test sets across all models.
Optimal Model Selection: Compares test set R² of all single and ensemble models to select the highest-performing model as the final model.
###Visualization and Interpretation
Performance Visualization:
True vs. predicted value scatter plots (with ±10% error lines and bands).
Line charts comparing predictions and true values (limited to first 50 samples for readability).
Diagnostic Plots: Residual distribution histograms, residual Q-Q plots, and residuals vs. predictions scatter plots (via plot_model_diagnostics).
SHAP Analysis (triggered when optimal model R² > 0.8):
Feature importance bar plots (based on mean SHAP values).
Feature dependence plots (showing relationships between feature values and SHAP values).
Beeswarm plots (displaying SHAP value distribution across all samples).
###Result Export
Evaluation Metrics: Exports test set metrics of all models (excluding intermediate optimized models) to model_evaluation_metrics.xlsx.
Hyperparameters: Exports key hyperparameters of all models to all_hyperparameters.xlsx.
Residual Ensemble Configuration: Generates detailed configuration for residual Boosting ensemble (including performance metrics and dataset info) saved as config_final.json.

##Input and Output
Input
Concrete dataset: Concrete_Data.xls (must be placed in the specified path, or modify the excel_file path in the code).
Output Files
Image Files:
Scatter plots (with error lines): scatter_10p_error_<model_name>.png
Prediction comparison line charts: prediction_line_<model_name>.png
SHAP analysis plots: shap_feature_importance_square.png, shap_dependence_square_<feature_name>.png, shap_bee_swarm_square.png
abular Files:
Model evaluation metrics: model_evaluation_metrics.xlsx
Hyperparameter statistics: all_hyperparameters.xlsx
Configuration File: Residual ensemble model configuration config_final.json

##Usage Instructions
Environment Setup: Install the required libraries, ensuring version compatibility.
Data Preparation: Place the concrete dataset Excel file in the specified path, or modify the path in pd.ExcelFile().
Run the Code: Execute the script directly. It will automatically perform data processing, model training, evaluation, visualization, and result export.
View Results: All output files are saved in the script's running directory. Metrics can be viewed in Excel, and model performance analyzed via image files.

##Notes
Path Issues: The default data path is for Windows systems. Modify excel_file path for other systems or different file locations.
Computational Resources: Bayesian optimization and ensemble training may be time-consuming; run on a high-performance device.
SHAP Analysis: Triggered only if the optimal model's R² > 0.8; skipped if model performance is insufficient.
Parameter Adjustment: Key parameters (e.g., train-test split ratio test_size, cross-validation folds cv, Bayesian optimization iterations) can be modified as needed.
The code may have certain requirements for the computer's graphics card. It is recommended to use an NVIDIA GeForce RTX 4060 or higher for better performance.
The code may take a relatively long time to run, so please wait patiently.

## Contact
If you have any questions about the above content, please feel free to contact us via email: 13797426612@163.com.
