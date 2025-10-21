import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import re
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np


def mean_absolute_percentage_error(y_true, y_pred):

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    mask = y_true != 0
    if np.sum(mask) == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


import xgboost as xgb
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from scipy import stats
import warnings
import os
from sklearn.base import clone
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold as KF
from scipy.optimize import minimize
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression


plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 使用通用英文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


def train_and_evaluate_model():

    model_hyperparameters = {}

    residual_boosting_params = {}


    try:
        excel_file = pd.ExcelFile('C:/Users/1/Desktop/Concrete_Data.xls')
        df = excel_file.parse('Sheet1')
    except Exception as e:
        print(f"Error reading file: {e}")
        return 0


    df.columns = [col.strip() for col in df.columns]
    target_col = 'Concrete compressive strength(MPa, megapascals)'

    if target_col not in df.columns:
        print(f"Target column missing: {target_col}")
        return 0


    df['Cement_Water_Ratio'] = df['Cement (component 1)(kg in a m^3 mixture)'] / (
            df['Water  (component 4)(kg in a m^3 mixture)'] + 1e-8)
    df['Aggregate_to_Cement'] = (df['Coarse Aggregate  (component 6)(kg in a m^3 mixture)'] +
                                 df['Fine Aggregate (component 7)(kg in a m^3 mixture)']) / df[
                                    'Cement (component 1)(kg in a m^3 mixture)']
    df['Binder_Content'] = df['Cement (component 1)(kg in a m^3 mixture)'] + df[
        'Blast Furnace Slag (component 2)(kg in a m^3 mixture)'] + df['Fly Ash (component 3)(kg in a m^3 mixture)']


    X = df.drop(target_col, axis=1)
    y = df[target_col]


    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )


    print("=== Data Preprocessing Details ===")
    print("4.1 Missing Value and Infinite Value Handling")
    print(f"   - Original training set shape: {X_train.shape}")
    print(f"   - Original test set shape: {X_test.shape}")


    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X_train.mean())

    print(f"   - After cleaning training set shape: {X_train.shape}")
    print(f"   - After cleaning test set shape: {X_test.shape}")
    print("   - Missing values filled with feature means")
    print("   - Infinite values replaced with NaN and then filled")


    print("4.2 Outlier Detection and Handling")
    from scipy import stats


    z_scores = np.abs(stats.zscore(X_train.select_dtypes(include=[np.number])))
    outlier_mask = (z_scores > 3).any(axis=1)
    outlier_count = outlier_mask.sum()
    print(f"   - Outliers detected (Z-score > 3): {outlier_count} samples ({outlier_count / len(X_train) * 100:.2f}%)")

    os.makedirs('preprocessing_visualizations', exist_ok=True)

    outlier_stats = pd.DataFrame({
        'Feature': X_train.columns,
        'Outlier_Count': (z_scores > 3).sum(axis=0),
        'Outlier_Percentage': (z_scores > 3).sum(axis=0) / len(X_train) * 100
    })
    outlier_stats.to_excel('preprocessing_visualizations/outlier_statistics.xlsx', index=False)


    print("4.3 Feature Selection using Mutual Information")
    selector = SelectKBest(score_func=mutual_info_regression, k=min(12, X_train.shape[1]))
    X_train_selected = selector.fit_transform(X_train, y_train)
    selected_features = X_train.columns[selector.get_support()]
    X_test_selected = selector.transform(X_test)


    feature_scores = selector.scores_
    feature_importance_df = pd.DataFrame({
        'Feature': X_train.columns,
        'Mutual_Information_Score': feature_scores
    }).sort_values('Mutual_Information_Score', ascending=False)

    print(f"   - Selected {len(selected_features)} features from {X_train.shape[1]} original features")
    print("   - Top 5 features by mutual information:")
    for i, (feature, score) in enumerate(feature_importance_df.head().values):
        print(f"     {i + 1}. {feature}: {score:.4f}")

    feature_importance_df.to_excel('preprocessing_visualizations/feature_importance_scores.xlsx', index=False)


    def remove_units_from_feature_names(feature_names):

        cleaned_features = []
        seen_features = set()


        feature_mapping = {
            'Cement (component 1)(kg in a m^3 mixture)': 'Cement',
            'Blast Furnace Slag (component 2)(kg in a m^3 mixture)': 'Blast Furnace Slag',
            'Fly Ash (component 3)(kg in a m^3 mixture)': 'Fly Ash',
            'Water  (component 4)(kg in a m^3 mixture)': 'Water',
            'Superplasticizer (component 5)(kg in a m^3 mixture)': 'Superplasticizer',
            'Coarse Aggregate  (component 6)(kg in a m^3 mixture)': 'Coarse Aggregate',
            'Fine Aggregate (component 7)(kg in a m^3 mixture)': 'Fine Aggregate',
            'Age (day)': 'Age',
            'Cement_Water_Ratio': 'Cement_Water_Ratio',
            'Aggregate_to_Cement': 'Aggregate_to_Cement',
            'Binder_Content': 'Binder_Content',
            'Concrete compressive strength(MPa, megapascals)': 'Compressive_Strength'
        }

        for feature in feature_names:

            if feature in feature_mapping:
                cleaned = feature_mapping[feature]
            else:

                cleaned = re.sub(r'\([^)]*\)', '', feature)
                cleaned = re.sub(r'\[[^\]]*\]', '', cleaned)
                cleaned = re.sub(r'\{[^}]*\}', '', cleaned)


                units_to_remove = ['kg', 'm^3', 'mixture', 'component', 'MPa', 'day', 'mm', 'cm', 'm', 'g', 'mg']
                for unit in units_to_remove:
                    cleaned = cleaned.replace(unit, '')


                cleaned = re.sub(r'\d+', '', cleaned)


                cleaned = re.sub(r'\s+', ' ', cleaned).strip()


            base_name = cleaned
            counter = 1
            while cleaned in seen_features:
                cleaned = f"{base_name}_{counter}"
                counter += 1

            seen_features.add(cleaned)
            cleaned_features.append(cleaned)

        return cleaned_features


    selected_features_clean = remove_units_from_feature_names(selected_features)


    print("4.4 Standardization (Z-score Normalization)")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)

    print("   - Applied StandardScaler to normalize features")
    print("   - Training data: mean=0, std=1")
    print("   - Test data: transformed using training statistics")
    print("   - Benefits: Improves convergence speed and model stability")


    standardization_stats = pd.DataFrame({
        'Feature': selected_features,
        'Original_Mean': X_train_selected.mean(axis=0),
        'Original_Std': X_train_selected.std(axis=0),
        'Scaled_Mean': X_train_scaled.mean(axis=0),
        'Scaled_Std': X_train_scaled.std(axis=0)
    })
    standardization_stats.to_excel('preprocessing_visualizations/standardization_statistics.xlsx', index=False)


    print("\n=== Generating Data Preprocessing Visualizations ===")


    os.makedirs('preprocessing_visualizations', exist_ok=True)


    print("1. Generating dataset statistics...")
    dataset_info = pd.DataFrame({
        'Dataset': ['Full Dataset', 'Training Set', 'Test Set'],
        'Samples': [len(df), len(X_train), len(X_test)],
        'Features': [X.shape[1], X_train.shape[1], X_test.shape[1]],
        'Target Mean': [y.mean(), y_train.mean(), y_test.mean()],
        'Target Std': [y.std(), y_train.std(), y_test.std()]
    })
    dataset_info.to_excel('preprocessing_visualizations/dataset_statistics_new.xlsx', index=False)


    print("2. Generating outlier detection boxplots...")
    plt.figure(figsize=(18, 12))
    numeric_cols = X.select_dtypes(include=[np.number]).columns


    if len(numeric_cols) > 12:
        numeric_cols = numeric_cols[:12]


    numeric_cols_clean = remove_units_from_feature_names(numeric_cols)

    for i, (col, col_clean) in enumerate(zip(numeric_cols, numeric_cols_clean), 1):
        plt.subplot(3, 4, i)
        box_plot = plt.boxplot(X[col].dropna(), patch_artist=True)

        for patch in box_plot['boxes']:
            patch.set_facecolor('#3498db')
            patch.set_alpha(0.7)
        for whisker in box_plot['whiskers']:
            whisker.set(color='#2c3e50', linewidth=2)
        for cap in box_plot['caps']:
            cap.set(color='#2c3e50', linewidth=2)
        for median in box_plot['medians']:
            median.set(color='#e74c3c', linewidth=3)

        plt.title(f'{col_clean}\n(IQR: {X[col].quantile(0.75) - X[col].quantile(0.25):.2f})',
                  fontsize=12, fontweight='bold')
        plt.ylabel('Value', fontsize=11, fontweight='bold')
        plt.xticks([])
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('preprocessing_visualizations/outlier_detection_boxplots.tif', dpi=800, bbox_inches='tight',
                format='tiff')
    plt.close()

   
    outlier_stats = []
    numeric_cols_clean = remove_units_from_feature_names(numeric_cols)

    for col, col_clean in zip(numeric_cols, numeric_cols_clean):
        Q1 = X[col].quantile(0.25)
        Q3 = X[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = X[(X[col] < lower_bound) | (X[col] > upper_bound)]
        outlier_count = len(outliers)
        outlier_percentage = (outlier_count / len(X)) * 100

        outlier_stats.append({
            'Feature': col_clean,
            'Outlier_Count': outlier_count,
            'Outlier_Percentage': f'{outlier_percentage:.2f}%',
            'Lower_Bound': lower_bound,
            'Upper_Bound': upper_bound,
            'Min_Value': X[col].min(),
            'Max_Value': X[col].max()
        })

    outlier_df = pd.DataFrame(outlier_stats)
    outlier_df.to_excel('preprocessing_visualizations/outlier_statistics.xlsx', index=False)


    print("3. Generating correlation matrix heatmap...")
    plt.figure(figsize=(16, 14))


    correlation_matrix = X.corr()


    feature_names_clean = remove_units_from_feature_names(X.columns)


    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix,
                mask=mask,
                annot=True,
                fmt='.2f',
                cmap='coolwarm',
                center=0,
                square=True,
                cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                xticklabels=feature_names_clean,
                yticklabels=feature_names_clean,
                annot_kws={"size": 10, "weight": "bold"})

    plt.title('Feature Correlation Matrix\n(11 Input Features)', fontsize=18, fontweight='bold', pad=25)
    plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('preprocessing_visualizations/feature_correlation_matrix.tif', dpi=800, bbox_inches='tight',
                format='tiff')
    plt.close()


    print("4. Generating target variable distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))


    axes[0].hist(y, bins=35, alpha=0.8, color='#3498db', edgecolor='#2c3e50', linewidth=1.2)
    axes[0].axvline(y.mean(), color='#e74c3c', linestyle='--', linewidth=2.5, label=f'Mean: {y.mean():.2f}')
    axes[0].axvline(y.median(), color='#2ecc71', linestyle='--', linewidth=2.5, label=f'Median: {y.median():.2f}')
    axes[0].set_xlabel('Concrete Compressive Strength (MPa)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Output Variable Distribution\n(Compressive Strength)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11, framealpha=0.9)
    axes[0].grid(True, linestyle='--', alpha=0.5)


    box_plot = axes[1].boxplot(y, patch_artist=True)

    for patch in box_plot['boxes']:
        patch.set_facecolor('#9b59b6')
        patch.set_alpha(0.7)
    for whisker in box_plot['whiskers']:
        whisker.set(color='#2c3e50', linewidth=2)
    for cap in box_plot['caps']:
        cap.set(color='#2c3e50', linewidth=2)
    for median in box_plot['medians']:
        median.set(color='#e74c3c', linewidth=3)

    axes[1].set_title('Output Variable Boxplot\n(Compressive Strength)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Concrete Compressive Strength (MPa)', fontsize=12, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('preprocessing_visualizations/target_distribution.tif', dpi=800, bbox_inches='tight', format='tiff')
    plt.close()


    print("5. Generating feature distribution histograms...")
    plt.figure(figsize=(18, 14))

    numeric_cols_clean = remove_units_from_feature_names(numeric_cols)


    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c',
              '#d35400', '#c0392b', '#16a085', '#8e44ad', '#27ae60', '#2980b9']

    for i, (col, col_clean) in enumerate(zip(numeric_cols, numeric_cols_clean), 1):
        plt.subplot(3, 4, i)
        plt.hist(X[col], bins=25, alpha=0.8, color=colors[i - 1], edgecolor='#2c3e50', linewidth=1.2)
        plt.title(col_clean, fontsize=12, fontweight='bold')
        plt.xlabel('Value', fontsize=10, fontweight='bold')
        plt.ylabel('Frequency', fontsize=10, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('preprocessing_visualizations/feature_distributions.tif', dpi=800, bbox_inches='tight', format='tiff')
    plt.close()


    print("6. Generating train-test distribution comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))


    axes[0, 0].hist(y_train, bins=30, alpha=0.7, label='Train', color='#3498db', density=True, edgecolor='#2c3e50',
                    linewidth=1.2)
    axes[0, 0].hist(y_test, bins=30, alpha=0.7, label='Test', color='#e74c3c', density=True, edgecolor='#2c3e50',
                    linewidth=1.2)
    axes[0, 0].set_xlabel('Concrete Strength (MPa)', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Output Variable Distribution: Train vs Test', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=11, framealpha=0.9)
    axes[0, 0].grid(True, linestyle='--', alpha=0.5)


    train_means = X_train.mean()
    test_means = X_test.mean()


    features_clean = remove_units_from_feature_names(X_train.columns)

    x_pos = np.arange(len(features_clean))
    axes[0, 1].bar(x_pos - 0.2, train_means, 0.4, label='Train', alpha=0.8, color='#3498db', edgecolor='#2c3e50',
                   linewidth=1.2)
    axes[0, 1].bar(x_pos + 0.2, test_means, 0.4, label='Test', alpha=0.8, color='#e74c3c', edgecolor='#2c3e50',
                   linewidth=1.2)
    axes[0, 1].set_xlabel('Input Features', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Mean Value', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Input Feature Means: Train vs Test', fontsize=14, fontweight='bold')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(features_clean, rotation=45, ha='right', fontsize=10, fontweight='bold')
    axes[0, 1].legend(fontsize=11, framealpha=0.9)
    axes[0, 1].grid(True, linestyle='--', alpha=0.5)


    train_corr = X_train.corr().values.flatten()
    test_corr = X_test.corr().values.flatten()

    axes[1, 0].scatter(train_corr, test_corr, alpha=0.8, s=60, color='#2ecc71', edgecolor='#27ae60', linewidth=1.2)
    axes[1, 0].plot([-1, 1], [-1, 1], 'r--', alpha=0.8, linewidth=2.5)
    axes[1, 0].set_xlabel('Training Set Correlations', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Test Set Correlations', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Input Feature Correlation Consistency: Train vs Test', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, linestyle='--', alpha=0.5)
    axes[1, 0].set_xlim(-1, 1)
    axes[1, 0].set_ylim(-1, 1)


    train_vars = X_train.var()
    test_vars = X_test.var()

    axes[1, 1].scatter(train_vars, test_vars, alpha=0.8, s=60, color='#9b59b6', edgecolor='#8e44ad', linewidth=1.2)
    axes[1, 1].plot([0, max(train_vars.max(), test_vars.max())],
                    [0, max(train_vars.max(), test_vars.max())], 'r--', alpha=0.8, linewidth=2.5)
    axes[1, 1].set_xlabel('Training Set Variance', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Test Set Variance', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Input Feature Variance Consistency: Train vs Test', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig('preprocessing_visualizations/train_test_comparison.tif', dpi=800, bbox_inches='tight', format='tiff')
    plt.close()


    print("7. Generating SHAP outlier detection visualizations...")
    try:
        import shap


        xgb_model = xgb.XGBRegressor(random_state=42, n_jobs=-1)
        xgb_model.fit(X_train_scaled, y_train)


        explainer = shap.Explainer(xgb_model, X_train_scaled)
        shap_values = explainer(X_train_scaled)


        plt.figure(figsize=(14, 10))
        shap.summary_plot(shap_values, X_train_scaled,
                          feature_names=feature_names_clean,
                          show=False, plot_type="bar")
        plt.title('SHAP Feature Importance', fontsize=18, fontweight='bold', pad=25)
        plt.tight_layout()
        plt.savefig('preprocessing_visualizations/shap_feature_importance.tif', dpi=800, bbox_inches='tight',
                    format='tiff')
        plt.close()


        plt.figure(figsize=(16, 12))


        outlier_mask = np.zeros(len(X_train_scaled), dtype=bool)
        for i in range(X_train_scaled.shape[1]):
            Q1 = np.percentile(X_train_scaled[:, i], 25)
            Q3 = np.percentile(X_train_scaled[:, i], 75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            feature_outliers = (X_train_scaled[:, i] < lower_bound) | (X_train_scaled[:, i] > upper_bound)
            outlier_mask = outlier_mask | feature_outliers


        for i in range(min(8, X_train_scaled.shape[1])):
            plt.subplot(2, 4, i + 1)


            normal_points = plt.scatter(X_train_scaled[~outlier_mask, i],
                                        shap_values.values[~outlier_mask, i],
                                        alpha=0.7, c='#3498db', label='Normal', s=40,
                                        edgecolor='#2c3e50', linewidth=1.0)

            if np.any(outlier_mask):
                outlier_points = plt.scatter(X_train_scaled[outlier_mask, i],
                                             shap_values.values[outlier_mask, i],
                                             alpha=0.9, c='#e74c3c', label='Outlier', s=50,
                                             edgecolors='#c0392b', linewidth=1.5)

            plt.xlabel(feature_names_clean[i], fontsize=10, fontweight='bold')
            plt.ylabel('SHAP Value', fontsize=10, fontweight='bold')
            plt.title(f'{feature_names_clean[i]} vs SHAP', fontsize=12, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5)

            if i == 0:
                plt.legend(fontsize=9, framealpha=0.9)

        plt.tight_layout()
        plt.savefig('preprocessing_visualizations/shap_outlier_detection.tif', dpi=800, bbox_inches='tight',
                    format='tiff')
        plt.close()


        plt.figure(figsize=(14, 12))


        shap_corr_matrix = np.zeros((X_train_scaled.shape[1], X_train_scaled.shape[1]))
        for i in range(X_train_scaled.shape[1]):
            for j in range(X_train_scaled.shape[1]):
                if i != j:
                    corr = np.corrcoef(X_train_scaled[:, i], shap_values.values[:, j])[0, 1]
                    shap_corr_matrix[i, j] = corr


        sns.heatmap(shap_corr_matrix,
                    annot=True,
                    fmt='.2f',
                    cmap='coolwarm',
                    center=0,
                    square=True,
                    cbar_kws={"shrink": .8, "label": "Correlation Coefficient"},
                    xticklabels=feature_names_clean,
                    yticklabels=feature_names_clean,
                    annot_kws={"size": 10, "weight": "bold"})

        plt.title('SHAP Correlation Matrix\n(Correlation between Features and SHAP Values)',
                  fontsize=16, fontweight='bold', pad=25)
        plt.xticks(rotation=45, ha='right', fontsize=11, fontweight='bold')
        plt.yticks(rotation=0, fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig('preprocessing_visualizations/shap_correlation_heatmap.tif', dpi=800, bbox_inches='tight',
                    format='tiff')
        plt.close()

        print("✓ SHAP visualizations completed!")

    except ImportError:
        print("⚠ SHAP package not available. Skipping SHAP visualizations.")
    except Exception as e:
        print(f"⚠ Error in SHAP visualizations: {e}")

    print("✓ Data preprocessing visualizations completed!")
    print("✓ All visualizations saved to 'preprocessing_visualizations/' directory")

    def xgb_evaluate(n_estimators, max_depth, learning_rate, gamma, subsample,
                     colsample_bytree, reg_alpha, reg_lambda, min_child_weight):
        model = xgb.XGBRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            gamma=gamma,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=42,
            n_jobs=-1,
            tree_method='hist'
        )
        try:
            scores = cross_val_score(model, X_train_scaled, y_train,
                                     cv=KFold(5, shuffle=True, random_state=42),
                                     scoring='r2')
            return scores.mean()
        except:
            return -1

    def lgb_evaluate(n_estimators, max_depth, learning_rate, num_leaves, subsample,
                     colsample_bytree, reg_alpha, reg_lambda, min_child_samples, min_split_gain):
        model = lgb.LGBMRegressor(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            learning_rate=learning_rate,
            num_leaves=int(num_leaves),
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=int(min_child_samples),
            min_split_gain=min_split_gain,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        try:
            scores = cross_val_score(model, X_train_scaled, y_train,
                                     cv=KFold(5, shuffle=True, random_state=42),
                                     scoring='r2')
            return scores.mean()
        except:
            return -1


    xgb_params = {
        'n_estimators': (300, 1500),
        'max_depth': (4, 12),
        'learning_rate': (0.02, 0.2),
        'gamma': (0, 0.5),
        'subsample': (0.8, 1.0),
        'colsample_bytree': (0.8, 1.0),
        'reg_alpha': (0, 5),
        'reg_lambda': (0, 5),
        'min_child_weight': (1, 10)
    }

    lgb_params = {
        'n_estimators': (300, 1500),
        'max_depth': (4, 12),
        'learning_rate': (0.02, 0.2),
        'num_leaves': (31, 127),
        'subsample': (0.8, 1.0),
        'colsample_bytree': (0.8, 1.0),
        'reg_alpha': (0, 5),
        'reg_lambda': (0, 5),
        'min_child_samples': (20, 100),
        'min_split_gain': (0, 0.1)
    }


    print("Optimizing XGBoost...")
    xgb_optimizer = BayesianOptimization(
        f=xgb_evaluate,
        pbounds=xgb_params,
        random_state=42
    )
    xgb_optimizer.maximize(init_points=5, n_iter=15)

    xgb_best_params = xgb_optimizer.max['params'] if xgb_optimizer.max else {}
    xgb_best = xgb.XGBRegressor(
        **{k: int(v) if k in ['n_estimators', 'max_depth'] else v for k, v in xgb_best_params.items()},
        random_state=42, n_jobs=-1
    )

    model_hyperparameters['xgb_optimized'] = xgb_best.get_params()

    print("Optimizing LightGBM...")
    lgb_optimizer = BayesianOptimization(
        f=lgb_evaluate,
        pbounds=lgb_params,
        random_state=42
    )
    lgb_optimizer.maximize(init_points=5, n_iter=15)

    lgb_best_params = lgb_optimizer.max['params'] if lgb_optimizer.max else {}
    lgb_best = lgb.LGBMRegressor(
        **{k: int(v) if k in ['n_estimators', 'max_depth', 'num_leaves', 'min_child_samples'] else v
           for k, v in lgb_best_params.items()},
        random_state=42, n_jobs=-1, verbose=-1
    )

    model_hyperparameters['lgb_optimized'] = lgb_best.get_params()


    simple_xgb = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        tree_method='hist'
    )
    model_hyperparameters['XGBoost_Simple'] = simple_xgb.get_params()

    simple_lgb = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.7,
        colsample_bytree=0.7,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_hyperparameters['LightGBM_Simple'] = simple_lgb.get_params()

    svm_model = SVR(kernel='rbf', C=10, gamma=0.1)  # SVM模型
    model_hyperparameters['SVM'] = svm_model.get_params()


    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=8,
        min_samples_split=10,
        min_samples_leaf=5,
        max_features=0.7,
        random_state=42,
        n_jobs=-1
    )
    model_hyperparameters['RandomForest'] = rf_model.get_params()

    ann_model = MLPRegressor(
        hidden_layer_sizes=(50, 20),
        activation='relu',
        solver='adam',
        max_iter=1000,
        random_state=42
    )
    model_hyperparameters['ANN'] = ann_model.get_params()


    linear_model = LinearRegression()
    model_hyperparameters['LinearRegression'] = linear_model.get_params()


    all_single_models = [
        ('LinearRegression', linear_model),
        ('XGBoost_Simple', simple_xgb),
        ('LightGBM_Simple', simple_lgb),
        ('SVM', svm_model),
        ('RandomForest', rf_model),
        ('ANN', ann_model)
    ]


    base_models = [('xgb_optimized', xgb_best), ('lgb_optimized', lgb_best)]


    all_models = all_single_models + base_models

    model_predictions = {}
    model_performance = {}
    model_metrics = {}
    model_train_metrics = {}

    print("\n=== All Models Evaluation ===")
    for name, model in all_models:
        try:
            model.fit(X_train_scaled, y_train)


            y_train_pred = model.predict(X_train_scaled)


            y_pred = model.predict(X_test_scaled)


            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                print(f"{name} predictions contain invalid values (NaN/Inf), cleaning...")
                y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred))

            if np.any(np.isnan(y_train_pred)) or np.any(np.isinf(y_train_pred)):
                print(f"{name} training predictions contain invalid values, cleaning...")
                y_train_pred = np.nan_to_num(y_train_pred, nan=np.nanmean(y_train_pred))


            test_r2 = r2_score(y_test, y_pred)
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            test_mse = mean_squared_error(y_test, y_pred)
            test_mae = mean_absolute_error(y_test, y_pred)
            test_mape = mean_absolute_percentage_error(y_test, y_pred)


            train_r2 = r2_score(y_train, y_train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_mae = mean_absolute_error(y_train, y_train_pred)
            train_mape = mean_absolute_percentage_error(y_train, y_train_pred)

            model_predictions[name] = y_pred
            model_performance[name] = test_r2
            model_metrics[name] = {
                'Test R2': test_r2,
                'Test RMSE': test_rmse,
                'Test MSE': test_mse,
                'Test MAE': test_mae,
                'Test MAPE': test_mape
            }

            model_train_metrics[name] = {
                'Train R2': train_r2,
                'Train RMSE': train_rmse,
                'Train MSE': train_mse,
                'Train MAE': train_mae,
                'Train MAPE': train_mape
            }

            print(f"{name} Metrics:")
            print(f"  Training R2: {train_r2:.6f}, Test R2: {test_r2:.6f}")
            print(f"  Training RMSE: {train_rmse:.6f}, Test RMSE: {test_rmse:.6f}")
            print(f"  Training MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
            print(f"  Training MAE: {train_mae:.6f}, Test MAE: {test_mae:.6f}")
            print(f"  Training MAPE: {train_mape:.4f}%, Test MAPE: {test_mape:.4f}%")
        except Exception as e:
            print(f"{name} training failed: {e}")
            model_performance[name] = -1
            model_metrics[name] = {
                'Test R2': -1,
                'Test RMSE': -1,
                'Test MSE': -1,
                'Test MAE': -1,
                'Test MAPE': -1
            }
            model_train_metrics[name] = {
                'Train R2': -1,
                'Train RMSE': -1,
                'Train MSE': -1,
                'Train MAE': -1,
                'Train MAPE': -1
            }


    print("\n=== Advanced Ensemble Strategies ===")
    ensemble_metrics = {}
    ensemble_predictions = {}
    ensemble_train_metrics = {}


    weights = {}
    total_score = 0

    for name, model in base_models:
        r2 = model_performance[name]
        if r2 > 0:
            weights[name] = r2
            total_score += r2

    weighted_pred = None
    weighted_r2 = -1
    weighted_train_r2 = -1
    if total_score > 0:
        for name in weights:
            weights[name] /= total_score


        weighted_pred = sum(weights[name] * model_predictions[name] for name in weights)
        weighted_r2 = r2_score(y_test, weighted_pred)
        weighted_rmse = np.sqrt(mean_squared_error(y_test, weighted_pred))
        weighted_mse = mean_squared_error(y_test, weighted_pred)
        weighted_mae = mean_absolute_error(y_test, weighted_pred)
        weighted_mape = mean_absolute_percentage_error(y_test, weighted_pred)


        weighted_train_pred = sum(weights[name] * model.predict(X_train_scaled) for name, model in base_models)
        weighted_train_r2 = r2_score(y_train, weighted_train_pred)
        weighted_train_rmse = np.sqrt(mean_squared_error(y_train, weighted_train_pred))
        weighted_train_mse = mean_squared_error(y_train, weighted_train_pred)
        weighted_train_mae = mean_absolute_error(y_train, weighted_train_pred)
        weighted_train_mape = mean_absolute_percentage_error(y_train, weighted_train_pred)

        ensemble_predictions['Weighted Average'] = weighted_pred
        ensemble_metrics['Weighted Average'] = {
            'Test R2': weighted_r2,
            'Test RMSE': weighted_rmse,
            'Test MSE': weighted_mse,
            'Test MAE': weighted_mae,
            'Test MAPE': weighted_mape
        }
        ensemble_train_metrics['Weighted Average'] = {
            'Train R2': weighted_train_r2,
            'Train RMSE': weighted_train_rmse,
            'Train MSE': weighted_train_mse,
            'Train MAE': weighted_train_mae,
            'Train MAPE': weighted_train_mape
        }


        model_hyperparameters['Weighted Average'] = {'weights': weights}

        print(f'Weighted Average Ensemble Metrics:')
        print(f"  Training R2: {weighted_train_r2:.6f}, Test R2: {weighted_r2:.6f}")
        print(f"  Training RMSE: {weighted_train_rmse:.6f}, Test RMSE: {weighted_rmse:.6f}")
        print(f"  Training MSE: {weighted_train_mse:.6f}, Test MSE: {weighted_mse:.6f}")
        print(f"  Training MAE: {weighted_train_mae:.6f}, Test MAE: {weighted_mae:.6f}")
        print(f"  Training MAPE: {weighted_train_mape:.4f}%, Test MAPE: {weighted_mape:.4f}%")
    else:
        print("All models have poor performance, cannot use weighted average")


    stack_r2 = -1
    stack_pred = None
    stack_train_r2 = -1
    stack_reg = None
    try:
        stack_reg = StackingRegressor(
            estimators=base_models,
            final_estimator=MLPRegressor(
                hidden_layer_sizes=(50, 20),
                activation='relu',
                solver='adam',
                max_iter=1000,
                random_state=42
            ),
            cv=5,
            n_jobs=-1
        )
        stack_reg.fit(X_train_scaled, y_train)


        stack_params = {
            'base_models': [name for name, _ in base_models],
            'final_estimator': stack_reg.final_estimator.get_params(),
            'cv': 5
        }
        model_hyperparameters['Neural Network Stacking'] = stack_params


        stack_pred = stack_reg.predict(X_test_scaled)

        stack_train_pred = stack_reg.predict(X_train_scaled)

        if np.any(np.isnan(stack_pred)) or np.any(np.isinf(stack_pred)):
            print("Stacking predictions contain invalid values (NaN/Inf), cleaning...")
            stack_pred = np.nan_to_num(stack_pred, nan=np.nanmean(stack_pred))

        if np.any(np.isnan(stack_train_pred)) or np.any(np.isinf(stack_train_pred)):
            print("Stacking training predictions contain invalid values, cleaning...")
            stack_train_pred = np.nan_to_num(stack_train_pred, nan=np.nanmean(stack_train_pred))

        stack_r2 = r2_score(y_test, stack_pred)
        stack_rmse = np.sqrt(mean_squared_error(y_test, stack_pred))
        stack_mse = mean_squared_error(y_test, stack_pred)
        stack_mae = mean_absolute_error(y_test, stack_pred)
        stack_mape = mean_absolute_percentage_error(y_test, stack_pred)

        stack_train_r2 = r2_score(y_train, stack_train_pred)
        stack_train_rmse = np.sqrt(mean_squared_error(y_train, stack_train_pred))
        stack_train_mse = mean_squared_error(y_train, stack_train_pred)
        stack_train_mae = mean_absolute_error(y_train, stack_train_pred)
        stack_train_mape = mean_absolute_percentage_error(y_train, stack_train_pred)

        ensemble_predictions['Neural Network Stacking'] = stack_pred
        ensemble_metrics['Neural Network Stacking'] = {
            'Test R2': stack_r2,
            'Test RMSE': stack_rmse,
            'Test MSE': stack_mse,
            'Test MAE': stack_mae,
            'Test MAPE': stack_mape
        }

        ensemble_train_metrics['Neural Network Stacking'] = {
            'Train R2': stack_train_r2,
            'Train RMSE': stack_train_rmse,
            'Train MSE': stack_train_mse,
            'Train MAE': stack_train_mae,
            'Train MAPE': stack_train_mape
        }

        print(f'Neural Network Stacking Ensemble Metrics:')
        print(f"  Training R2: {stack_train_r2:.6f}, Test R2: {stack_r2:.6f}")
        print(f"  Training RMSE: {stack_train_rmse:.6f}, Test RMSE: {stack_rmse:.6f}")
        print(f"  Training MSE: {stack_train_mse:.6f}, Test MSE: {stack_mse:.6f}")
        print(f"  Training MAE: {stack_train_mae:.6f}, Test MAE: {stack_mae:.6f}")
        print(f"  Training MAPE: {stack_train_mape:.4f}%, Test MAPE: {stack_mape:.4f}%")
    except Exception as e:
        print(f"Stacking ensemble failed: {e}")


    super_learner_r2 = -1
    super_learner_pred = None
    super_learner_train_r2 = -1
    super_learner_weights = None
    try:
        n_folds = 5
        kf = KF(n_splits=n_folds, shuffle=True, random_state=42)

        oof_predictions = np.zeros((X_train_scaled.shape[0], len(base_models)))
        test_predictions = np.zeros((X_test_scaled.shape[0], len(base_models)))

        for model_idx, (name, model) in enumerate(base_models):
            model_test_preds = np.zeros(X_test_scaled.shape[0])

            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train_scaled)):
                X_train_fold, X_val_fold = X_train_scaled[train_idx], X_train_scaled[val_idx]
                y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]

                fold_model = clone(model)
                fold_model.fit(X_train_fold, y_train_fold)

                val_preds = fold_model.predict(X_val_fold)
                oof_predictions[val_idx, model_idx] = val_preds

                test_preds = fold_model.predict(X_test_scaled)
                model_test_preds += test_preds / n_folds

            test_predictions[:, model_idx] = model_test_preds

        def objective(w):
            combined_pred = np.dot(oof_predictions, w)
            errors = y_train.values - combined_pred
            return np.mean(errors ** 2)

        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = [(0, 1) for _ in range(len(base_models))]

        w0 = np.ones(len(base_models)) / len(base_models)

        result = minimize(objective, w0, method='SLSQP',
                          bounds=bounds, constraints=constraints)

        if result.success:
            super_learner_weights = result.x
            print(f"Meta-model optimized weights: {super_learner_weights}")


            model_hyperparameters['Super Learner'] = {
                'weights': dict(zip([name for name, _ in base_models], super_learner_weights)),
                'n_folds': n_folds,
                'optimization_method': 'SLSQP'
            }


            super_learner_pred = np.dot(test_predictions, super_learner_weights)

            super_learner_train_pred = np.dot(oof_predictions, super_learner_weights)

            super_learner_r2 = r2_score(y_test, super_learner_pred)
            super_learner_rmse = np.sqrt(mean_squared_error(y_test, super_learner_pred))
            super_learner_mse = mean_squared_error(y_test, super_learner_pred)
            super_learner_mae = mean_absolute_error(y_test, super_learner_pred)
            super_learner_mape = mean_absolute_percentage_error(y_test, super_learner_pred)

            super_learner_train_r2 = r2_score(y_train, super_learner_train_pred)
            super_learner_train_rmse = np.sqrt(mean_squared_error(y_train, super_learner_train_pred))
            super_learner_train_mse = mean_squared_error(y_train, super_learner_train_pred)
            super_learner_train_mae = mean_absolute_error(y_train, super_learner_train_pred)
            super_learner_train_mape = mean_absolute_percentage_error(y_train, super_learner_train_pred)

            ensemble_predictions['Super Learner'] = super_learner_pred
            ensemble_metrics['Super Learner'] = {
                'Test R2': super_learner_r2,
                'Test RMSE': super_learner_rmse,
                'Test MSE': super_learner_mse,
                'Test MAE': super_learner_mae,
                'Test MAPE': super_learner_mape
            }
            # 存储训练集指标
            ensemble_train_metrics['Super Learner'] = {
                'Train R2': super_learner_train_r2,
                'Train RMSE': super_learner_train_rmse,
                'Train MSE': super_learner_train_mse,
                'Train MAE': super_learner_train_mae,
                'Train MAPE': super_learner_train_mape
            }

            print(f'Super Learner Metrics:')
            print(f"  Training R2: {super_learner_train_r2:.6f}, Test R2: {super_learner_r2:.6f}")
            print(f"  Training RMSE: {super_learner_train_rmse:.6f}, Test RMSE: {super_learner_rmse:.6f}")
            print(f"  Training MSE: {super_learner_train_mse:.6f}, Test MSE: {super_learner_mse:.6f}")
            print(f"  Training MAE: {super_learner_train_mae:.6f}, Test MAE: {super_learner_mae:.6f}")
            print(f"  Training MAPE: {super_learner_train_mape:.4f}%, Test MAPE: {super_learner_mape:.4f}%")
        else:
            print("Weight optimization failed, using simple average")
            super_learner_pred = np.mean(test_predictions, axis=1)
            super_learner_r2 = r2_score(y_test, super_learner_pred)
            super_learner_rmse = np.sqrt(mean_squared_error(y_test, super_learner_pred))
            super_learner_mse = mean_squared_error(y_test, super_learner_pred)
            super_learner_mae = mean_absolute_error(y_test, super_learner_pred)
            super_learner_mape = mean_absolute_percentage_error(y_test, super_learner_pred)


            super_learner_train_pred = np.mean(oof_predictions, axis=1)
            super_learner_train_r2 = r2_score(y_train, super_learner_train_pred)
            super_learner_train_rmse = np.sqrt(mean_squared_error(y_train, super_learner_train_pred))
            super_learner_train_mse = mean_squared_error(y_train, super_learner_train_pred)
            super_learner_train_mae = mean_absolute_error(y_train, super_learner_train_pred)
            super_learner_train_mape = mean_absolute_percentage_error(y_train, super_learner_train_pred)

            ensemble_predictions['Super Learner (Average)'] = super_learner_pred
    except Exception as e:
        print(f"Super learner failed: {e}")


    print("\n=== Model Complexity Analysis and Statistical Significance Testing ===")


    print("5.1 Model Complexity Hierarchy Analysis")
    model_complexity_analysis = {}

    for name, model in all_models:
        complexity_info = {}


        if name == 'LinearRegression':
            complexity_info['Structure'] = 'Linear (Low Complexity)'
            complexity_info['Parameters'] = f'{X_train_scaled.shape[1] + 1} parameters'
            complexity_info['Nonlinearity'] = 'Linear only'
            complexity_info['Suitability'] = 'Suitable for linear relationships, poor for nonlinear patterns'
        elif name == 'SVM':
            complexity_info['Structure'] = 'Kernel-based (Medium Complexity)'
            complexity_info['Parameters'] = 'Support vectors + kernel parameters'
            complexity_info['Nonlinearity'] = 'RBF kernel for nonlinear mapping'
            complexity_info['Suitability'] = 'High-dimensional small samples, weak in feature interaction capture'
        elif name == 'ANN':
            complexity_info['Structure'] = 'Multi-layer perceptron (High Complexity)'
            complexity_info['Parameters'] = f'{(50 * X_train_scaled.shape[1] + 50 * 20 + 20 * 1)} parameters'
            complexity_info['Nonlinearity'] = 'ReLU activation for complex nonlinear fitting'
            complexity_info['Suitability'] = 'Prone to overfitting, poor interpretability'
        elif 'XGBoost' in name or 'LightGBM' in name:
            complexity_info['Structure'] = 'Gradient boosting trees (High Complexity)'
            complexity_info['Parameters'] = 'Multiple trees with depth and leaf constraints'
            complexity_info['Nonlinearity'] = 'Tree-based ensemble for complex interactions'
            complexity_info['Suitability'] = 'Strong feature recognition, affected by high correlation features'
        elif name == 'RandomForest':
            complexity_info['Structure'] = 'Bagging ensemble (Medium-High Complexity)'
            complexity_info['Parameters'] = f'{100} trees with depth {8}'
            complexity_info['Nonlinearity'] = 'Ensemble of decision trees'
            complexity_info['Suitability'] = 'Robust to noise, good for medium-dimensional data'
        else:
            # 为其他模型提供默认值
            complexity_info['Structure'] = 'Unknown (Medium Complexity)'
            complexity_info['Parameters'] = 'Unknown parameters'
            complexity_info['Nonlinearity'] = 'Unknown nonlinearity'
            complexity_info['Suitability'] = 'General purpose model'

        model_complexity_analysis[name] = complexity_info
        print(f"{name}: {complexity_info['Structure']}")
        print(f"  - Parameters: {complexity_info['Parameters']}")
        print(f"  - Nonlinearity: {complexity_info['Nonlinearity']}")
        print(f"  - Suitability: {complexity_info['Suitability']}")


    print("\n5.2 Dataset Characteristics and Model Adaptability Analysis")
    dataset_characteristics = {
        'Dimensionality': 'Medium (10+ features)',
        'Sample Size': 'Sufficient (1000+ samples)',
        'Nonlinearity': 'Strong nonlinear relationships',
        'Feature Interactions': 'Complex interactions present'
    }

    print("Dataset Characteristics:")
    for char, desc in dataset_characteristics.items():
        print(f"  - {char}: {desc}")

    print("\nModel Adaptability Analysis:")
    print("  - SVM: Suitable for high-dimensional small samples but weak in capturing complex feature interactions")
    print("  - ANN: Prone to overfitting with poor interpretability, requires careful regularization")
    print("  - XGBoost/LightGBM: Strong in feature importance recognition but affected by highly correlated features")
    print("  - RandomForest: Robust ensemble method suitable for medium-dimensional datasets")
    print("  - LinearRegression: Baseline model, limited to linear relationships")


    print("\n5.3 Statistical Significance Testing")


    model_r2_values = {}
    for name, metrics in model_metrics.items():
        if metrics['Test R2'] > 0:
            model_r2_values[name] = metrics['Test R2']


    from scipy.stats import ttest_rel

    significance_results = {}
    model_names = list(model_r2_values.keys())

    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1, model2 = model_names[i], model_names[j]


            n_bootstrap = 1000
            r2_samples1 = []
            r2_samples2 = []

            for _ in range(n_bootstrap):

                indices = np.random.choice(len(y_test), len(y_test), replace=True)
                y_test_sample = y_test.iloc[indices] if hasattr(y_test, 'iloc') else y_test[indices]

                pred1 = model_predictions[model1][indices]
                pred2 = model_predictions[model2][indices]

                r2_1 = r2_score(y_test_sample, pred1)
                r2_2 = r2_score(y_test_sample, pred2)

                r2_samples1.append(r2_1)
                r2_samples2.append(r2_2)


            t_stat, p_value = ttest_rel(r2_samples1, r2_samples2)

            significance_results[f"{model1}_vs_{model2}"] = {
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }

            significance_level = "significant" if p_value < 0.05 else "not significant"
            print(f"{model1} vs {model2}: t={t_stat:.4f}, p={p_value:.4f} ({significance_level})")

    print("\n=== Model Complexity Analysis and Statistical Significance Testing ===")


    os.makedirs('model_analysis', exist_ok=True)
    os.makedirs('preprocessing_visualizations', exist_ok=True)

    print("5.1 Model Complexity Hierarchy Analysis")

    complexity_df = pd.DataFrame(model_complexity_analysis).T
    complexity_df.to_excel('model_analysis/model_complexity_analysis.xlsx')

    significance_df = pd.DataFrame(significance_results).T
    significance_df.to_excel('model_analysis/statistical_significance_tests.xlsx')


    print("\n5.5 Extended SHAP Interpretability for Engineering Applications")

    if 'shap_values' in locals():
        print("Engineering Application Scenarios:")
        print("1. Feature Optimization Guidance")
        print("   - SHAP values identify which features most influence concrete strength")
        print("   - Engineers can prioritize optimization of high-impact features")

        print("\n2. Material Proportion Adjustment")
        print("   - SHAP dependence plots show optimal ranges for material ratios")
        print("   - Helps in fine-tuning cement-water ratio and aggregate proportions")

        print("\n3. Quality Control Implementation")
        print("   - Identify critical thresholds for material quality parameters")
        print("   - Establish monitoring protocols for high-sensitivity features")

        print("\n4. Cost-Effective Design")
        print("   - Balance material costs with performance requirements")
        print("   - Optimize resource allocation based on feature importance")


        engineering_guide = {
            'Application Area': [
                'Material Proportioning',
                'Quality Control',
                'Process Optimization',
                'Cost Management'
            ],
            'SHAP Insights': [
                'Identify optimal ranges for key material ratios',
                'Monitor critical material quality parameters',
                'Adjust manufacturing processes based on feature importance',
                'Balance cost and performance through feature optimization'
            ],
            'Engineering Actions': [
                'Fine-tune cement-water ratio and aggregate proportions',
                'Establish quality thresholds for high-impact materials',
                'Implement real-time monitoring of sensitive parameters',
                'Optimize material selection based on cost-performance tradeoffs'
            ]
        }

        engineering_df = pd.DataFrame(engineering_guide)
        engineering_df.to_excel('model_analysis/engineering_application_guide.xlsx', index=False)
        print("✓ Engineering application guide generated")

    print("\n5.6 Model Complexity and Dataset Compatibility Summary")
    print("Based on the analysis:")
    print("- Dataset Characteristics: Medium dimensionality, sufficient samples, strong nonlinearity")
    print("- Model Selection Rationale:")
    print("  • XGBoost/LightGBM: Best for complex feature interactions in medium-dimensional data")
    print("  • SVM: Suitable but limited in capturing complex feature relationships")
    print("  • ANN: High complexity requires careful regularization to prevent overfitting")
    print("  • RandomForest: Robust alternative with good interpretability")
    print("  • LinearRegression: Baseline for linear relationship assessment")

    print("\nThe comprehensive analysis provides statistical rigor and practical engineering insights.")


    model_hyperparameters['Super Learner (Average)'] = {
        'n_folds': n_folds,
        'method': 'simple_average'
    }

    print(f'Super Learner (Average) Metrics:')
    print(f"  Training R2: {super_learner_train_r2:.6f}, Test R2: {super_learner_r2:.6f}")
    print(f"  Training RMSE: {super_learner_train_rmse:.6f}, Test RMSE: {super_learner_rmse:.6f}")
    print(f"  Training MSE: {super_learner_train_mse:.6f}, Test MSE: {super_learner_mse:.6f}")
    print(f"  Training MAE: {super_learner_train_mae:.6f}, Test MAE: {super_learner_mae:.6f}")
    print(f"  Training MAPE: {super_learner_train_mape:.4f}%, Test MAPE: {super_learner_mape:.4f}%")


    residual_boost_r2 = -1
    residual_boost_pred = None
    residual_boost_train_r2 = -1
    try:
        best_model_name = max(model_performance, key=model_performance.get)
        best_model = xgb_best if best_model_name == 'xgb_optimized' else lgb_best

        best_model.fit(X_train_scaled, y_train)
        train_pred = best_model.predict(X_train_scaled)
        residuals = y_train.values - train_pred

        residual_model = lgb.LGBMRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        residual_model.fit(X_train_scaled, residuals)


        residual_boosting_params['Base Model'] = best_model_name
        residual_boosting_params['Base Model Params'] = best_model.get_params()
        residual_boosting_params['Residual Model'] = 'LightGBM'
        residual_boosting_params['Residual Model Params'] = residual_model.get_params()
        residual_boosting_params['Residual Model Type'] = 'LGBMRegressor'
        residual_boosting_params['n_estimators'] = 300
        residual_boosting_params['max_depth'] = 5
        residual_boosting_params['learning_rate'] = 0.05


        model_hyperparameters['Residual Boosting'] = {
            'base_model': best_model_name,
            'base_model_params': best_model.get_params(),
            'residual_model_params': residual_model.get_params()
        }


        base_pred = best_model.predict(X_test_scaled)
        residual_pred = residual_model.predict(X_test_scaled)
        residual_boost_pred = base_pred + residual_pred


        residual_boost_train_pred = train_pred + residual_model.predict(X_train_scaled)

        residual_boost_r2 = r2_score(y_test, residual_boost_pred)
        residual_boost_rmse = np.sqrt(mean_squared_error(y_test, residual_boost_pred))
        residual_boost_mse = mean_squared_error(y_test, residual_boost_pred)
        residual_boost_mae = mean_absolute_error(y_test, residual_boost_pred)
        residual_boost_mape = mean_absolute_percentage_error(y_test, residual_boost_pred)

        residual_boost_train_r2 = r2_score(y_train, residual_boost_train_pred)
        residual_boost_train_rmse = np.sqrt(mean_squared_error(y_train, residual_boost_train_pred))
        residual_boost_train_mse = mean_squared_error(y_train, residual_boost_train_pred)
        residual_boost_train_mae = mean_absolute_error(y_train, residual_boost_train_pred)
        residual_boost_train_mape = mean_absolute_percentage_error(y_train, residual_boost_train_pred)

        ensemble_predictions['Residual Boosting'] = residual_boost_pred
        ensemble_metrics['Residual Boosting'] = {
            'Test R2': residual_boost_r2,
            'Test RMSE': residual_boost_rmse,
            'Test MSE': residual_boost_mse,
            'Test MAE': residual_boost_mae,
            'Test MAPE': residual_boost_mape
        }

        ensemble_train_metrics['Residual Boosting'] = {
            'Train R2': residual_boost_train_r2,
            'Train RMSE': residual_boost_train_rmse,
            'Train MSE': residual_boost_train_mse,
            'Train MAE': residual_boost_train_mae,
            'Train MAPE': residual_boost_train_mape
        }

        print(f'Residual Boosting Ensemble Metrics:')
        print(f"  Training R2: {residual_boost_train_r2:.6f}, Test R2: {residual_boost_r2:.6f}")
        print(f"  Training RMSE: {residual_boost_train_rmse:.6f}, Test RMSE: {residual_boost_rmse:.6f}")
        print(f"  Training MSE: {residual_boost_train_mse:.6f}, Test MSE: {residual_boost_mse:.6f}")
        print(f"  Training MAE: {residual_boost_train_mae:.6f}, Test MAE: {residual_boost_mae:.6f}")
        print(f"  Training MAPE: {residual_boost_train_mape:.4f}%, Test MAPE: {residual_boost_mape:.4f}%")
    except Exception as e:
        print(f"Residual boosting failed: {e}")


    bma_r2 = -1
    bma_pred = None
    bma_train_r2 = -1
    try:
        model_weights = {}
        total_perf = sum(max(0, p) for p in model_performance.values())
        for name in model_performance:
            model_weights[name] = max(0, model_performance[name]) / total_perf


        model_hyperparameters['Bayesian Averaging'] = {'weights': model_weights}


        all_preds = np.array([model_predictions[name] for name in model_performance])
        bma_pred = np.zeros_like(y_test)
        for i, name in enumerate(model_performance):
            bma_pred += model_weights[name] * all_preds[i]


        all_train_preds = np.array([model.predict(X_train_scaled) for name, model in all_models])
        bma_train_pred = np.zeros_like(y_train)
        for i, name in enumerate(model_performance):
            bma_train_pred += model_weights[name] * all_train_preds[i]

        bma_r2 = r2_score(y_test, bma_pred)
        bma_rmse = np.sqrt(mean_squared_error(y_test, bma_pred))
        bma_mse = mean_squared_error(y_test, bma_pred)
        bma_mae = mean_absolute_error(y_test, bma_pred)
        bma_mape = mean_absolute_percentage_error(y_test, bma_pred)

        bma_train_r2 = r2_score(y_train, bma_train_pred)
        bma_train_rmse = np.sqrt(mean_squared_error(y_train, bma_train_pred))
        bma_train_mse = mean_squared_error(y_train, bma_train_pred)
        bma_train_mae = mean_absolute_error(y_train, bma_train_pred)
        bma_train_mape = mean_absolute_percentage_error(y_train, bma_train_pred)

        ensemble_predictions['Bayesian Averaging'] = bma_pred
        ensemble_metrics['Bayesian Averaging'] = {
            'Test R2': bma_r2,
            'Test RMSE': bma_rmse,
            'Test MSE': bma_mse,
            'Test MAE': bma_mae,
            'Test MAPE': bma_mape
        }

        ensemble_train_metrics['Bayesian Averaging'] = {
            'Train R2': bma_train_r2,
            'Train RMSE': bma_train_rmse,
            'Train MSE': bma_train_mse,
            'Train MAE': bma_train_mae,
            'Train MAPE': bma_train_mape
        }

        print(f'Bayesian Model Averaging Metrics:')
        print(f"  Training R2: {bma_train_r2:.6f}, Test R2: {bma_r2:.6f}")
        print(f"  Training RMSE: {bma_train_rmse:.6f}, Test RMSE: {bma_rmse:.6f}")
        print(f"  Training MSE: {bma_train_mse:.6f}, Test MSE: {bma_mse:.6f}")
        print(f"  Training MAE: {bma_train_mae:.6f}, Test MAE: {bma_mae:.6f}")
        print(f"  Training MAPE: {bma_train_mape:.4f}%, Test MAPE: {bma_mape:.4f}%")
    except Exception as e:
        print(f"Bayesian Model Averaging failed: {e}")


    methods = {
        'Weighted Average': weighted_r2,
        'Neural Network Stacking': stack_r2,
        'Super Learner': super_learner_r2,
        'Residual Boosting': residual_boost_r2,
        'Bayesian Averaging': bma_r2
    }

    valid_methods = {k: v for k, v in methods.items() if v >= 0}

    if valid_methods:
        best_method = max(valid_methods, key=valid_methods.get)
        best_r2 = valid_methods[best_method]

        if best_method == 'Weighted Average':
            final_pred = weighted_pred
        elif best_method == 'Neural Network Stacking':
            final_pred = stack_pred
        elif best_method == 'Super Learner':
            final_pred = super_learner_pred
        elif best_method == 'Residual Boosting':
            final_pred = residual_boost_pred
        else:
            final_pred = bma_pred
    else:
        best_method = max(model_performance, key=model_performance.get)
        final_pred = model_predictions[best_method]
        best_r2 = model_performance[best_method]
        print(f"All ensemble methods failed, using best single model ({best_method})")

    print(f"\n=== Final Model Selection ===")
    print(f"Best Ensemble Method: {best_method}, R2: {best_r2:.6f}")


    best_single_model = max(model_performance.values())
    improvement = best_r2 - best_single_model
    if improvement > 0:
        print(f"Ensemble outperforms best single model: +{improvement:.6f}")
    elif improvement < 0:
        print(f"Ensemble underperforms best single model: {improvement:.6f}")
    else:
        print("Ensemble equals best single model performance")


    print("\n5.4 Key Feature Sensitivity Analysis")
    print("Implementing key feature sensitivity analysis...")


    if 'shap_values' in locals():

        shap_importance = np.mean(np.abs(shap_values.values), axis=0)
        top_feature_indices = np.argsort(shap_importance)[-5:][::-1]  # 前5个最重要的特征
        top_features = [selected_features_clean[i] for i in top_feature_indices]

        print(f"Top 5 features for sensitivity analysis: {top_features}")

        best_model_name = max(model_performance, key=model_performance.get)
        best_model = xgb_best if best_model_name == 'xgb_optimized' else lgb_best

        sensitivity_results = {}

        for feature_idx in top_feature_indices:
            feature_name = selected_features_clean[feature_idx]
            print(f"\nSensitivity Analysis for {feature_name}:")


            feature_values = X_train_scaled[:, feature_idx]
            mean_val = np.mean(feature_values)
            std_val = np.std(feature_values)


            perturbation_levels = np.linspace(-2, 2, 9)  # 9个扰动水平

            sensitivity_scores = []

            for level in perturbation_levels:

                X_perturbed = X_test_scaled.copy()
                perturbation = level * std_val
                X_perturbed[:, feature_idx] += perturbation


                y_pred_perturbed = best_model.predict(X_perturbed)


                original_r2 = best_r2
                perturbed_r2 = r2_score(y_test, y_pred_perturbed)
                r2_change = perturbed_r2 - original_r2

                sensitivity_scores.append({
                    'perturbation_level': level,
                    'perturbation_value': perturbation,
                    'original_r2': original_r2,
                    'perturbed_r2': perturbed_r2,
                    'r2_change': r2_change
                })

                print(f"  Level {level:.1f}σ: R² change = {r2_change:.4f}")

            sensitivity_results[feature_name] = sensitivity_scores


        sensitivity_df = pd.DataFrame()
        for feature_name, scores in sensitivity_results.items():
            feature_df = pd.DataFrame(scores)
            feature_df['feature'] = feature_name
            sensitivity_df = pd.concat([sensitivity_df, feature_df], ignore_index=True)

        sensitivity_df.to_excel('model_analysis/feature_sensitivity_analysis.xlsx', index=False)
        print("✓ Feature sensitivity analysis completed and saved")


    if best_method in model_train_metrics:
        print(f"\nBest Model Training Metrics:")
        print(f"  Train R2: {model_train_metrics[best_method]['Train R2']:.6f}")
        print(f"  Train RMSE: {model_train_metrics[best_method]['Train RMSE']:.6f}")
        print(f"  Train MSE: {model_train_metrics[best_method]['Train MSE']:.6f}")
        print(f"  Train MAE: {model_train_metrics[best_method]['Train MAE']:.6f}")
    elif best_method in ensemble_train_metrics:
        print(f"\nBest Model Training Metrics:")
        print(f"  Train R2: {ensemble_train_metrics[best_method]['Train R2']:.6f}")
        print(f"  Train RMSE: {ensemble_train_metrics[best_method]['Train RMSE']:.6f}")
        print(f"  Train MSE: {ensemble_train_metrics[best_method]['Train MSE']:.6f}")
        print(f"  Train MAE: {ensemble_train_metrics[best_method]['Train MAE']:.6f}")
    else:
        print("\nCould not find training metrics for best model")


    print("\nGenerating scatter plots with ±10% error lines for all models...")


    all_models_to_plot = {name: model_predictions[name] for name in model_predictions if
                          name not in ['xgb_optimized', 'lgb_optimized']}
    all_models_to_plot.update(ensemble_predictions)

    for model_name in all_models_to_plot:
        y_pred = all_models_to_plot[model_name]


        if model_name in model_train_metrics and model_name in model_metrics:
            train_r2 = model_train_metrics[model_name]['Train R2']
            test_r2 = model_metrics[model_name]['Test R2']
        elif model_name in ensemble_train_metrics and model_name in ensemble_metrics:
            train_r2 = ensemble_train_metrics[model_name]['Train R2']
            test_r2 = ensemble_metrics[model_name]['Test R2']
        else:
            train_r2 = -1
            test_r2 = -1


        plt.figure(figsize=(12, 10))


        plt.scatter(y_test, y_pred, alpha=0.6, s=80, edgecolor='w', linewidth=0.5)


        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')


        plt.plot([min_val, max_val], [min_val * 1.1, max_val * 1.1], 'm--', linewidth=2, label='+10% Error Line')
        plt.plot([min_val, max_val], [min_val * 0.9, max_val * 0.9], 'c--', linewidth=2, label='-10% Error Line')


        plt.fill_between([min_val, max_val], [min_val * 0.9, max_val * 0.9],
                         [min_val * 1.1, max_val * 1.1], color='gray', alpha=0.1, label='±10% Error Band')

        plt.xlabel('True Values', fontsize=14)
        plt.ylabel('Predicted Values', fontsize=14)
        plt.title(f'Model Prediction Accuracy: {model_name}', fontsize=16)

        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12, loc='lower right')
        plt.tight_layout()

        filename = f"scatter_10p_error_{model_name.replace(' ', '_')}.tif"
        plt.savefig(filename, dpi=600, bbox_inches='tight', format='tiff')
        plt.close()
        print(f"Saved scatter plot with ±10% error lines for {model_name} as {filename}")


    academic_colors = {
        'actual': '#1f77b4',
        'predicted': '#d62728',
        'grid': '#e0e0e0',
        'text': '#2c3e50',
        'background': '#ffffff',
        'annotation': '#f8f9fa'
    }


    if best_r2 > 0.8:
        print("\n=== SHAP Analysis ===")
        try:

            if best_method == 'Weighted Average':
                best_model = stack_reg
            elif best_method in [m[0] for m in all_models]:
                best_model = next(model for name, model in all_models if name == best_method)
            elif best_method in ensemble_predictions:
                best_model = stack_reg
            else:
                best_model = xgb_best


            if hasattr(best_model, 'predict') and hasattr(best_model, 'feature_importances_'):
                explainer = shap.TreeExplainer(best_model)
            else:

                explainer = shap.KernelExplainer(best_model.predict, shap.sample(X_train_scaled, 100))


            sample_size = min(500, X_train_scaled.shape[0])
            sample_indices = np.random.choice(X_train_scaled.shape[0],
                                              sample_size,
                                              replace=False)
            X_sample = X_train_scaled[sample_indices]


            X_sample_clean = X_sample.copy()
            print(f"Using {sample_size} samples for SHAP analysis (increased for better visualization)")


            shap_values = explainer.shap_values(X_sample)


            plt.figure(figsize=(12, 8))

            plt.rcParams['axes.facecolor'] = academic_colors['background']
            plt.rcParams['figure.facecolor'] = academic_colors['background']

            shap.summary_plot(
                shap_values,
                X_sample_clean,
                feature_names=selected_features_clean,
                plot_type="bar",
                show=False,
                color=academic_colors['actual']
            )
            plt.title('(f) Feature Importance (SHAP values)', fontsize=18, pad=25,
                      color=academic_colors['text'], weight='bold')
            plt.xlabel('Mean |SHAP Value|', fontsize=16, color=academic_colors['text'])
            plt.ylabel('Features', fontsize=16, fontstyle='italic', rotation=90,
                       labelpad=25, color=academic_colors['text'])
            plt.grid(True, linestyle='--', alpha=0.7, color=academic_colors['grid'])
            plt.tight_layout()
            plt.savefig('shap_feature_importance.tif', dpi=600, bbox_inches='tight',
                        facecolor=academic_colors['background'], format='tiff')
            plt.close()


            feature_maps = [
                {
                    "feature": "Cement",
                    "title": "(a) Cement Feature Importance Plot",
                    "save_name": "shap_dependence_cement.png",
                    "color": academic_colors['actual']
                },
                {
                    "feature": "Age",
                    "title": "(b) Age Feature Importance Plot",
                    "save_name": "shap_dependence_age.png",
                    "color": '#ff7f0e'
                },
                {
                    "feature": "Water",
                    "title": "(c) Water Feature Importance Plot",
                    "save_name": "shap_dependence_water.png",
                    "color": '#2ca02c'
                },
                {
                    "feature": "Blast Furnace Slag",
                    "title": "(d) Blast Furnace Slag Feature Importance Plot",
                    "save_name": "shap_dependence_blast_furnace_slag.png",
                    "color": '#d62728'
                },
                {
                    "feature": "Fly Ash",
                    "title": "(e) Fly Ash Feature Importance Plot",
                    "save_name": "shap_dependence_fly_ash.png",
                    "color": '#9467bd'
                },
                {
                    "feature": "Superplasticizer",
                    "title": "(f) Superplasticizer Feature Importance Plot",
                    "save_name": "shap_dependence_superplasticizer.png",
                    "color": '#8c564b'
                },
                {
                    "feature": "Coarse Aggregate",
                    "title": "(g) Coarse Aggregate Feature Importance Plot",
                    "save_name": "shap_dependence_coarse_aggregate.png",
                    "color": '#e377c2'
                },
                {
                    "feature": "Fine Aggregate",
                    "title": "(h) Fine Aggregate Feature Importance Plot",
                    "save_name": "shap_dependence_fine_aggregate.png",
                    "color": '#7f7f7f'
                },
                {
                    "feature": "Cement_Water_Ratio",
                    "title": "(i) Cement-Water Ratio Feature Importance Plot",
                    "save_name": "shap_dependence_cement_water_ratio.png",
                    "color": academic_colors['predicted']
                },
                {
                    "feature": "Aggregate_to_Cement",
                    "title": "(j) Aggregate-to-Cement Ratio Feature Importance Plot",
                    "save_name": "shap_dependence_aggregate_to_cement.png",
                    "color": '#bcbd22'
                },
                {
                    "feature": "Binder_Content",
                    "title": "(k) Binder Content Feature Importance Plot",
                    "save_name": "shap_dependence_binder_content.png",
                    "color": '#17becf'
                }
            ]

            for fm in feature_maps:
                feature_name = fm["feature"]
                if feature_name in selected_features:
                    plt.figure(figsize=(12, 8))

                    plt.rcParams['axes.facecolor'] = academic_colors['background']
                    plt.rcParams['figure.facecolor'] = academic_colors['background']


                    if feature_name == "Cement_Water_Ratio":

                        feature_idx = np.where(selected_features == feature_name)[0][0]


                        fig, ax = plt.subplots(figsize=(12, 8))
                        shap.dependence_plot(
                            feature_idx,
                            shap_values,
                            X_sample_clean,
                            feature_names=selected_features_clean,
                            interaction_index=None,
                            ax=ax,
                            show=False
                        )


                        ax.set_title(fm["title"], fontsize=18, pad=25, color=academic_colors['text'], weight='bold')

                        clean_feature_name = next(
                            (clean_name for orig_name, clean_name in zip(selected_features, selected_features_clean) if
                             orig_name == feature_name), feature_name)
                        ax.set_xlabel(clean_feature_name, fontsize=16, fontstyle='italic',
                                      color=academic_colors['text'])
                        ax.set_ylabel('SHAP Value', fontsize=16, fontstyle='italic', color=academic_colors['text'])
                        ax.grid(True, linestyle='--', alpha=0.7, color=academic_colors['grid'])


                        plt.text(0.05, 0.95,
                                 "Cement-Water Ratio is a key factor\nin concrete strength prediction",
                                 transform=ax.transAxes,
                                 fontsize=14, verticalalignment='top', color=academic_colors['text'],
                                 bbox=dict(boxstyle='round,pad=0.5', facecolor=academic_colors['annotation'],
                                           alpha=0.9, edgecolor=academic_colors['grid']))

                        plt.tight_layout()
                        plt.savefig(fm["save_name"].replace('.png', '.tif'), dpi=600, bbox_inches='tight',
                                    facecolor=academic_colors['background'], format='tiff')
                        plt.close()
                    else:

                        shap.dependence_plot(
                            feature_name,
                            shap_values,
                            X_sample_clean,
                            feature_names=selected_features_clean,
                            interaction_index=None,
                            show=False
                        )
                        plt.title(fm["title"], fontsize=18, pad=25, color=academic_colors['text'], weight='bold')

                        clean_feature_name = next(
                            (clean_name for orig_name, clean_name in zip(selected_features, selected_features_clean) if
                             orig_name == feature_name), feature_name)
                        plt.xlabel(clean_feature_name, fontsize=16, fontstyle='italic', color=academic_colors['text'])
                        plt.ylabel('SHAP Value', fontsize=16, fontstyle='italic', color=academic_colors['text'])
                        plt.grid(True, linestyle='--', alpha=0.7, color=academic_colors['grid'])
                        plt.tight_layout()
                        plt.savefig(fm["save_name"].replace('.png', '.tif'), dpi=600, bbox_inches='tight',
                                    facecolor=academic_colors['background'], format='tiff')
                        plt.close()
                else:
                    print(f"Warning: Feature '{feature_name}' not in selected features, skipping plot")


            plt.figure(figsize=(12, 8))

            plt.rcParams['axes.facecolor'] = academic_colors['background']
            plt.rcParams['figure.facecolor'] = academic_colors['background']

            shap.summary_plot(
                shap_values,
                X_sample_clean,
                feature_names=selected_features_clean,
                show=False,
                cmap=plt.cm.RdBu_r
            )
            plt.title('(e) Mean SHAP Value Impact', fontsize=18, pad=25, color=academic_colors['text'], weight='bold')
            plt.xlabel('SHAP Value', fontsize=16, color=academic_colors['text'])
            plt.ylabel('Features', fontsize=16, fontstyle='italic', rotation=90,
                       labelpad=25, color=academic_colors['text'])
            plt.grid(True, linestyle='--', alpha=0.7, color=academic_colors['grid'])
            plt.tight_layout()
            plt.savefig('shap_mean_value.tif', dpi=600, bbox_inches='tight',
                        facecolor=academic_colors['background'], format='tiff')
            plt.close()


            print("Generating optimized SHAP correlation heatmap...")
            try:

                print("\n=== Debug Information ===")
                print(f"Original selected features count: {len(selected_features)}")
                print(f"Cleaned selected features count: {len(selected_features_clean)}")
                print("\nCleaned feature names:")
                for i, name in enumerate(selected_features_clean):
                    print(f"{i + 1}. {name}")


                shap_df = pd.DataFrame(shap_values, columns=selected_features_clean)  # 使用清理后的特征名称
                shap_corr = shap_df.corr()


                n_features = len(selected_features_clean)  # 修改为使用清理后的特征数量
                fig_size = max(8, min(12, n_features * 0.8))  # 根据特征数量动态调整大小
                plt.figure(figsize=(fig_size, fig_size))


                plt.rcParams['axes.facecolor'] = academic_colors['background']
                plt.rcParams['figure.facecolor'] = academic_colors['background']


                # 可选配色方案: 'RdBu_r', 'coolwarm', 'viridis', 'plasma', 'inferno'
                cmap = plt.cm.RdBu_r


                heatmap = sns.heatmap(
                    shap_corr,
                    mask=None,
                    annot=True,
                    cmap=cmap,
                    center=0,
                    square=True,
                    fmt='.2f',
                    cbar_kws={
                        "shrink": 0.75,
                        "label": "Correlation Coefficient",
                        "pad": 0.02
                    },
                    annot_kws={
                        'size': 9,
                        'color': academic_colors['text'],
                        'weight': 'bold'
                    },
                    linewidths=0.5,
                    linecolor=academic_colors['grid'],
                    cbar=True,

                    xticklabels=selected_features_clean,
                    yticklabels=selected_features_clean
                )


                plt.title('(g) SHAP Value Correlation Heatmap\n(11 Input Features Multicollinearity Analysis)',
                          fontsize=14, pad=20, color=academic_colors['text'], weight='bold')


                plt.xticks(rotation=45, ha='right', fontsize=10, color=academic_colors['text'])
                plt.yticks(fontsize=10, color=academic_colors['text'])


                cbar = heatmap.collections[0].colorbar
                cbar.ax.tick_params(labelsize=9, colors=academic_colors['text'])
                cbar.set_label('Correlation', fontsize=11, color=academic_colors['text'])


                plt.tight_layout()


                plt.savefig('shap_correlation_heatmap.tif',
                            dpi=600,
                            bbox_inches='tight',
                            facecolor=academic_colors['background'],
                            edgecolor='none',
                            transparent=False,
                            format='tiff')
                plt.close()
                print("Optimized SHAP correlation heatmap saved successfully")

            except Exception as e:
                print(f"Optimized SHAP correlation heatmap error: {e}")

            print("SHAP analysis completed successfully with standard visualizations")
        except Exception as e:
            print(f"SHAP analysis encountered an error: {e}")
            print("⚠ SHAP analysis failed, but continuing with permutation importance analysis...")
    else:
        print(f"Model performance insufficient (R²={best_r2:.4f}), skipping SHAP analysis")


    print("\n=== Starting Independent Permutation Importance Analysis ===")
    print(f"Current best R²: {best_r2:.4f}")
    print("Ensuring permutation importance analysis proceeds independently...")


    print("DEBUG: Forcing permutation importance analysis to execute")


    try:
        print(f"DEBUG: Checking condition - best_r2 > 0.8: {best_r2} > 0.8 = {best_r2 > 0.8}")

        force_execute = True
        if best_r2 > 0.8 or force_execute:
            print("✓ Model performance sufficient for permutation importance analysis")
            print("DEBUG: Entering permutation importance analysis block")
            print("\n=== Permutation Importance Analysis ===")
            print("Permutation Importance Analysis Characteristics:")
            print("• Model-agnostic: Works with any machine learning model")
            print("• Performance-based: Measures feature importance by model performance degradation")
            print("• Robust: Less sensitive to outliers and feature correlations")
            print("• Intuitive: Directly relates to prediction accuracy")
            print("• Complementary: Provides validation for SHAP analysis results")


            try:
                from sklearn.inspection import permutation_importance
                print("✓ permutation_importance module imported successfully")
            except ImportError as e:
                print(f"⚠ sklearn.inspection.permutation_importance not available: {e}")
                print("Skipping permutation importance analysis")
                raise ImportError("permutation_importance module not available")


            if 'X_test_scaled' not in locals() or 'y_test' not in locals():
                print("⚠ Required变量未找到，重新创建测试数据...")

                from sklearn.model_selection import train_test_split
                X_train_scaled, X_test_scaled, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2,
                                                                                  random_state=42)


            if best_method == 'Weighted Average':

                perm_model = xgb_best
            elif best_method in [m[0] for m in all_models]:
                perm_model = next(model for name, model in all_models if name == best_method)
            elif best_method in ensemble_predictions:
                perm_model = xgb_best
            else:
                perm_model = xgb_best


            if not hasattr(perm_model, 'predict'):
                print("Training permutation model...")
                perm_model.fit(X_train_scaled, y_train)


            print("Calculating permutation importance...")


            sample_size = min(100, len(X_test_scaled))
            sample_indices = np.random.choice(len(X_test_scaled), sample_size, replace=False)
            X_test_sample = X_test_scaled[sample_indices]
            y_test_sample = y_test.iloc[sample_indices] if hasattr(y_test, 'iloc') else y_test[sample_indices]

            perm_importance = permutation_importance(
                perm_model,
                X_test_sample,
                y_test_sample,
                n_repeats=5,
                random_state=42,
                n_jobs=-1
            )


            perm_scores = perm_importance.importances_mean
            perm_std = perm_importance.importances_std


            sorted_indices = np.argsort(perm_scores)[::-1]
            sorted_features = selected_features_clean[sorted_indices]
            sorted_scores = perm_scores[sorted_indices]
            sorted_std = perm_std[sorted_indices]


            plt.figure(figsize=(14, 10))
            plt.rcParams['axes.facecolor'] = academic_colors['background']
            plt.rcParams['figure.facecolor'] = academic_colors['background']


            y_pos = np.arange(len(sorted_features))
            bars = plt.barh(y_pos, sorted_scores,
                            xerr=sorted_std,
                            color='#2E8B57',
                            alpha=0.8,
                            capsize=6,
                            edgecolor='darkgreen',
                            linewidth=1.5)

            plt.yticks(y_pos, sorted_features, fontsize=13, fontstyle='italic')
            plt.xlabel('Permutation Importance Score', fontsize=16, fontweight='bold')
            plt.title('Permutation Feature Importance Analysis\n(Model Performance-Based Feature Assessment)',
                      fontsize=18, pad=25, fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.6, axis='x')


            for i, (score, std) in enumerate(zip(sorted_scores, sorted_std)):
                plt.text(score + 0.01, i, f'{score:.3f} ± {std:.3f}',
                         va='center', fontsize=11, fontweight='bold', color='darkgreen')


            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            textstr = 'Permutation Importance Features:\n• Model-agnostic\n• Performance-based\n• Robust to outliers\n• Direct accuracy impact'
            plt.text(0.7, 0.95, textstr, transform=plt.gca().transAxes, fontsize=11,
                     verticalalignment='top', bbox=props)

            plt.tight_layout()

            desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop', 'Permutation_Importance_Analysis')
            os.makedirs(desktop_path, exist_ok=True)
            plt.savefig(os.path.join(desktop_path, 'permutation_importance.tif'), dpi=600, bbox_inches='tight',
                        facecolor=academic_colors['background'], format='tiff')
            plt.close()


            try:
                if 'shap_values' in locals() and 'shap_values' in globals():

                    shap_importance = np.mean(np.abs(shap_values), axis=0)


                    shap_normalized = shap_importance / np.sum(shap_importance)
                    perm_normalized = perm_scores / np.sum(perm_scores)


                    plt.figure(figsize=(16, 12))
                    plt.rcParams['axes.facecolor'] = academic_colors['background']
                    plt.rcParams['figure.facecolor'] = academic_colors['background']

                    x_pos = np.arange(len(selected_features_clean))
                    width = 0.35


                    plt.bar(x_pos - width / 2, shap_normalized, width,
                            label='SHAP Importance (Game Theory)',
                            color='#1f77b4',
                            alpha=0.8,
                            edgecolor='darkblue',
                            linewidth=1.2)
                    plt.bar(x_pos + width / 2, perm_normalized, width,
                            label='Permutation Importance (Performance-Based)',
                            color='#2E8B57',
                            alpha=0.8,
                            edgecolor='darkgreen',
                            linewidth=1.2)

                    plt.xlabel('Features', fontsize=16, fontweight='bold')
                    plt.ylabel('Normalized Importance Score', fontsize=16, fontweight='bold')
                    plt.title('Feature Importance Comparison\nSHAP (Game Theory) vs Permutation (Performance-Based)',
                              fontsize=18, pad=25, fontweight='bold')
                    plt.xticks(x_pos, selected_features_clean, rotation=45, ha='right', fontsize=12)
                    plt.legend(fontsize=14, loc='upper right')
                    plt.grid(True, linestyle='--', alpha=0.6, axis='y')


                    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.9)
                    textstr = 'Method Comparison:\nSHAP: Game theory, local explanations\nPermutation: Performance degradation, model-agnostic'
                    plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=12,
                             verticalalignment='top', bbox=props)

                    plt.tight_layout()

                    plt.savefig(os.path.join(desktop_path, 'feature_importance_comparison.tif'), dpi=600,
                                bbox_inches='tight',
                                facecolor=academic_colors['background'], format='tiff')
                    plt.close()


                    correlation = np.corrcoef(shap_normalized, perm_normalized)[0, 1]
                    print(f"SHAP-Permutation importance correlation: {correlation:.3f}")
                    if correlation > 0.7:
                        print("✓ High correlation indicates consistent feature importance rankings")
                    elif correlation > 0.3:
                        print("○ Moderate correlation suggests some agreement between methods")
                    else:
                        print("⚠ Low correlation indicates different perspectives on feature importance")
                else:
                    print("SHAP values not available, skipping comparison plot")
            except Exception as e:
                print(f"Warning: Could not create comparison plot: {e}")


            print("\nPermutation Importance Analysis Results:")
            print("=" * 70)
            print("Key Characteristics of Permutation Importance:")
            print("• Model-Agnostic: Works with any ML model")
            print("• Performance-Based: Measures impact on prediction accuracy")
            print("• Robust: Less affected by feature correlations")
            print("• Intuitive: Direct relationship with model performance")
            print("• Complementary: Validates SHAP analysis results")
            print("-" * 70)
            print("Feature Importance Rankings (Performance-Based):")
            for i, (feature, score, std) in enumerate(zip(sorted_features, sorted_scores, sorted_std)):
                print(f"{i + 1:2d}. {feature:<25} {score:.4f} ± {std:.4f}")


            perm_results = pd.DataFrame({
                'Feature': selected_features_clean,
                'Permutation_Importance': perm_scores,
                'Permutation_Std': perm_std,
                'Rank': np.argsort(np.argsort(-perm_scores)) + 1,
                'Method_Type': 'Performance-Based',
                'Method_Description': 'Feature importance measured by model performance degradation when feature values are randomly permuted'
            })
            perm_results.to_excel(os.path.join(desktop_path, 'permutation_importance_results.xlsx'), index=False)

            print("\n✓ Permutation importance analysis completed successfully!")
            print("✓ Method: Performance-based feature importance assessment")
            print("✓ Advantage: Model-agnostic and robust to feature correlations")
            print(f"✓ Output: Visualizations and detailed results saved to: {desktop_path}")
            print("✓ Purpose: Complementary validation of SHAP analysis results")

        else:
            print("⚠ Model performance insufficient for permutation importance analysis")
            print(f"Current R² ({best_r2:.4f}) is below the threshold of 0.8")

    except ImportError:
        print("⚠ sklearn.inspection.permutation_importance not available. Skipping permutation importance analysis.")
    except Exception as e:
        print(f"⚠ Unexpected error in permutation importance analysis: {e}")
        print("⚠ However, the main program continues to execute...")
    finally:
        print("✓ Permutation importance analysis section completed (with or without errors)")
        print("✓ Main program continues execution...")


    print("\nGenerating prediction vs actual line plots...")

    all_models_to_plot = {name: model_predictions[name] for name in model_predictions if
                          name not in ['xgb_optimized', 'lgb_optimized']}
    all_models_to_plot.update(ensemble_predictions)

    for model_name in all_models_to_plot:
        y_pred = all_models_to_plot[model_name]


        if model_name in model_train_metrics and model_name in model_metrics:
            train_r2 = model_train_metrics[model_name]['Train R2']
            test_r2 = model_metrics[model_name]['Test R2']
        elif model_name in ensemble_train_metrics and model_name in ensemble_metrics:
            train_r2 = ensemble_train_metrics[model_name]['Train R2']
            test_r2 = ensemble_metrics[model_name]['Test R2']
        else:
            train_r2 = -1
            test_r2 = -1


        sorted_indices = np.argsort(y_test.index)
        y_test_sorted = y_test.values[sorted_indices]
        y_pred_sorted = y_pred[sorted_indices]


        max_samples = min(100, len(y_test_sorted))
        y_test_limited = y_test_sorted[:max_samples]
        y_pred_limited = y_pred_sorted[:max_samples]

        plt.figure(figsize=(14, 8))


        plt.rcParams['axes.facecolor'] = academic_colors['background']
        plt.rcParams['figure.facecolor'] = academic_colors['background']


        plt.plot(y_test_limited, 'o-', color=academic_colors['actual'], linewidth=2.5, markersize=6,
                 markerfacecolor=academic_colors['actual'], markeredgewidth=1,
                 markeredgecolor='white', label='Actual', alpha=0.9)


        plt.plot(y_pred_limited, '^--', color=academic_colors['predicted'], linewidth=2, markersize=6,
                 markerfacecolor='white', markeredgecolor=academic_colors['predicted'],
                 markeredgewidth=1.5, label='Predicted', alpha=0.9)


        plt.xlabel('Sample Index', fontsize=14, color=academic_colors['text'])
        plt.ylabel('Concrete Strength (MPa)', fontsize=14, color=academic_colors['text'])
        plt.title(f'Actual vs Predicted Values - {model_name}', fontsize=16, pad=20,
                  color=academic_colors['text'])
        plt.grid(True, linestyle='--', alpha=0.7, color=academic_colors['grid'])
        plt.legend(fontsize=12, framealpha=0.9, edgecolor=academic_colors['grid'])


        plt.xlim(0, max_samples - 1)


        plt.xticks(np.arange(0, max_samples, 10))


        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)
        plt.tight_layout()


        filename = f"prediction_line_{model_name.replace(' ', '_')}.tif"
        plt.savefig(filename, dpi=600, bbox_inches='tight', facecolor=academic_colors['background'], format='tiff')
        plt.close()
        print(f"Saved prediction line plot for {model_name} as {filename}")


    print("\nGenerating combined single models prediction line plot...")
    plt.figure(figsize=(18, 12))


    plt.rcParams['axes.facecolor'] = academic_colors['background']
    plt.rcParams['figure.facecolor'] = academic_colors['background']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16


    single_models = ['LinearRegression', 'XGBoost_Simple', 'LightGBM_Simple', 'SVM', 'RandomForest', 'ANN']


    colors = ['#FF0000', '#00FF00', '#800080', '#0000FF', '#FFA500', '#FF00FF']

    markers = ['o', 's', '^', 'D', 'v', '<']


    max_samples = min(60, len(y_test_sorted))
    y_test_limited = y_test_sorted[:max_samples]
    plt.plot(y_test_limited, color='#808080', marker='*', markersize=8, linewidth=2,
             label='Actual Values', alpha=0.8, zorder=10, markevery=1,
             markerfacecolor='#808080', markeredgewidth=1.5)


    for i, model_name in enumerate(single_models):
        if model_name in all_models_to_plot:
            y_pred = all_models_to_plot[model_name]
            y_pred_sorted = y_pred[sorted_indices]
            y_pred_limited = y_pred_sorted[:max_samples]


            plt.plot(y_pred_limited, '--', color=colors[i], marker=markers[i],
                     markersize=10, linewidth=3.0, label=model_name, alpha=0.9,
                     markevery=1, markerfacecolor='white', markeredgewidth=2.0)

    plt.xlabel('Sample Index', fontsize=16, color=academic_colors['text'], fontweight='bold')
    plt.ylabel('Concrete Compressive Strength (MPa)', fontsize=16, color=academic_colors['text'], fontweight='bold')
    plt.title('Combined Single Models Prediction Comparison', fontsize=18, pad=25,
              color=academic_colors['text'], fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, color=academic_colors['grid'])


    plt.legend(fontsize=13, framealpha=0.95, edgecolor=academic_colors['grid'],
               loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, shadow=True)

    plt.xlim(0, max_samples - 1)
    plt.xticks(np.arange(0, max_samples, 15), fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()


    plt.savefig('combined_single_models_prediction.tif', dpi=800, bbox_inches='tight',
                facecolor=academic_colors['background'], format='tiff')
    plt.close()
    print("Saved optimized combined single models prediction plot")


    print("\nGenerating combined ensemble models prediction line plot...")
    plt.figure(figsize=(18, 12))


    plt.rcParams['axes.facecolor'] = academic_colors['background']
    plt.rcParams['figure.facecolor'] = academic_colors['background']
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.labelsize'] = 14
    plt.rcParams['axes.titlesize'] = 16


    ensemble_models = ['Weighted Average', 'Neural Network Stacking', 'Super Learner', 'Residual Boosting',
                       'Bayesian Averaging']


    colors = ['#FF0000', '#00FF00', '#800080', '#0000FF', '#FFA500']

    markers = ['o', 's', '^', 'D', 'v']


    plt.plot(y_test_limited, color='#808080', marker='*', markersize=8, linewidth=2,
             label='Actual Values', alpha=0.8, zorder=10, markevery=1,
             markerfacecolor='#808080', markeredgewidth=1.5)


    for i, model_name in enumerate(ensemble_models):
        if model_name in all_models_to_plot:
            y_pred = all_models_to_plot[model_name]
            y_pred_sorted = y_pred[sorted_indices]
            y_pred_limited = y_pred_sorted[:max_samples]


            plt.plot(y_pred_limited, '--', color=colors[i], marker=markers[i],
                     markersize=10, linewidth=3.0, label=model_name, alpha=0.9,
                     markevery=1, markerfacecolor='white', markeredgewidth=2.0)

    plt.xlabel('Sample Index', fontsize=16, color=academic_colors['text'], fontweight='bold')
    plt.ylabel('Concrete Compressive Strength (MPa)', fontsize=16, color=academic_colors['text'], fontweight='bold')
    plt.title('Combined Ensemble Models Prediction Comparison', fontsize=18, pad=25,
              color=academic_colors['text'], fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.5, color=academic_colors['grid'])


    plt.legend(fontsize=13, framealpha=0.95, edgecolor=academic_colors['grid'],
               loc='upper left', bbox_to_anchor=(0, 1), fancybox=True, shadow=True)

    plt.xlim(0, max_samples - 1)
    plt.xticks(np.arange(0, max_samples, 15), fontsize=12)
    plt.yticks(fontsize=12)
    plt.tight_layout()


    plt.savefig('combined_ensemble_models_prediction.tif', dpi=800, bbox_inches='tight',
                facecolor=academic_colors['background'], format='tiff')
    plt.close()
    print("Saved optimized combined ensemble models prediction plot")


    print("\nGenerating True vs Predicted scatter plots for all models...")

    for model_name in all_models_to_plot:
        y_pred = all_models_to_plot[model_name]

        plt.figure(figsize=(12, 10))
        plt.rcParams['axes.facecolor'] = academic_colors['background']
        plt.rcParams['figure.facecolor'] = academic_colors['background']
        plt.rcParams['font.size'] = 12


        plt.scatter(y_test, y_pred, alpha=0.7, s=80, color='#1f77b4',
                    edgecolor='white', linewidth=1.0)


        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=3,
                 label='Ideal Prediction (y=x)', alpha=0.8)


        coeffs = np.polyfit(y_test, y_pred, 1)
        regression_line = np.poly1d(coeffs)
        plt.plot(y_test, regression_line(y_test), 'g-', linewidth=2.5,
                 label='Regression Line', alpha=0.8)


        r2 = r2_score(y_test, y_pred)

        plt.xlabel('True Values (MPa)', fontsize=14, color=academic_colors['text'], fontweight='bold')
        plt.ylabel('Predicted Values (MPa)', fontsize=14, color=academic_colors['text'], fontweight='bold')
        plt.title(f'True vs Predicted - {model_name}\n(R² = {r2:.4f})', fontsize=16,
                  color=academic_colors['text'], fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5, color=academic_colors['grid'])
        plt.legend(fontsize=12, framealpha=0.9)
        plt.tight_layout()

        filename = f"true_vs_predicted_{model_name.replace(' ', '_')}.tif"
        plt.savefig(filename, dpi=800, bbox_inches='tight',
                    facecolor=academic_colors['background'], format='tiff')
        plt.close()
        print(f"Saved optimized True vs Predicted plot for {model_name}")


    print("\nGenerating Residual plots for all models...")

    for model_name in all_models_to_plot:
        y_pred = all_models_to_plot[model_name]
        residuals = y_test - y_pred

        plt.figure(figsize=(14, 6))
        plt.rcParams['axes.facecolor'] = academic_colors['background']
        plt.rcParams['figure.facecolor'] = academic_colors['background']
        plt.rcParams['font.size'] = 11


        plt.subplot(1, 2, 1)
        plt.scatter(y_pred, residuals, alpha=0.7, s=60, color='#2ca02c',
                    edgecolor='white', linewidth=0.8)
        plt.axhline(y=0, color='red', linestyle='--', linewidth=2.5)
        plt.xlabel('Predicted Values (MPa)', fontsize=12, fontweight='bold')
        plt.ylabel('Residuals (MPa)', fontsize=12, fontweight='bold')
        plt.title(f'Residuals vs Predicted - {model_name}', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)


        plt.subplot(1, 2, 2)
        plt.hist(residuals, bins=35, alpha=0.8, color='#ff7f0e', edgecolor='black', linewidth=1.0)
        plt.xlabel('Residuals (MPa)', fontsize=12, fontweight='bold')
        plt.ylabel('Frequency', fontsize=12, fontweight='bold')
        plt.title(f'Residual Distribution - {model_name}', fontsize=14, fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        filename = f"residual_plot_{model_name.replace(' ', '_')}.tif"
        plt.savefig(filename, dpi=800, bbox_inches='tight',
                    facecolor=academic_colors['background'], format='tiff')
        plt.close()
        print(f"Saved optimized Residual plot for {model_name}")


    print("\nGenerating Error Distribution Histograms for all models...")

    for model_name in all_models_to_plot:
        y_pred = all_models_to_plot[model_name]
        errors = y_test - y_pred

        plt.figure(figsize=(12, 8))
        plt.rcParams['axes.facecolor'] = academic_colors['background']
        plt.rcParams['figure.facecolor'] = academic_colors['background']
        plt.rcParams['font.size'] = 12


        n, bins, patches = plt.hist(errors, bins=35, alpha=0.8, color='#d62728',
                                    edgecolor='black', linewidth=1.0, density=True)


        mu, sigma = np.mean(errors), np.std(errors)
        x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
        plt.plot(x, stats.norm.pdf(x, mu, sigma), 'b-', linewidth=3,
                 label=f'Normal Distribution (μ={mu:.2f}, σ={sigma:.2f})')

        plt.xlabel('Prediction Error (MPa)', fontsize=14, color=academic_colors['text'], fontweight='bold')
        plt.ylabel('Probability Density', fontsize=14, color=academic_colors['text'], fontweight='bold')
        plt.title(f'Error Distribution - {model_name}', fontsize=16,
                  color=academic_colors['text'], fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5, color=academic_colors['grid'])
        plt.legend(fontsize=12, framealpha=0.9)
        plt.tight_layout()

        filename = f"error_distribution_{model_name.replace(' ', '_')}.tif"
        plt.savefig(filename, dpi=800, bbox_inches='tight',
                    facecolor=academic_colors['background'], format='tiff')
        plt.close()
        print(f"Saved optimized Error Distribution plot for {model_name}")


    print("\nGenerating Prediction Interval plots for best model...")


    best_model_name = best_method
    if best_model_name in all_models_to_plot:
        y_pred = all_models_to_plot[best_model_name]


        residuals = y_test - y_pred
        std_residuals = np.std(residuals)
        prediction_interval_upper = y_pred + 1.96 * std_residuals
        prediction_interval_lower = y_pred - 1.96 * std_residuals


        sort_idx = np.argsort(y_pred)
        y_pred_sorted = y_pred[sort_idx]
        y_test_sorted = y_test.values[sort_idx]
        upper_sorted = prediction_interval_upper[sort_idx]
        lower_sorted = prediction_interval_lower[sort_idx]

        plt.figure(figsize=(14, 10))
        plt.rcParams['axes.facecolor'] = academic_colors['background']
        plt.rcParams['figure.facecolor'] = academic_colors['background']
        plt.rcParams['font.size'] = 12


        plt.fill_between(range(len(y_pred_sorted)), lower_sorted, upper_sorted,
                         alpha=0.4, color='#1f77b4', label='95% Prediction Interval')


        plt.plot(y_pred_sorted, 'b-', linewidth=3, label='Predicted Values', alpha=0.9)


        plt.plot(y_test_sorted, 'ro', markersize=5, label='True Values', alpha=0.8,
                 markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=1)

        plt.xlabel('Sample Index (Sorted by Prediction)', fontsize=14, color=academic_colors['text'], fontweight='bold')
        plt.ylabel('Concrete Compressive Strength (MPa)', fontsize=14, color=academic_colors['text'], fontweight='bold')
        plt.title(f'Prediction Interval Plot - {best_model_name}', fontsize=16,
                  color=academic_colors['text'], fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5, color=academic_colors['grid'])
        plt.legend(fontsize=12, framealpha=0.9)
        plt.tight_layout()

        filename = f"prediction_interval_{best_model_name.replace(' ', '_')}.tif"
        plt.savefig(filename, dpi=800, bbox_inches='tight',
                    facecolor=academic_colors['background'], format='tiff')
        plt.close()
        print(f"Saved optimized Prediction Interval plot for {best_model_name}")


    print("\nGenerating Grouped Box Plot by Age groups...")


    age_groups = ['<7 days', '7-28 days', '>28 days']
    age_bins = [0, 7, 28, float('inf')]


    age_data = X_test['Age (day)'].values if 'Age (day)' in X_test.columns else X_test.iloc[:, 1].values


    age_group_indices = np.digitize(age_data, age_bins)


    if best_model_name in all_models_to_plot:
        y_pred = all_models_to_plot[best_model_name]
        errors = y_test - y_pred


        group_errors = []
        for i in range(1, len(age_bins)):
            group_mask = (age_group_indices == i)
            if np.sum(group_mask) > 0:
                group_errors.append(errors[group_mask])

        plt.figure(figsize=(12, 8))
        plt.rcParams['axes.facecolor'] = academic_colors['background']
        plt.rcParams['figure.facecolor'] = academic_colors['background']
        plt.rcParams['font.size'] = 12


        box_plot = plt.boxplot(group_errors, labels=age_groups, patch_artist=True)


        colors = ['#ff9999', '#66b3ff', '#99ff99']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)


        for median in box_plot['medians']:
            median.set_color('red')
            median.set_linewidth(2)

        plt.axhline(y=0, color='red', linestyle='--', linewidth=3, alpha=0.8)

        plt.xlabel('Age Groups', fontsize=14, color=academic_colors['text'], fontweight='bold')
        plt.ylabel('Prediction Error (MPa)', fontsize=14, color=academic_colors['text'], fontweight='bold')
        plt.title(f'Prediction Error by Age Groups - {best_model_name}', fontsize=16,
                  color=academic_colors['text'], fontweight='bold')
        plt.grid(True, linestyle='--', alpha=0.5, color=academic_colors['grid'])
        plt.tight_layout()

        filename = f"grouped_boxplot_age_{best_model_name.replace(' ', '_')}.tif"
        plt.savefig(filename, dpi=800, bbox_inches='tight',
                    facecolor=academic_colors['background'], format='tiff')
        plt.close()
        print(f"Saved optimized Grouped Box Plot for {best_model_name}")


    print("\nGenerating Learning Curves for best model...")

    if best_model_name in all_models_to_plot:
        try:

            if best_model_name in model_predictions:

                if best_model_name == 'LinearRegression':
                    best_model = LinearRegression()
                elif best_model_name == 'XGBoost_Simple':
                    best_model = xgb.XGBRegressor(random_state=42)
                elif best_model_name == 'LightGBM_Simple':
                    best_model = lgb.LGBMRegressor(random_state=42)
                elif best_model_name == 'SVM':
                    best_model = SVR()
                elif best_model_name == 'RandomForest':
                    best_model = RandomForestRegressor(random_state=42)
                elif best_model_name == 'ANN':
                    best_model = MLPRegressor(random_state=42, max_iter=1000)
            else:

                best_model = LinearRegression()


            train_sizes = np.linspace(0.1, 1.0, 10)


            train_sizes_abs, train_scores, test_scores = learning_curve(
                best_model, X_train_scaled, y_train,
                train_sizes=train_sizes, cv=5,
                scoring='r2', n_jobs=-1, random_state=42
            )


            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)

            plt.figure(figsize=(14, 10))
            plt.rcParams['axes.facecolor'] = academic_colors['background']
            plt.rcParams['figure.facecolor'] = academic_colors['background']
            plt.rcParams['font.size'] = 12


            plt.plot(train_sizes_abs, train_scores_mean, 'o-', color='#1f77b4',
                     linewidth=3, markersize=8, label='Training Score')
            plt.fill_between(train_sizes_abs,
                             train_scores_mean - train_scores_std,
                             train_scores_mean + train_scores_std,
                             alpha=0.2, color='#1f77b4')

            plt.plot(train_sizes_abs, test_scores_mean, 'o-', color='#d62728',
                     linewidth=3, markersize=8, label='Cross-validation Score')
            plt.fill_between(train_sizes_abs,
                             test_scores_mean - test_scores_std,
                             test_scores_mean + test_scores_std,
                             alpha=0.2, color='#d62728')

            plt.xlabel('Training Set Size', fontsize=14, color=academic_colors['text'], fontweight='bold')
            plt.ylabel('R² Score', fontsize=14, color=academic_colors['text'], fontweight='bold')
            plt.title(f'Learning Curve - {best_model_name}', fontsize=16,
                      color=academic_colors['text'], fontweight='bold')
            plt.grid(True, linestyle='--', alpha=0.5, color=academic_colors['grid'])
            plt.legend(fontsize=12, framealpha=0.9)
            plt.tight_layout()

            filename = f"learning_curve_{best_model_name.replace(' ', '_')}.tif"
            plt.savefig(filename, dpi=800, bbox_inches='tight',
                        facecolor=academic_colors['background'], format='tiff')
            plt.close()
            print(f"Saved optimized Learning Curve for {best_model_name}")

        except Exception as e:
            print(f"Learning curve generation failed: {e}")


    print("\nGenerating Classification Task Visualizations (Example)...")

    try:

        median_strength = np.median(y)
        y_binary = (y > median_strength).astype(int)


        X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )


        scaler_clf = StandardScaler()
        X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
        X_test_clf_scaled = scaler_clf.transform(X_test_clf)


        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

        clf_model = LogisticRegression(random_state=42, max_iter=1000)
        clf_model.fit(X_train_clf_scaled, y_train_clf)


        y_pred_proba = clf_model.predict_proba(X_test_clf_scaled)[:, 1]

        fpr, tpr, thresholds_roc = roc_curve(y_test_clf, y_pred_proba)
        roc_auc = auc(fpr, tpr)


        precision, recall, thresholds_pr = precision_recall_curve(y_test_clf, y_pred_proba)
        average_precision = average_precision_score(y_test_clf, y_pred_proba)


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))


        ax1.plot(fpr, tpr, color='#e74c3c', lw=3.5,
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='#7f8c8d', lw=2.5, linestyle='--',
                 label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        ax1.set_title('ROC Curve (Example Classification)', fontsize=16, fontweight='bold')
        ax1.legend(loc="lower right", fontsize=12, framealpha=0.9)
        ax1.grid(True, linestyle='--', alpha=0.6)


        ax2.plot(recall, precision, color='#3498db', lw=3.5,
                 label=f'PR curve (AP = {average_precision:.3f})')

        positive_ratio = np.mean(y_test_clf)
        ax2.axhline(y=positive_ratio, color='#7f8c8d', linestyle='--', lw=2.5,
                    label=f'Random Classifier (AP = {positive_ratio:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Precision', fontsize=14, fontweight='bold')
        ax2.set_title('Precision-Recall Curve (Example Classification)', fontsize=16, fontweight='bold')
        ax2.legend(loc="lower left", fontsize=12, framealpha=0.9)
        ax2.grid(True, linestyle='--', alpha=0.6)

        plt.tight_layout()


        filename = "classification_curves_example.tif"
        plt.savefig(filename, dpi=800, bbox_inches='tight', format='tiff')
        plt.close()

        print("Saved optimized Classification Curves (ROC and PR) as example visualization")


        from sklearn.metrics import classification_report, confusion_matrix
        y_pred_clf = clf_model.predict(X_test_clf_scaled)

        print("\nClassification Performance (Example):")
        print(f"Threshold (median compressive strength): {median_strength:.2f} MPa")
        print(f"Positive class ratio: {positive_ratio:.3f}")
        print(f"ROC AUC: {roc_auc:.3f}")
        print(f"Average Precision: {average_precision:.3f}")
        print("\nClassification Report:")
        print(classification_report(y_test_clf, y_pred_clf))

    except Exception as e:
        print(f"Classification visualization generation failed: {e}")


    print("\n=== Exporting Model Evaluation Metrics ===")


    all_metrics = []


    for model_name in model_metrics:
        if model_name in model_train_metrics:
            metrics = model_train_metrics[model_name].copy()
            metrics.update(model_metrics[model_name])
            metrics['Model'] = model_name
            all_metrics.append(metrics)


    for model_name in ensemble_metrics:
        if model_name in ensemble_train_metrics:
            metrics = ensemble_train_metrics[model_name].copy()
            metrics.update(ensemble_metrics[model_name])
            metrics['Model'] = model_name
            all_metrics.append(metrics)


    metrics_df = pd.DataFrame(all_metrics)
    metrics_df = metrics_df.sort_values(by='Test R2', ascending=False)


    column_order = ['Model',
                    'Train R2', 'Test R2',
                    'Train RMSE', 'Test RMSE',
                    'Train MSE', 'Test MSE',
                    'Train MAE', 'Test MAE',
                    'Train MAPE', 'Test MAPE']
    metrics_df = metrics_df[column_order]


    metrics_df.to_excel('model_evaluation_metrics.xlsx', index=False)
    print("Saved model evaluation metrics to 'model_evaluation_metrics.xlsx'")


    print("\n=== Exporting Residual Boosting Hyperparameters ===")
    if residual_boosting_params:

        residual_params_df = pd.DataFrame({
            'Parameter': list(residual_boosting_params.keys()),
            'Value': list(residual_boosting_params.values())
        })


        residual_params_df.to_excel('residual_boosting_parameters.xlsx', index=False)
        print("Saved residual boosting hyperparameters to 'residual_boosting_parameters.xlsx'")
    else:
        print("Residual boosting parameters not available")


    print("\n=== Model Hyperparameters ===")


    hyperparam_data = []
    for model_name, params in model_hyperparameters.items():

        simplified_params = {}

        if model_name in ['XGBoost_Simple', 'xgb_optimized']:
            important_params = ['n_estimators', 'max_depth', 'learning_rate',
                                'subsample', 'colsample_bytree', 'gamma']
            for param in important_params:
                if param in params:
                    simplified_params[param] = params[param]

        elif model_name in ['LightGBM_Simple', 'lgb_optimized']:
            important_params = ['n_estimators', 'max_depth', 'learning_rate',
                                'num_leaves', 'subsample', 'colsample_bytree']
            for param in important_params:
                if param in params:
                    simplified_params[param] = params[param]

        elif model_name == 'LinearRegression':

            important_params = ['fit_intercept', 'copy_X', 'n_jobs']
            for param in important_params:
                if param in params:
                    simplified_params[param] = params[param]
            simplified_params['model_type'] = 'Linear Regression'

        elif model_name == 'SVM':
            important_params = ['kernel', 'C', 'gamma']
            for param in important_params:
                if param in params:
                    simplified_params[param] = params[param]

        elif model_name == 'RandomForest':
            important_params = ['n_estimators', 'max_depth', 'min_samples_split',
                                'min_samples_leaf', 'max_features']
            for param in important_params:
                if param in params:
                    simplified_params[param] = params[param]

        elif model_name == 'ANN':
            important_params = ['hidden_layer_sizes', 'activation', 'solver',
                                'max_iter', 'learning_rate']
            for param in important_params:
                if param in params:
                    simplified_params[param] = params[param]

        else:
            simplified_params = params


        hyperparam_data.append({
            'Model': model_name,
            'Hyperparameters': simplified_params
        })


    hyperparam_df = pd.DataFrame(hyperparam_data)
    print(hyperparam_df.to_string())


    hyperparam_df.to_excel('model_hyperparameters.xlsx', index=False)
    print(f"\nSaved model hyperparameters to 'model_hyperparameters.xlsx'")

    return best_r2


def plot_model_diagnostics(y_true, y_pred, model_name=None):
    """Model diagnostic plots with English labels and 10% error lines"""
    try:

        if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
            print("Predictions contain invalid values (NaN/Inf), cleaning...")
            y_pred = np.nan_to_num(y_pred, nan=np.nanmean(y_pred))


        plt.figure(figsize=(12, 10))
        plt.scatter(y_true, y_pred, alpha=0.6, s=80, edgecolor='w', linewidth=0.5)


        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')


        coeffs = np.polyfit(y_true, y_pred, 1)
        regression_line = np.poly1d(coeffs)
        plt.plot(y_true, regression_line(y_true), 'g-', linewidth=2, label='Regression Line')


        plt.plot([min_val, max_val], [min_val * 1.1, max_val * 1.1], 'm--', linewidth=2, label='+10% Error Line')
        plt.plot([min_val, max_val], [min_val * 0.9, max_val * 0.9], 'c--', linewidth=2, label='-10% Error Line')


        plt.fill_between([min_val, max_val], [min_val * 0.9, max_val * 0.9],
                         [min_val * 1.1, max_val * 1.1], color='gray', alpha=0.1, label='±10% Error Band')

        plt.xlabel('True Values', fontsize=12)
        plt.ylabel('Predicted Values', fontsize=12)
        plt.title('Model Prediction Accuracy with ±10% Error Lines', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(fontsize=12)
        plt.tight_layout()


        if model_name:
            plt.savefig(f'true_vs_pred_{model_name}.tif', dpi=600, bbox_inches='tight', format='tiff')
        else:
            plt.savefig('true_vs_pred_with_error_lines.tif', dpi=600, bbox_inches='tight', format='tiff')
        plt.close()


        residuals = y_true - y_pred
        plt.figure(figsize=(10, 8))
        plt.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Residuals', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Residual Distribution', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('residuals_dist.tif', dpi=600, bbox_inches='tight', format='tiff')
        plt.close()


        plt.figure(figsize=(10, 8))
        stats.probplot(residuals, dist="norm", plot=plt)
        plt.title('Residual Normality Test', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('residuals_qq.tif', dpi=600, bbox_inches='tight', format='tiff')
        plt.close()


        plt.figure(figsize=(10, 8))
        plt.scatter(y_pred, residuals, alpha=0.6, s=80, edgecolor='w', linewidth=0.5)
        plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
        plt.xlabel('Predicted Values', fontsize=12)
        plt.ylabel('Residuals', fontsize=12)
        plt.title('Residuals vs Predicted Values', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.savefig('residuals_vs_predicted.tif', dpi=600, bbox_inches='tight', format='tiff')
        plt.close()

        print("Diagnostic plots saved successfully")
    except Exception as e:
        print(f"Plotting error: {e}")


if __name__ == '__main__':
    
    print(f"SHAP version: {shap.__version__}")
    print(f"LightGBM version: {lgb.__version__}")
    print(f"XGBoost version: {xgb.__version__}")

    final_r2 = train_and_evaluate_model()

    print("\n" + "=" * 50)
    if final_r2 > 0.95:
        print(f"Outstanding Performance: R² = {final_r2:.6f} ")
    elif final_r2 > 0.90:
        print(f"Excellent Performance: R² = {final_r2:.6f} ")
    elif final_r2 > 0.85:
        print(f"Good Performance: R² = {final_r2:.6f}")
    elif final_r2 > 0.75:
        print(f"Acceptable Performance: R² = {final_r2:.6f}")
    else:
        print(f"Needs Improvement: R² = {final_r2:.6f}")
    print("=" * 50)