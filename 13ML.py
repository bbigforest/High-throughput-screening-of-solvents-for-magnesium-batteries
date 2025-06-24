import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge, Ridge, LinearRegression as LR
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Arial'

file_path = "feature.xlsx"
df = pd.read_excel(file_path)

X = df.iloc[:, 1:-7]
y = df.iloc[:, -4]
print(X)
print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

ss = MinMaxScaler()
X_train = ss.fit_transform(X_train)
X_test = ss.transform(X_test)

def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on training set
    train_predictions = model.predict(X_train)

    # Make predictions on testing set
    test_predictions = model.predict(X_test)

    # Calculate and print metrics
    mae_train = mean_absolute_error(y_train, train_predictions)
    r2_train = r2_score(y_train, train_predictions)
    mae_test = mean_absolute_error(y_test, test_predictions)
    r2_test = r2_score(y_test, test_predictions)

    print(f"{model} - train MAE: {mae_train}, train R-squared: {r2_train}")
    print(f"{model} - test MAE: {mae_test}, test R-squared: {r2_test}")

    return mae_train, r2_train, mae_test, r2_test

model_params = {
    'RF': {'model': RandomForestRegressor,
           'params': {'n_estimators': 100, 'max_depth': 15, 'min_samples_split': 2, 'min_samples_leaf': 1,
                      'max_features': 'sqrt', 'random_state': 42}},
    'GBR': {'model': GradientBoostingRegressor,
            'params': {'n_estimators': 100, 'learning_rate': 0.05, 'max_depth': 5, 'min_samples_split': 2,
                       'min_samples_leaf': 1, 'random_state': 42}},
    'SVR': {'model': SVR, 'params': {'kernel': 'rbf', 'C': 1.0, 'epsilon': 0.1}},
    'KNN': {'model': KNeighborsRegressor, 'params': {'n_neighbors': 5, 'weights': 'uniform'}},
    'DT': {'model': DecisionTreeRegressor,
           'params': {'max_depth': 10, 'min_samples_split': 2, 'min_samples_leaf': 6, 'random_state': 42}},
    'BR': {'model': BayesianRidge, 'params': {'max_iter': 300}, 'random_state': 42},
    'LR': {'model': LR, 'params': {'fit_intercept': True}, 'random_state': 42},
    'XGBR': {'model': XGBRegressor,
             'params': {'n_estimators': 100, 'learning_rate': 0.02, 'max_depth': 6, 'min_child_weight': 1,
                        'subsample': 0.8, 'colsample_bytree': 0.8, 'objective': 'reg:squarederror',
                        'random_state': 42}},
    'Ridge': {'model': Ridge, 'params': {'alpha': 1.0}},
    'ETR': {'model': ExtraTreesRegressor,
            'params': {'n_estimators': 100, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 2,
                       'random_state': 42}},
    'KRR': {'model': KernelRidge, 'params': {'alpha': 0.1, 'kernel': 'linear', 'gamma': None}},
    'Bagging': {'model': BaggingRegressor,
                'params': {'n_estimators': 100, 'max_samples': 1.0, 'max_features': 1.0, 'bootstrap': True,
                           'bootstrap_features': False, 'random_state': 42}},
    'ANN': {'model': MLPRegressor,
            'params': {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam',
                       'alpha': 0.0001, 'batch_size': 'auto', 'learning_rate': 'constant',
                       'learning_rate_init': 0.001, 'power_t': 0.5, 'max_iter': 200,
                       'shuffle': True, 'random_state': 42, 'tol': 1e-4, 'verbose': False,
                       'warm_start': False, 'momentum': 0.9, 'nesterovs_momentum': True,
                       'early_stopping': False, 'validation_fraction': 0.1, 'beta_1': 0.9,
                       'beta_2': 0.999, 'epsilon': 1e-8, 'n_iter_no_change': 10}}
}

results = []
for name, config in model_params.items():
    model = config['model'](**config['params'])
    mae_train, r2_train, mae_test, r2_test = train_and_evaluate(model, X_train, y_train, X_test, y_test)
    results.append((name, mae_train, r2_train, mae_test, r2_test))

average_mae_train = np.mean([result[1] for result in results])
average_mae_test = np.mean([result[3] for result in results])
average_r2_train = np.mean([result[2] for result in results])
average_r2_test = np.mean([result[4] for result in results])

with open('LUMO-HOMO_MAE_results.txt', 'w') as f:
    f.write(f"{'Model':<10} {'Train MAE':<15} {'Train R2':<15} {'Test MAE':<15} {'Test R2':<15}\n")
    for name, mae_train, r2_train, mae_test, r2_test in results:
        f.write(f"{name:<10} {mae_train:<15.6f} {r2_train:<15.6f} {mae_test:<15.6f} {r2_test:<15.6f}\n")
    f.write(
        f"\n{'Average':<10} {average_mae_train:<15.6f} {average_r2_train:<15.6f} {average_mae_test:<15.6f} {average_r2_test:<15.6f}\n")


