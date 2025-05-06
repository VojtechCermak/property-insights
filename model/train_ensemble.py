import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import RidgeCV, LassoCV
from inference import ModelExplainer

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
ENSEMBLE_DIR = os.path.join(ARTIFACTS_DIR, 'ensemble')
BAGGING_DIR = os.path.join(ENSEMBLE_DIR, 'bagging')
STACKING_DIR = os.path.join(ENSEMBLE_DIR, 'stacking')
os.makedirs(ENSEMBLE_DIR, exist_ok=True)
os.makedirs(BAGGING_DIR, exist_ok=True)
os.makedirs(STACKING_DIR, exist_ok=True)

class GreedyBaggingEnsemble:
    def __init__(self):
        self.selected_models = None

    def fit(self, oof_df, model_names):
        current_models = list(model_names)
        best_rmse = mean_squared_error(oof_df['y_true'], oof_df[[f'y_pred_{m}' for m in current_models]].mean(axis=1), squared=False)
        changed = True
        while changed and len(current_models) > 1:
            changed = False
            for m in current_models:
                candidate = [mm for mm in current_models if mm != m]
                if not candidate:
                    continue
                preds = oof_df[[f'y_pred_{mm}' for mm in candidate]].mean(axis=1)
                rmse = mean_squared_error(oof_df['y_true'], preds, squared=False)
                if rmse < best_rmse:
                    best_rmse = rmse
                    current_models = candidate
                    changed = True
                    break  # restart after removal
        self.selected_models = current_models
        return self

    def predict(self, df):
        return df[[f'y_pred_{m}' for m in self.selected_models]].mean(axis=1)

class StackedLassoEnsemble:
    def __init__(self, alphas=np.logspace(-3, 3, 20), cv=4):
        self.alphas = alphas
        self.cv = cv
        self.model = None

    def fit(self, oof_df):
        X = oof_df[[c for c in oof_df.columns if c.startswith('y_pred_')]].values
        y = oof_df['y_true'].values
        self.model = LassoCV(alphas=self.alphas, cv=self.cv)
        self.model.fit(X, y)
        return self

    def predict(self, df):
        X = df[[c for c in df.columns if c.startswith('y_pred_')]].values
        return self.model.predict(X)



# Find all OOF and test prediction files
model_dirs = [d for d in glob.glob(os.path.join(ARTIFACTS_DIR, '*')) if os.path.isdir(d) and not d.endswith('ensemble')]
model_oof_files = {os.path.basename(d): os.path.join(d, 'predictions_oof.csv') for d in model_dirs if os.path.exists(os.path.join(d, 'predictions_oof.csv'))}
model_test_files = {os.path.basename(d): os.path.join(d, 'predictions_test.csv') for d in model_dirs if os.path.exists(os.path.join(d, 'predictions_test.csv'))}
# Also include neural net full test predictions
if os.path.exists(os.path.join(ARTIFACTS_DIR, 'image_dense', 'predictions_test_full.csv')):
    model_test_files['image_dense'] = os.path.join(ARTIFACTS_DIR, 'image_dense', 'predictions_test_full.csv')
    model_oof_files['image_dense'] = os.path.join(ARTIFACTS_DIR, 'image_dense', 'predictions_oof.csv')

# Load all OOF and test predictions into DataFrames
all_oof = []
all_test = []
model_names = []
for name, oof_path in model_oof_files.items():
    df = pd.read_csv(oof_path)
    df = df.rename(columns={'y_pred': f'y_pred_{name}'})
    all_oof.append(df.set_index('id'))
    model_names.append(name)
for name, test_path in model_test_files.items():
    df = pd.read_csv(test_path)
    df = df.rename(columns={'y_pred': f'y_pred_{name}'})
    all_test.append(df.set_index('id'))

# Align by id
oof_df = pd.concat(all_oof, axis=1, join='inner')
test_df = pd.concat(all_test, axis=1, join='inner')

# Only keep one y_true column
oof_df = oof_df.loc[:,~oof_df.columns.duplicated()]
test_df = test_df.loc[:,~test_df.columns.duplicated()]

# Mean ensemble
mean_ensemble = GreedyBaggingEnsemble().fit(oof_df, model_names)
mean_oof = mean_ensemble.predict(oof_df)
mean_test = mean_ensemble.predict(test_df)

# Save bagging ensemble predictions and metrics in bagging dir
pd.DataFrame({'id': oof_df.index, 'y_true': oof_df['y_true'], 'y_pred': mean_oof}).to_csv(os.path.join(BAGGING_DIR, 'mean_oof.csv'), index=False)
pd.DataFrame({'id': test_df.index, 'y_true': test_df['y_true'], 'y_pred': mean_test}).to_csv(os.path.join(BAGGING_DIR, 'mean_test.csv'), index=False)
mean_metrics = {
    'oof_rmse': float(mean_squared_error(oof_df['y_true'], mean_oof, squared=False)),
    'test_rmse': float(mean_squared_error(test_df['y_true'], mean_test, squared=False)),
    'test_mae': float(mean_absolute_error(test_df['y_true'], mean_test)),
    'test_mape': float(mean_absolute_percentage_error(test_df['y_true'], mean_test)),
    'models': mean_ensemble.selected_models
}
with open(os.path.join(BAGGING_DIR, 'mean_metrics.json'), 'w') as f:
    import json; json.dump(mean_metrics, f, indent=2)

# Stacked ensemble
stacked_ensemble = StackedLassoEnsemble().fit(oof_df)
stack_oof = stacked_ensemble.predict(oof_df)
stack_test = stacked_ensemble.predict(test_df)

# Save stacking ensemble predictions and metrics in stacking dir
pd.DataFrame({'id': oof_df.index, 'y_true': oof_df['y_true'], 'y_pred': stack_oof}).to_csv(os.path.join(STACKING_DIR, 'stacked_oof.csv'), index=False)
pd.DataFrame({'id': test_df.index, 'y_true': test_df['y_true'], 'y_pred': stack_test}).to_csv(os.path.join(STACKING_DIR, 'stacked_test.csv'), index=False)
stack_metrics = {
    'oof_rmse': float(mean_squared_error(oof_df['y_true'], stack_oof, squared=False)),
    'test_rmse': float(mean_squared_error(test_df['y_true'], stack_test, squared=False)),
    'test_mae': float(mean_absolute_error(test_df['y_true'], stack_test)),
    'test_mape': float(mean_absolute_percentage_error(test_df['y_true'], stack_test)),
    'coefs': stacked_ensemble.model.coef_.tolist(),
    'intercept': float(stacked_ensemble.model.intercept_)
}
with open(os.path.join(STACKING_DIR, 'stacked_metrics.json'), 'w') as f:
    import json; json.dump(stack_metrics, f, indent=2)

# Generate summary reports for ensembles
mean_report = ModelExplainer(
    save_folder=BAGGING_DIR,
    model=None,
    X_test=None,
    y_test=test_df['y_true'].values,
    feature_names=None,
    model_name='MeanEnsemble',
    inv_transform=None,
    skip_feature_importance=True,
    X_raw=pd.DataFrame({'id': test_df.index})
)
mean_report.run_all(best_params=mean_metrics['models'], y_pred=mean_test.values)

stacked_report = ModelExplainer(
    save_folder=STACKING_DIR,
    model=None,
    X_test=None,
    y_test=test_df['y_true'].values,
    feature_names=None,
    model_name='StackedEnsemble',
    inv_transform=None,
    skip_feature_importance=True,
    X_raw=pd.DataFrame({'id': test_df.index})
)
stacked_report.run_all(best_params={'coefs': stack_metrics['coefs'], 'intercept': stack_metrics['intercept']}, y_pred=stack_test)

print('Mean and stacked ensemble results saved to', ENSEMBLE_DIR) 