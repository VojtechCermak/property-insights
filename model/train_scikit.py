import os
import pandas as pd
import numpy as np
import random
import argparse
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.base import clone
from features import create_pipeline
from inference import ModelExplainer
import joblib
import json

# Set all random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

parser = argparse.ArgumentParser(description='Train regression models for various objectives.')
parser.add_argument('--objective', choices=['price', 'price_psm', 'lognorm_price', 'lognorm_price_psm'], default='price', help='Training objective')
parser.add_argument('--cut', type=float, default=0.8, help='quantile cutoff for price')
args = parser.parse_args()

# 1. Load data
df = pd.read_csv('../data/processed/metadata.csv')
cut = df['price'].quantile(args.cut)
df = df[df['price'] < cut]

# --- Target selection and transformation ---
if args.objective == 'price':
    y = df['price'].values
    y_label = 'price'
    def inv_transform(y, X): return y
elif args.objective == 'price_psm':
    y = df['price_per_sqm'].values
    y_label = 'price_psm'
    def inv_transform(y, X): return y * X['usableArea'].values
elif args.objective == 'lognorm_price':
    y_raw = np.log(df['price'].values)
    mean = y_raw.mean()
    std = y_raw.std()
    y = (y_raw - mean) / std
    y_label = 'lognorm_price'
    def inv_transform(y, X): return np.exp(y * std + mean)
elif args.objective == 'lognorm_price_psm':
    y_raw = np.log(df['price_per_sqm'].values)
    mean = y_raw.mean()
    std = y_raw.std()
    y = (y_raw - mean) / std
    y_label = 'lognorm_price_psm'
    def inv_transform(y, X): return np.exp(y * std + mean) * X['usableArea'].values
else:
    raise ValueError('Unknown objective')

print(len(y))
X = df.copy()
X_trainval, X_test, y_trainval, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Model definitions and hyperparameters
models = {
    'RandomForest': (
        RandomForestRegressor(random_state=SEED, n_jobs=-1),
        create_pipeline(cat_encoder='onehot', loc_clusters=100),
        {
            'n_estimators': [300, 500, 800],
            'max_depth': [None, 10, 20, 30, 50],
            'min_samples_split': [2, 5, 10],
        }
    ),
    'XGBoost': (
        XGBRegressor(random_state=SEED, n_jobs=-1, verbosity=0),
        create_pipeline(cat_encoder='onehot', loc_clusters=100),
        {
            'n_estimators': [300, 500, 800],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
        }
    ),
    'CatBoost': (
        CatBoostRegressor(random_state=SEED, verbose=0),
        create_pipeline(cat_encoder='onehot', loc_clusters=100),
        {
            'iterations': [300, 500, 800],
            'depth': [4, 6, 8, 10],
            'learning_rate': [0.005, 0.01, 0.03, 0.05, 0.1],
            'l2_leaf_reg': [1, 3, 5, 7, 10],
        }
    ),
    'Ridge': (
        Ridge(),
        create_pipeline(cat_encoder='onehot', loc_clusters=300),
        {
            'alpha': [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 50.0, 100.0],
        }
    ),
    'Lasso': (
        Lasso(max_iter=5000),
        create_pipeline(cat_encoder='onehot', loc_clusters=300),
        {
            'alpha': [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 50.0, 100.0],
        }
    ),
    'ElasticNet': (
        ElasticNet(max_iter=5000),
        create_pipeline(cat_encoder='onehot', loc_clusters=300),
        {
            'alpha': [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 50.0, 100.0],
            'l1_ratio': [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 50.0, 100.0],
        }
    ),
}




cv = KFold(n_splits=4, shuffle=True, random_state=SEED)
rmse_scorer = make_scorer(lambda y_true, y_pred: mean_squared_error(y_true, y_pred, squared=False), greater_is_better=False)

for name, (model, feature_pipeline, param_grid) in models.items():
    print(f"\n=== {name} ({y_label}) ===")
    out_dir = f"artifacts/{y_label}-{name}"
    os.makedirs(out_dir, exist_ok=True)
    pipe = SklearnPipeline([
        ('features', feature_pipeline),
        ('reg', model)
    ])
    search = RandomizedSearchCV(
        pipe,
        param_distributions={"reg__" + k: v for k, v in param_grid.items()},
        n_iter=30,
        cv=cv,
        scoring=rmse_scorer,
        n_jobs=-1,
        verbose=1,
        random_state=SEED,
        refit=True,
        return_train_score=True
    )
    search.fit(X_trainval, y_trainval)
    best_params = search.best_params_
    print(f"Best params: {search.best_params_}")
    print(f"Best CV RMSE: {-search.best_score_:.4f}")


    # Best model
    best_model = search.best_estimator_
    joblib.dump(best_model, os.path.join(out_dir, 'best_model.joblib'))

    # Save test predictions
    preds_test = best_model.predict(X_test)
    y_true_test = inv_transform(y_test, X_test)
    y_pred_test = inv_transform(preds_test, X_test)
    pd.DataFrame({
        'id': X_test['id'].values,
        'y_true': y_true_test,
        'y_pred': y_pred_test
    }).to_csv(os.path.join(out_dir, 'predictions_test.csv'), index=False)

    # Save OOF predictions
    preds = np.zeros_like(y_trainval)
    for train_idx, val_idx in cv.split(X_trainval):
        fold_model = clone(best_model)
        fold_model.fit(X_trainval.iloc[train_idx], y_trainval[train_idx])
        preds[val_idx] = fold_model.predict(X_trainval.iloc[val_idx])

    pd.DataFrame({
        'id': X_trainval['id'].values,
        'y_true': inv_transform(y_trainval, X_trainval),
        'y_pred': inv_transform(preds, X_trainval)
    }).to_csv(os.path.join(out_dir, 'predictions_oof.csv'), index=False)

    # Save summary JSON with best_params, cv_results, and test metrics
    cv_results = pd.DataFrame(search.cv_results_).to_dict(orient='records')
    metrics = ModelExplainer.compute_metrics(None, y_true_test, y_pred_test)
    with open(os.path.join(out_dir, 'results_summary.json'), 'w') as f:
        summary = {'best_params': best_params, 'cv_results': cv_results, 'metrics': metrics}
        json.dump(summary, f, indent=2)

    # Explainability and reporting
    feature_names = best_model.named_steps['features'].get_feature_names_out()
    explainer = ModelExplainer(
        save_folder=out_dir,
        model=best_model,
        X_test=best_model.named_steps['features'].transform(X_test),
        y_test=y_test,
        feature_names=feature_names,
        model_name=name,
        inv_transform=inv_transform,
        X_raw=X_test
    )
    explainer.run_all(best_params, preds_test)

print("Done") 