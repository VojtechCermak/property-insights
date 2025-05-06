import os
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay
import json
import pandas as pd

class ModelExplainer:
    def __init__(self, save_folder, model, X_test, y_test, feature_names, model_name=None, inv_transform=None, X_raw=None, skip_feature_importance=False):
        self.save_folder = save_folder
        self.model = model
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = feature_names
        self.model_name = model_name or 'Model'
        self.inv_transform = inv_transform
        self.X_raw = X_raw
        self.skip_feature_importance = skip_feature_importance
        os.makedirs(self.save_folder, exist_ok=True)

    def compute_metrics(self, y_true, y_pred):
        rmse = mean_squared_error(y_true, y_pred, squared=False)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        medae = np.median(np.abs(y_pred - y_true))
        return dict(RMSE=rmse, MAPE=mape, MAE=mae, Median_AE=medae)

    def plot_feature_importance(self):
        reg = self.model.named_steps['reg']
        if hasattr(reg, 'feature_importances_'):
            importances = reg.feature_importances_
            indices = np.argsort(importances)[::-1][:20]
            plt.figure(figsize=(10, 6))
            plt.title(f"{self.model_name} Feature Importances (Top 20)")
            plt.bar(range(len(indices)), importances[indices], align='center')
            plt.xticks(range(len(indices)), [self.feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            out = os.path.join(self.save_folder, 'explain_importance.png')
            plt.savefig(out)
            plt.close()
            return out
        return None

    def plot_residuals(self, y_true, y_pred):
        residuals = y_true - y_pred
        plt.figure(figsize=(8, 5))
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.title(f'{self.model_name} Residuals (True - Predicted)')
        plt.xlabel('Residual')
        plt.ylabel('Count')
        plt.tight_layout()
        out = os.path.join(self.save_folder, 'explain_residuals.png')
        plt.savefig(out)
        plt.close()
        return out

    def plot_mape_by_segment(self, y_true, y_pred):
        bins = np.percentile(y_true, [0, 20, 40, 60, 80, 100])
        bin_ids = np.digitize(y_true, bins) - 1
        segment_mape = []
        for i in range(5):
            mask = bin_ids == i
            if np.any(mask):
                mape = mean_absolute_percentage_error(y_true[mask], y_pred[mask])
                segment_mape.append(mape)
            else:
                segment_mape.append(np.nan)
        plt.figure(figsize=(8, 5))
        plt.bar(range(1, 6), segment_mape)
        plt.xlabel('Price Segment (quintile)')
        plt.ylabel('MAPE')
        plt.title(f'{self.model_name} MAPE by Price Segment')
        plt.tight_layout()
        out = os.path.join(self.save_folder, 'explain_segments.png')
        plt.savefig(out)
        plt.close()
        return out

    def extreme_and_mid_errors_table(self, y_true, y_pred):
        if self.X_raw is None or 'id' not in self.X_raw.columns:
            return None
        pct_error = (y_pred - y_true) / y_true
        abs_pct_error = np.abs(pct_error)
        # Over-prediction: y_pred > y_true (pct_error > 0)
        over_idx = np.argsort(-pct_error)[:5]
        # Under-prediction: y_pred < y_true (pct_error < 0)
        under_idx = np.argsort(pct_error)[:5]
        median_abs = np.median(abs_pct_error)
        mid_idx = np.argsort(np.abs(abs_pct_error - median_abs))[:5]
        rows = []
        for idx in over_idx:
            rows.append({
                'id': self.X_raw.iloc[idx]['id'],
                'type': 'worst_over',
                'true_price': y_true[idx],
                'pred_price': y_pred[idx],
                'pct_diff': pct_error[idx]
            })
        for idx in under_idx:
            rows.append({
                'id': self.X_raw.iloc[idx]['id'],
                'type': 'worst_under',
                'true_price': y_true[idx],
                'pred_price': y_pred[idx],
                'pct_diff': pct_error[idx]
            })
        for idx in mid_idx:
            rows.append({
                'id': self.X_raw.iloc[idx]['id'],
                'type': 'mid',
                'true_price': y_true[idx],
                'pred_price': y_pred[idx],
                'pct_diff': pct_error[idx]
            })
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.save_folder, 'explain_examples.csv')
        df.to_csv(csv_path, index=False)
        return df

    def run_all(self, best_params, y_pred):
        # Inverse transform if needed
        if self.inv_transform is not None and self.X_raw is not None:
            y_true = self.inv_transform(self.y_test, self.X_raw)
            y_pred = self.inv_transform(y_pred, self.X_raw)
        else:
            y_true = self.y_test
            y_pred = y_pred
        metrics = self.compute_metrics(y_true, y_pred)
        if self.skip_feature_importance:
            fi_path = None
        else:
            fi_path = self.plot_feature_importance()
        res_path = self.plot_residuals(y_true, y_pred)
        mape_path = self.plot_mape_by_segment(y_true, y_pred)
        error_df = self.extreme_and_mid_errors_table(y_true, y_pred)
        self.save_html_report(best_params, metrics, [fi_path, res_path, mape_path], error_df)


    def save_html_report(self, best_params, metrics, artifact_paths, error_df):
        # Bootstrap CSS
        bootstrap = '<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">'
        # Metrics card
        metrics_html = ''.join([
            f'<li class="list-group-item"><b>{k}:</b> {v:.4f}</li>' for k, v in metrics.items()
        ])
        # Artifacts gallery
        gallery = ''
        for path in artifact_paths:
            if path and os.path.exists(path):
                fname = os.path.basename(path)
                gallery += f'<div class="mb-4"><img src="{fname}" class="img-fluid rounded shadow d-block mx-auto"></div>'
        # Error table
        if error_df is not None:
            table_html = error_df.to_html(index=False, classes="table table-striped table-bordered", float_format=lambda x: f'{x:,.2f}')
        else:
            table_html = '<p class="text-warning">No error analysis table available.</p>'
        # HTML
        html = f'''
        <html>
        <head>
        <meta charset="utf-8">
        <title>{self.model_name} Model Report</title>
        {bootstrap}
        </head>
        <body class="bg-light">
        <div class="container py-4">
            <h1 class="mb-4">{self.model_name} Model Report</h1>
            <div class="row mb-4">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-primary text-white">Best Parameters</div>
                        <div class="card-body"><pre>{json.dumps(best_params, indent=2)}</pre></div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header bg-success text-white">Test Metrics</div>
                        <ul class="list-group list-group-flush">{metrics_html}</ul>
                    </div>
                </div>
            </div>
            <h2 class="mb-3">Artifacts</h2>
            <div class="d-flex flex-column align-items-center">{gallery}</div>
            <h2 class="mt-4">Extreme and Mid Errors</h2>
            {table_html}
        </div>
        </body>
        </html>
        '''
        with open(os.path.join(self.save_folder, 'explain_summary_report.html'), 'w') as f:
            f.write(html)
        print(f"Summary report saved to {os.path.join(self.save_folder, 'explain_summary_report.html')}") 