import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler, OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

# --- Top-level functions for FunctionTransformer ---
def identity(x):
    return x

def astype_int(x):
    return x.astype(int)

class MaxImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.fill_value = np.nanmax(X, axis=0)
        return self

    def transform(self, X):
        X = np.array(X)
        return np.where(np.isnan(X), self.fill_value, X)

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else np.array([])


class Log1pTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        X = np.clip(X, a_min=0, a_max=None)  # Replace negatives with 0
        return np.log1p(X)

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else np.array([])


class ClipTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, min_val=None, max_val=None):
        self.min_val = min_val
        self.max_val = max_val

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.array(X)
        return np.clip(X, self.min_val, self.max_val)

    def get_feature_names_out(self, input_features=None):
        return input_features if input_features is not None else np.array([])


class LatLonClustering(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=10, lat_col='latitude', lon_col='longitude', prefix='location_cluster', method='kmeans'):
        self.n_clusters = n_clusters
        self.lat_col = lat_col
        self.lon_col = lon_col
        self.prefix = prefix
        self.method = method
        self.model = None

    def fit(self, X, y=None):
        coords = X[[self.lat_col, self.lon_col]].dropna()

        if self.method == 'kmeans':
            self.model = KMeans(n_clusters=self.n_clusters, random_state=42, n_init='auto')
        elif self.method == 'gmm':
            self.model = GaussianMixture(n_components=self.n_clusters, random_state=42)
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

        self.model.fit(coords)
        return self

    def transform(self, X):
        X_copy = X.copy()
        mask = X_copy[[self.lat_col, self.lon_col]].notnull().all(axis=1)
        coords = X_copy.loc[mask, [self.lat_col, self.lon_col]]
        X_copy[self.prefix] = np.nan
        X_copy.loc[mask, self.prefix] = self.model.predict(coords)
        return X_copy[[self.prefix]]

    def get_feature_names_out(self, input_features=None):
        return np.array([self.prefix])


cols_nearest = [
    'nearest_small_shop',
    'nearest_tavern',
    'nearest_candy_shop',
    'nearest_movies',
    'nearest_theater',
    'nearest_vet',
    'nearest_playground',
    'nearest_atm',
    'nearest_post_office',
    'nearest_shop',
    'nearest_school',
    'nearest_bus_public_transport',
    'nearest_medic',
    'nearest_sports',
    'nearest_drugstore',
    'nearest_train',
    'nearest_restaurant',
    'nearest_kindergarten',
    'nearest_tram',
    'nearest_sightseeing',
    'nearest_natural_attraction',
    'nearest_metro',
]

cols_area = [
    'usableArea',
    'balconyArea',
    'cellarArea',
    'terraceArea',
    'loggiaArea',
    'basinArea',
    'gardenArea',
]

cols_cat = [
    'premise_name',
    'objectLocation',
    'furnished',
    'buildingCondition',
    'buildingType',
    'category_sub',
    'energyEfficiencyRating',
    'energyPerformanceCertificate',
    'objectAge',
    'acceptanceYear',
    'city_part',
    'district',
]

cols_bool = [
    'priceFlagNegotiationCb',
    'parkingLots',
    'lowEnergy',
    'garage',
    'elevator',
    'isBasement',
]

def create_pipeline(cat_encoder=None, loc_clusters=10):
    # Setup categorical encoder
    if cat_encoder is None:
        encoder_cat = FunctionTransformer(func=identity, feature_names_out='one-to-one')
    elif cat_encoder == 'onehot':
        encoder_cat = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    elif cat_encoder == 'ordinal':
        encoder_cat = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    else:
        raise ValueError(f'Unknown encoder: {cat_encoder}')


    pipeline = ColumnTransformer(transformers=[
        # NUMERIC
        ('premise_review_count', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('log', Log1pTransformer()),
            ('standard_scaler', StandardScaler()),
        ]), ['premise_review_count']),
    
        ('premise_review_score', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('minmax_scaler', MinMaxScaler()),
        ]), ['premise_review_score']),
        
        ('stats', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('log', Log1pTransformer()),
            ("standard_scaler", StandardScaler()),
        ]), ['stats']),
    
        ('undergroundFloors', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('clip', ClipTransformer(max_val=5)),
            ('minmax_scaler', MinMaxScaler()),
        ]), ['undergroundFloors']),
    
        ('parking', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('clip', ClipTransformer(max_val=5)),
            ('minmax_scaler', MinMaxScaler()),
        ]), ['parking']),
    
        ('floors', Pipeline([
            ('log', Log1pTransformer()),
            ('standard_scaler', StandardScaler()),
            ('impute', SimpleImputer(strategy='median')),
        ]), ['floors']),
    
        ('floorNumber', Pipeline([
            ('log', Log1pTransformer()),
            ('standard_scaler', StandardScaler()),
            ('impute', SimpleImputer(strategy='median')),
        ]), ['floorNumber']),
    
        ('sinceMonths', Pipeline([
            ('log', Log1pTransformer()),
            ('standard_scaler', StandardScaler()),
            ('impute', SimpleImputer(strategy='median')),
        ]), ['sinceMonths']),
    
        # NEAREST
        ('nearest', Pipeline([
            ('log', Log1pTransformer()),
            ('standard_scaler', StandardScaler()),
            ('impute', MaxImputer()),
        ]), cols_nearest),
    
        # AREA
        ('area', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
            ('log', Log1pTransformer()),
            ('standard_scaler', StandardScaler()),
        ]), cols_area),
    
        # BOOL
        ('bool', Pipeline([
            ('astype_int', FunctionTransformer(astype_int, feature_names_out='one-to-one')),
            ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ]), cols_bool),
    
        # CATEGORICAL
        ('categorical', Pipeline([
            ('impute', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder_cat', encoder_cat),
        ]), cols_cat),

        ('latlon_clustering', Pipeline([
            ('latlon_clustering', LatLonClustering(n_clusters=loc_clusters, prefix='location_cluster')),
        ]), ['latitude', 'longitude']),
    ])
    return pipeline

