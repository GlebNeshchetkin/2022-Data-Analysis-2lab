from catboost import CatBoostRegressor
import config as cfg
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV
from category_encoders.count import CountEncoder

real_pipe = Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
    ]
)

cat_pipe = Pipeline([
    ('imputer', SimpleImputer(strategy='constant', fill_value='NA')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ]
)

preprocess_pipe = ColumnTransformer(transformers=[
    ('real_cols', real_pipe, cfg.REAL_COLS),
    ('cat_cols', cat_pipe, cfg.CAT_COLS),
    ('cat_bost_cols', CountEncoder(), cfg.CAT_COLS)
    ]
)

base_model = CatBoostRegressor(iterations=1000,
                          learning_rate=1,
                          depth=2)

rscv = GridSearchCV(
    estimator=base_model,
    param_grid={'learning_rate': [0.03, 0.1],
                'depth': [2, 4],
                'l2_leaf_reg': [0.2, 0.5],
                'model_size_reg': [0.5, 1]},
    scoring='explained_variance',
    cv=5,
    refit=True
)

catboost_regression_model = Pipeline([
    ('preprocess', preprocess_pipe),
    ('model', rscv)
    ]
)