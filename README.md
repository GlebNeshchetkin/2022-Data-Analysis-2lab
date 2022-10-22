Laboratory work 2
==============================

1. Bажная метрика при оценке модели - MSE (Mean Squared Error) для определения качетсва выполненной задачи регрессии.
2. Препроцессинг (\src\data\make_dataset.py)
3. Генерация признаков (\src\features\build_features.py)
4. Разделение данных train/val происходит перед обучением модели (\src\models\train_model.py).
5. В работе используются модели CatBoostRegression, XGBRegressor, LinearRegression (для каждой модели создан свой pipeline: src\models\catboost_model.py, src\models\xgb_regression_model.py, src\models\linear_regression_model.py).
6. Для модели CatBoost используется метод для работы с категориальными признаками из category_encoders и category_encoders, выбор параметров осуществляется с помощью GridSearchCV.
7. Оценка модели (\src\models\evaluate.py). Значения метрик приведены в \reports\metrics.json для CatBoost и XGB моделей.
8. Предсказание (инференс) модели на новых данных (\src\models\predict_model.py). 
9. Результаты предсказания - в \data\prediction\prediction.csv.
10. dvc.yaml файл содержит все stages и команды.

A short description of the project.

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
