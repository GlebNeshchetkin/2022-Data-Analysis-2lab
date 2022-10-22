 # -*- coding: utf-8 -*-
import sys
sys.path.append('src/')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import save_as_pickle
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, explained_variance_score, max_error
import pandas as pd
import pickle
import json


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path())
@click.argument('linear_model_filepath', type=click.Path())
@click.argument('catboost_model_filepath', type=click.Path())
@click.argument('xgb_model_filepath', type=click.Path())
@click.argument('output_metrics_filepath', type=click.Path())
def main(input_data_filepath, input_target_filepath, linear_model_filepath, catboost_model_filepath, xgb_model_filepath, output_metrics_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
     
    logger = logging.getLogger(__name__)
    logger.info('model evaluation')
    
    X_test = pd.read_pickle(input_data_filepath)
    y_test = pd.read_pickle(input_target_filepath)
    
    linear_regression_model = pickle.load(open(linear_model_filepath, 'rb'))
    preds_linear_regression = linear_regression_model.predict(X_test)
    
    catboost_regression_model = pickle.load(open(catboost_model_filepath, 'rb'))
    preds_catboost_regression = catboost_regression_model.predict(X_test)
    
    xgb_regression_model = pickle.load(open(xgb_model_filepath, 'rb'))
    preds_xgb_regression = xgb_regression_model.predict(X_test)
    
    
    linear_mae = mean_absolute_error(y_test, preds_linear_regression)
    linear_mse = mean_squared_error(y_test, preds_linear_regression)
    linear_r2 = r2_score(y_test, preds_linear_regression)
    linear_evs = explained_variance_score(y_test, preds_linear_regression)
    linear_me = max_error(y_test, preds_linear_regression)
    
    catboost_mae = mean_absolute_error(y_test, preds_catboost_regression)
    catboost_mse = mean_squared_error(y_test, preds_catboost_regression)
    catboost_r2 = r2_score(y_test, preds_catboost_regression)
    catboost_evs = explained_variance_score(y_test, preds_catboost_regression)
    catboost_me = max_error(y_test, preds_catboost_regression)
    
    xgb_r_mae = mean_absolute_error(y_test, preds_xgb_regression)
    xgb_r_mse = mean_squared_error(y_test, preds_xgb_regression)
    xgb_r_r2 = r2_score(y_test, preds_xgb_regression)
    xgb_r_evs = explained_variance_score(y_test, preds_xgb_regression)
    xgb_r_me = max_error(y_test, preds_xgb_regression)
    print(preds_xgb_regression)
    metrics_dictionary = {
        "Model 1 Name": "CatBoostRegression",
        "MAE": catboost_mae,
        "MSE": catboost_mse,
        "R2": catboost_r2,
        "Explained Varience Score": catboost_evs,
        "Max Error": catboost_me,
        "Model 2 Name": "LinearRegression",
        "MAE ": linear_mae,
        "MSE ": linear_mse,
        "R2 ": linear_r2,
        "Explained Varience Score ": linear_evs,
        "Max Error ": linear_me,
        "Model 3 Name": "XGBRegression",
        "MAE  ": str(xgb_r_mae),
        "MSE  ": str(xgb_r_mse),
        "R2  ": str(xgb_r_r2),
        "Explained Varience Score  ": str(xgb_r_evs),
        "Max Error  ": str(xgb_r_me)
    }
    
    
    
    json_object = json.dumps(metrics_dictionary, indent=18)
 
    # Writing to sample.json
    with open(output_metrics_filepath, "w") as outfile:
        outfile.write(json_object)



if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

 
