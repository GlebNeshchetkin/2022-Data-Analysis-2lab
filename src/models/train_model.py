# -*- coding: utf-8 -*-
import sys
sys.path.append('src/')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import save_as_pickle
from catboost_model import catboost_regression_model
from linear_regression_model import linear_regression_model
from xgb_regression_model import xgb_regression_model
import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('input_target_filepath', type=click.Path())
@click.argument('output_model_catboost_filepath', type=click.Path())
@click.argument('output_model_linear_filepath', type=click.Path())
@click.argument('output_model_xgb_filepath', type=click.Path())
@click.argument('output_data_test_filepath', type=click.Path())
@click.argument('output_target_test_filepath', type=click.Path())
def main(input_data_filepath, input_target_filepath, output_model_catboost_filepath, output_model_linear_filepath, output_model_xgb_filepath, output_data_test_filepath, output_target_test_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('training models')
    
    data = pd.read_pickle(input_data_filepath)
    target = pd.read_pickle(input_target_filepath)
    
    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.4, random_state=7)
    
    catboost_regression_model.fit(X_train, y_train)
    save_as_pickle(catboost_regression_model, output_model_catboost_filepath)
    
    linear_regression_model.fit(X_train, y_train)
    save_as_pickle(linear_regression_model, output_model_linear_filepath)
    
    xgb_regression_model.fit(X_train, y_train)
    save_as_pickle(xgb_regression_model, output_model_xgb_filepath)

    save_as_pickle(X_test, output_data_test_filepath)
    save_as_pickle(y_test, output_target_test_filepath)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
