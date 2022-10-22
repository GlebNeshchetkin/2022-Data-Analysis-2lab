 # -*- coding: utf-8 -*-
import sys
sys.path.append('src/')
sys.path.append('src/data')
sys.path.append('src/features')
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from utils import save_as_pickle
from preprocess import preprocess_data
from featurization import make_featurization
import pandas as pd
import pickle
import numpy as np


@click.command()
@click.argument('input_data_filepath', type=click.Path(exists=True))
@click.argument('catboost_model_filepath', type=click.Path())
@click.argument('output_prediction_filepath', type=click.Path())
def main(input_data_filepath, catboost_model_filepath, output_prediction_filepath):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
     
    logger = logging.getLogger(__name__)
    logger.info('data prediction')
    
    data = pd.read_csv(input_data_filepath)
    
    data = preprocess_data(data)
    data = make_featurization(data)
    
    catboost_regression_model = pickle.load(open(catboost_model_filepath, 'rb'))
    preds_catboost_regression = catboost_regression_model.predict(data)
    
    np.savetxt(output_prediction_filepath, preds_catboost_regression, delimiter=",")
    


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

 
