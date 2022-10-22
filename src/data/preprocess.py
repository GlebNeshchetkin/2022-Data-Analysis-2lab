import pandas as pd
import numpy as np
import config as cfg

def fill_FireplaceQu(df: pd.DataFrame) -> pd.DataFrame:
    df['FireplaceQu'] = df['FireplaceQu'].fillna('No Fireplace')
    return df

def fill_Bsmt(df: pd.DataFrame) -> pd.DataFrame:
    df['BsmtQual'] = df['BsmtQual'].fillna('No Basement')
    df['BsmtCond'] = df['BsmtCond'].fillna('No Basement')
    df['BsmtExposure'] = df['BsmtExposure'].fillna('No Basement')
    df['BsmtFinType1'] = df['BsmtFinType1'].fillna('No Basement')
    df['BsmtFinType2'] = df['BsmtFinType2'].fillna('No Basement')
    return df

def fill_MasVnrArea(df: pd.DataFrame) -> pd.DataFrame:
    df['MasVnrArea'] = df['MasVnrArea'].fillna(0)
    return df

def fill_MasVnrType(df: pd.DataFrame) -> pd.DataFrame:
    df['MasVnrType'] = df['MasVnrType'].fillna('None')
    return df

def fill_Alley(df: pd.DataFrame) -> pd.DataFrame:
    df['Alley'] = df['Alley'].fillna('No alley access')
    return df

def fill_LotFrontage(df: pd.DataFrame) -> pd.DataFrame:
    most_freq = df['LotFrontage'].value_counts().index[0]
    df['LotFrontage'] = df['LotFrontage'].fillna(most_freq)
    return df

def fill_Garage(df: pd.DataFrame) -> pd.DataFrame:
    df['GarageType'] = df['GarageType'].fillna('No Garage')
    df['GarageFinish'] = df['GarageFinish'].fillna('No Garage')
    df['GarageQual'] = df['GarageQual'].fillna('No Garage')
    df['GarageCond'] = df['GarageCond'].fillna('No Garage')
    return df


def fill_Pool_Fence_Misc(df: pd.DataFrame) -> pd.DataFrame:
    df['PoolQC'] = df['PoolQC'].fillna('No Pool')
    df['Fence'] = df['Fence'].fillna('No Fence')
    df['MiscFeature'] = df['MiscFeature'].fillna('None')
    return df
    

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df = fill_LotFrontage(df)
    df = fill_Alley(df)
    df = fill_MasVnrType(df)
    df = fill_MasVnrArea(df)
    df = fill_Bsmt(df)
    df = fill_FireplaceQu(df)
    df = fill_Garage(df)
    df = fill_Pool_Fence_Misc(df)
    df = cast_types(df)
    return df

def cast_types(df: pd.DataFrame) -> pd.DataFrame:
    df['Id'] = df['Id'].astype(np.int8)
    df[cfg.CAT_COLS] = df[cfg.CAT_COLS].astype('category')
    df[cfg.REAL_COLS] = df[cfg.REAL_COLS].astype(np.float32)
    return df

def preprocess_target(df: pd.DataFrame) -> pd.DataFrame:
    df[cfg.TARGET_COL] = df[cfg.TARGET_COL].astype(np.int32)
    return df


def extract_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df, target = df.drop(cfg.TARGET_COL, axis=1), df[cfg.TARGET_COL].copy()
    return df, target
