import pandas as pd

def Extra_Amenities(data: pd.DataFrame) -> None:
    data['extra_amenities'] = 0
    for i in range(len(data['GarageType'])):
        if data['GarageType'][i] != 'No Garage' or data['PoolQC'][i] != 'No Pool':
            data['extra_amenities'][i] = 0
        else:
            data['extra_amenities'][i] = 1
            
def long_street(data: pd.DataFrame) -> None:
    data['long_street'] = 0
    for i in range(len(data['LotFrontage'])):
        if data['LotFrontage'][i] > 80:
            data['long_street'][i] = 0
        else:
            data['long_street'][i] = 1
            
def Easy_Access(data: pd.DataFrame) -> None:
    data['easy_access'] = 0
    for i in range(len(data['Street'])):
        if data['Street'][i] == 'Pave' and data['Alley'][i] == 'Pave' and data['LandContour'][i] == 'Lvl':
            data['easy_access'][i] = 0
        else:
            data['easy_access'][i] = 1
            
def make_featurization(data: pd.DataFrame) -> pd.DataFrame:
    Extra_Amenities(data)
    long_street(data)
    Easy_Access(data)
    return data