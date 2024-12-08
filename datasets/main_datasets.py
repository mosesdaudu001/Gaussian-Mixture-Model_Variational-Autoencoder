# Packages to import
import numpy as np
import pandas as pd

from .data_manager import DataManager
from sdv.datasets.demo import download_demo
from sdv.datasets.local import load_csvs
from sdv.metadata import Metadata


# Convert continuous variables to categorical, zero is a class and the rest of the elements are divided into 3
# classes
def cont2cat(df, cols, n=4):
    # Compute the bins for each column
    for col in cols:
        # Compute quartiles
        bins_aux = [df[col].quantile(i / n) for i in range(n + 1)]
        if len(bins_aux) == len(set(bins_aux)):
            df[col], bins = pd.qcut(df[col], q=n, retbins=True, duplicates='drop', labels=False)
        else:
            for i in range(1, len(bins_aux)):
                if bins_aux[i] < bins_aux[i - 1]:
                    bins_aux[i] = bins_aux[i - 1] + 0.0001
                elif bins_aux[i] == bins_aux[i - 1]:
                    bins_aux[i] = bins_aux[i] + 0.0001

            # Create interval index
            df[col] = pd.cut(df[col], bins_aux, labels=False, include_lowest=True, right=True)
            df[col] = df[col].astype('Int64')
            # column should have consecutive integers as categories
            # unique values
            df[col] = df[col].astype('category')
            # label encoding
            df[col] = df[col].astype('category').cat.codes
    return df


def preprocess_cardio_train(dataset_name, args):
   
    
    data_filename = args['input_dir'] + 'cardio_train/'
    data = load_csvs(folder_name=data_filename, read_csv_parameters = {'delimiter':';'})
    raw_df = data['cardio_train']
    
    # Drop irrelevant columns
    raw_df = raw_df.drop(labels=['id'], axis=1)
    
    if args['convert_cont_to_cat']:
        raw_df = cont2cat(raw_df, ['age', 'height', 'weight', 'ap_hi', 'ap_lo'])
        
    raw_metadata = Metadata.detect_from_dataframes(data)
    
    metadata_cols = raw_metadata.tables['cardio_train'].columns
    del metadata_cols['id']
    raw_metadata.tables['cardio_train'].columns = metadata_cols.copy()
    
    
    # Transform covariates and create df
    df = raw_df.copy()
    mapping_info = {}
    # # df['id'], classes = df['id'].factorize()
    # # mapping_info['id'] = np.array(classes.values)
    # # df['id'] = df['id'].replace(-1, np.nan)
    
    df['age'], classes = df['age'].factorize()
    mapping_info['age'] = np.array(classes.values)
    df['age'] = df['age'].replace(-1, np.nan)
    
    df['gender'], classes = df['gender'].factorize()
    mapping_info['gender'] = np.array(classes.values)
    df['gender'] = df['gender'].replace(-1, np.nan)
    
    df['height'], classes = df['height'].factorize()
    mapping_info['height'] = np.array(classes.values)
    df['height'] = df['height'].replace(-1, np.nan)
    
    df['weight'], classes = df['weight'].factorize()
    mapping_info['weight'] = np.array(classes.values)
    df['weight'] = df['weight'].replace(-1, np.nan)
    
    df['ap_hi'], classes = df['ap_hi'].factorize()
    mapping_info['ap_hi'] = np.array(classes.values)
    df['ap_hi'] = df['ap_hi'].replace(-1, np.nan)
    
    df['ap_lo'], classes = df['ap_lo'].factorize()
    # mapping_info['ap_lo'] = np.array(classes.values)
    # df['ap_lo'] = df['ap_lo'].replace(-1, np.nan)
    
    df['cholesterol'], classes = df['cholesterol'].factorize()
    # mapping_info['cholesterol'] = np.array(classes.values)
    # df['cholesterol'] = df['cholesterol'].replace(-1, np.nan)
    # df['cholesterol'] = df['cholesterol'].map({1: 0, 2: 1, 3: 2})
   
    
    df['gluc'], classes = df['gluc'].factorize()
    # mapping_info['gluc'] = np.array(classes.values)
    # df['gluc'] = df['gluc'].replace(-1, np.nan)
    # df['gluc'] = df['gluc'].map({1: 0, 2: 1, 3: 2})

    
    df['smoke'], classes = df['smoke'].factorize()
    mapping_info['smoke'] = np.array(classes.values)
    df['smoke'] = df['smoke'].replace(-1, np.nan)
    
    df['alco'], classes = df['alco'].factorize()
    mapping_info['alco'] = np.array(classes.values)
    df['alco'] = df['alco'].replace(-1, np.nan)
    
    df['active'], classes = df['active'].factorize()
    # mapping_info['active'] = np.array(classes.values)
    df['active'] = df['active'].replace(-1, np.nan)
    
    df['cardio'], classes = df['cardio'].factorize()
    # mapping_info['cardio'] = np.array(classes.values)
    df['cardio'] = df['cardio'].replace(-1, np.nan)
    
    # Create data manager object
    data_manager = DataManager(dataset_name, raw_df, df, mapping_info, raw_metadata=raw_metadata)

    feat_distributions = []
    for i in range(df.shape[1]):
        values = df.iloc[:, i].unique()
        no_nan_values = values[~pd.isnull(values)]
        if no_nan_values.size <= 2 and np.all(np.sort(no_nan_values).astype(int) ==
                                              np.array(range(no_nan_values.min().astype(int),
                                                             no_nan_values.min().astype(int) + len(no_nan_values)))):
            feat_distributions.append(('bernoulli', 1))
        elif np.amin(np.equal(np.mod(no_nan_values, 1), 0)):
            # Check if values are floats but don't have decimals and transform to int. They are floats because of NaNs
            if no_nan_values.dtype == 'float64':
                no_nan_values = no_nan_values.astype(int)
            if np.unique(no_nan_values).size < 50 and np.amin(no_nan_values) == 0:
                feat_distributions.append(('categorical', (np.max(no_nan_values) + 1).astype(int)))
            else:
                feat_distributions.append(('gaussian', 2))
        else:
            feat_distributions.append(('gaussian', 2))
    data_manager.set_feat_distributions(feat_distributions)

    # Normalize, impute data
    # Necessary to impute before normalization because of the categorical variables treated as gaussian.
    data_manager.norm_df = data_manager.transform_data(df)
    data_manager.imp_norm_df = data_manager.impute_data(data_manager.norm_df)

    # Create metadata for ctgan and tvae
    data_manager.get_metadata()

    return data_manager