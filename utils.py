# Packages to import
import os
import pickle

from vae_bgm_inov.datasets.main_datasets import *
from datasets.medical_datasets import *


# Check if file exists
def check_file(path, msg, csv=False):
    if os.path.exists(path):
        if csv:
            return pd.read_csv(path)
        file = open(path, 'rb')
        results = pickle.load(file)
        file.close()
        return results
    else:
        raise FileNotFoundError(msg)


# Save dictionary to pickle file
def save(res, path):
    with open(path, 'wb') as handle:
        pickle.dump(res, handle, protocol=pickle.HIGHEST_PROTOCOL)


# Function that creates output directories for each task
def create_output_dir(task, args):
    for dataset_name in args['datasets']:
        if 'sota' in task:
            for model in args['models']:
                os.makedirs(args['sota_output_dir'] + dataset_name + os.sep + model + os.sep, exist_ok=True)
        else:
            for params in args['param_comb']:
                for seed in range(args['n_seeds']):
                    model_path = str(params['latent_dim']) + '_' + str(
                        params['hidden_size']) + os.sep + 'seed_' + str(seed)
                    os.makedirs(args['output_dir'] + dataset_name + os.sep + 'bgm' + os.sep + model_path + os.sep,
                                exist_ok=True)
                    if args['gauss']:
                        os.makedirs(args['output_dir'] + dataset_name + os.sep + 'gauss' + os.sep + model_path + os.sep,
                                    exist_ok=True)


# Function that sets environment configuration
def run_args():
    args = {}

    # Data
    args['datasets'] = ['cardio_train']
    print('[INFO] Datasets: ', 'cardio_train')

    # Path
    args['abs_path'] = os.path.dirname(os.path.abspath(__file__)) + os.sep

    # Missing data related parameters
    # Additional parameters related to model architecture
    args['model_mask'] = True  # Use missing info mask during training

    # Depending on the task, set the arguments
    args['input_dir'] = args['abs_path'] + 'datasets' + os.sep + 'raw_data' + os.sep

    # Training and testing configurations for savae and sota models
    args['train'] = True
    args['eval'] = True
    args['early_stop'] = True
    args['batch_size'] = 500
    args['n_epochs'] = 1000
    args['lr'] = 1e-3

    # Gaussian generation to compare with TVAE
    args['gauss'] = False

    # VAE hyperparameters
    args['n_threads'] = -1
    # args['n_threads'] = 5
    args['n_seeds'] = 15
    args['param_comb'] = [{'hidden_size': 50, 'latent_dim': 5}]

    # Generation parameters
    args['classifiers_list'] = ['RF']

    # Dataset nature (for validation)
    args['sa_datasets'] = ['']
    args['cl_datasets'] = ['cardio_train']

    # VAE
    args['train_vae'] = True
    args['output_dir'] = args['abs_path'] + 'data_generation' + os.sep + 'output_generator' + os.sep

    return args
