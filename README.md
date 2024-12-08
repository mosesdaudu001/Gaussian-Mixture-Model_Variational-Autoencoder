<!-- ABOUT THE PROJECT -->
## Tabular Data Generator

Tabular data generator model based on the integration of Gaussian Mixture model and Variational Autoencoders as explained by Patricia et al in the [paper](https://arxiv.org/pdf/2404.08434). 

Dataset can be found [here](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset).


<!-- GETTING STARTED -->
## Getting Started
Follow these simple steps to make this project work on your local machine.

### Prerequisites
You should have the following installed on your machine:

* Ubuntu
* Python 3.8.0
* Packages in requirements.txt
  ```sh
  pip install -r requirements.txt
  ```

### Installation

Download the repo manually (as a .zip file) or clone it using Git.
   ```sh
   git clone https://github.com/mosesdaudu001/Gaussian-Mixture-Model_Variational-Autoencoder
   ```


<!-- USAGE EXAMPLES -->
## Usage

To train on the dataset, change `train` and `eval` at line 63, and 64 respectively to be `True`. To train without processing the data (converting continous variables to categorical variables), change `convert_cont_to_cat` in utils.py at line 94 to be `False`, while to train with processing, change it to be `True`

You can specify different configurations or training parameters in utils.py for both models.


To train/test the proposed model and show results, run the following command:
   ```sh
   python data_generation/main_generator.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>
