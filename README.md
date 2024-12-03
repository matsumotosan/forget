# Forget

Machine unlearning for facial recognition.

## Introduction

Machine unlearning arises from concerns surrounding privacy, security, and the
"right to be forgotten" - a particularly critical concern for applications that
handle sensitive information. A naive retraining of models on datasets modified
to exclude the data to be forgotten or unlearned is infeasible for large
models. Machine unlearning is not limited to applications concerned with
privacy and security as the same techniques may be applied to address the
effects of outliers or outdated samples in the original training data.

## Running the application

### Installing dependencies

The following instructions assumes that a package manager such as `pip` or `uv`
is installed.

Create a virtual environment based on the package manager of your
choice:

```sh
# venv
python3.11 -m venv .venv

# uv
uv venv --python 3.11
```

Once the virtual environment is created, activate the environment. Note that
the exact command may differ based on the shell being used.

```sh
# activate venv
source .venv/bin/activate

# activate venv (fish shell)
source .venv/bin/activate.fish
```

Install the project dependencies using `pip` or `uv` with one of the following commands:

```sh
# pip
pip install -r requirements.txt

# uv
uv pip install -r requirements.txt
```

### Download model weights

Once the dependencies are installed, execute the following command in the root
directory of the project to download model weights.

```sh
gdown --fuzzy https://drive.google.com/file/d/1KYoa8KsXjCQSv6Ka1ep9v3q1oQLNe85h/view?usp=sharing
unzip app_data.zip -d ./app_data
```

If the command does not work, please try manually downloading the file and extracting the file into the directory `app_data/`

### Download MUFAC dataset

To run the facial recognition portion of the application, download the MUFAC dataset into the `app_data` directory with the following commands:

```sh
wget https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/EbMhBPnmIb5MutZvGicPKggBWKm5hLs0iwKfGW7_TwQIKg?download=1 -O mufac.zip
unzip mufac.zip -d ./app_data/mufac
```

The MUFAC dataset should be downloaded and extracted into the `app_data/mufac` directory.

### Running the application

Run the app by executing the command:

```sh
streamlit run app.py
```

If this is your first time running a Streamlit application, you may be asked if you would like to register
your email. You are not required to register your email for the application and may leave it blank.

**Note: If a page on the application seems stuck or is otherwise not working, please try refreshing the page
or interacting with a component on the page.**

### Updating dependencies

To add or edit the dependencies, edit the `requirements.txt` and re-install the dependencies.

## Experiments

Experiments were run on an 2023 Apple M3 Pro with 18 GB of memory.

Results can be reproduced by executing the scripts as follows:

```sh
# Unlearn MNIST
python unlearn_mnist.py

# Unlearn CIFAR-10
python unlearn_cifar.py

# Unlearn MUFAC
python unlearn_mufac.py
```

Training and evaluation logs are saved into a logging directory `./logs/<dataset>/<datetime>/`.

### Plotting results

All unlearning experiments are logged into the `log_dir`. The results can be used to plot the training curves by executing the following command:

```sh
python plot_results.py --exp_dir=<experiment_log_dir> --fig_dir=<figure_dir>
```
