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

### Locally

To run the application natively, follow
[installation instructions](#installing-dependencies) to install the project
dependencies.

### Docker container

To run the application in a Docker container, execute the following command::

```sh
docker-compose up
```

## Installing dependencies

First, create your virtual environment using one of the following commands::

```sh
# uv
uv venv --python 3.11

# venv
python3.11 -m venv .venv
```

Once the virtual environment is created, activate the environment. Note that
the exact command may differ based on the shell being used.

```sh
# activate venv
source .venv/bin/activate

# activate venv (fish shell)
source .venv/bin/activate.fish
```

Install the project dependencies using `pip` or `uv` with one of the following commands::

```sh
# pip
pip install -r requirements.txt

# uv
uv pip install -r requirements.txt
```

### Updating dependencies

To add or edit the dependencies, edit the `requirements.txt` re-install the dependencies.

## Experiments

Experiments were run on a GPU????.

Results can be reproduced by executing the scripts as follows::

```sh
python experiments.py
```
