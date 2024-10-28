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

To run the application locally, follow
[installation instructions](#installing-dependencies) to first install the project
dependencies.

Once the dependencies are installed, execute the following command in the root
directory of the project to download model weights. The model weights will be
downloaded into `forget_downloads` under the root directory of the project.

```sh
gdown --folder https://drive.google.com/drive/folders/1Z2mHXy4C7AiCNd9W6qNsq1Spin5n-zDC
```

Run the app by executing the command:

```sh
streamlit run app.py
```

**Note: If a page on the application seems stuck or is otherwise not working, please try refreshing the page
or interacting with a component on the page.**

### Docker container

This section is not completed yet.

## Installing dependencies

The following instructions assumes that a package manager such as `pip` or `uv`
is installed. Create a virtual environment based on the package manager of your
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

### Updating dependencies

To add or edit the dependencies, edit the `requirements.txt` re-install the dependencies.

## Experiments

Experiments were run on an 2023 Apple M3 Pro with 18 GB of memory.

Results can be reproduced by executing the scripts as follows:

```sh
# Train MNIST classifier
python train_mnist.py

# Train face classifier
python train_face.py
```
