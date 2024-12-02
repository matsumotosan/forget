import random
import pandas as pd
import numpy as np
import torch
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_image_select import image_select
from torchvision.datasets import CIFAR10
from unlearning_datamodule import cifar10_transform
from models import load_resnet18
from utils import cifar10_class2idx

DATA_DIR = "app_data"
MODEL_DIR = "models"

TRAINED_MODEL_PATH = f"{MODEL_DIR}/weights_resnet18_cifar10.pt"
UNLEARNED_MODEL_PATH = f"{MODEL_DIR}/unlearned_resnet18_cifar10.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model(model_path):
    return load_resnet18(model_path, device)


@st.cache_resource
def load_dataset():
    return CIFAR10(root=DATA_DIR, train=False, download=True)


st.title("CIFAR-10 Image Classification")
st.header("Why CIFAR-10?")
st.write(
    "Given the time constraints of our project, we thought it would be wise to test unlearning methods "
    "on a simple dataset like CIFAR-10 before moving on to larger datasets requiring larger models. We "
    'realize that CIFAR-10 is a "solved problem", however, we intend for the focus of our project to be '
    "on the machine unlearning aspect and believe CIFAR-10 will be a suitable first dataset for that."
)

st.write(
    "With machine unlearning, we hope to train the CIFAR-10 classifier to unlearn or forget specific digits."
    "As mentioned above, the model may learn to give the illusion of unlearning a specific digit. One way we "
    "intend on testing this hypothesis is by evaluating the model's representation of images before the final layer."
)


st.header("Test on CIFAR-10 test set")
with st.spinner("Downloading dataset"):
    dataset = load_dataset()

if "rand_idx" not in st.session_state:
    st.session_state.rand_idx = random.sample(range(len(dataset)), 4)
if st.button("Get new choices"):
    st.session_state.rand_idx = random.sample(range(len(dataset)), 4)

img = image_select(
    "Select an image to classify",
    [dataset[i][0] for i in st.session_state.rand_idx],
    use_container_width=False,
)

st.image(img)

model = load_model(TRAINED_MODEL_PATH).eval()
x = cifar10_transform(img).unsqueeze(0).to(device)
test_pred = model(x)
test_probs = test_pred.softmax(dim=1).squeeze().cpu().detach().numpy()

df = pd.DataFrame({"Value": test_probs}, index=cifar10_class2idx.keys())

st.bar_chart(
    data=df,
    x_label="Probability",
    y_label="Class",
    horizontal=True,
)
