import random
import pandas as pd
import torch
import streamlit as st
from streamlit_image_select import image_select
from torchvision.datasets import CIFAR10
from unlearning_datamodule import cifar10_transform
from models import load_resnet18
from utils import cifar10_class2idx

APP_DIR = "app_data"

TRAINED_MODEL_PATH = f"{APP_DIR}/weights_resnet18_cifar10.pt"
UNLEARNED_MODEL_PATH = f"{APP_DIR}/2024-11-28-20-30-08_epoch-20.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@st.cache_resource
def load_model(model_path):
    return load_resnet18(model_path, device)


@st.cache_resource
def load_dataset():
    return CIFAR10(root=APP_DIR, train=False, download=True)


st.title("CIFAR-10 Image Classification")
st.header("Why CIFAR-10?")
st.write(
    "Given the time constraints of our project, we thought it would be wise to test unlearning methods "
    "on a simple dataset like CIFAR-10 before moving on to an unfamiliar dataset. We "
    'realize that CIFAR-10 is a "solved problem", however, we intend for the focus of our project to be '
    "on the machine unlearning aspect and believe CIFAR-10 will be a suitable first dataset for that."
)

st.write(
    "With machine unlearning, we hope to train the CIFAR-10 classifier to unlearn or forget specific digits."
    "As mentioned above, the model may learn to give the illusion of unlearning a specific class. One way we "
    "evaluated the extent to which a model unlearned a class was by performing a membership inference attack."
)


st.header("Test on CIFAR-10 test set")
with st.spinner("Downloading dataset"):
    dataset = load_dataset()

n_choices = 12
if "rand_idx" not in st.session_state:
    st.session_state.rand_idx = random.sample(range(len(dataset)), n_choices)
if st.button("Get new choices"):
    st.session_state.rand_idx = random.sample(range(len(dataset)), n_choices)

img = image_select(
    "Select an image to classify",
    [dataset[i][0] for i in st.session_state.rand_idx],
    use_container_width=False,
)

st.image(img, caption="Class", width=400)

# Trained model
trained_model = load_model(TRAINED_MODEL_PATH).eval()
x = cifar10_transform(img).unsqueeze(0).to(device)
trained_pred = trained_model(x)
trained_probs = trained_pred.softmax(dim=1).squeeze().cpu().detach().numpy()

df = pd.DataFrame({"Value": trained_probs}, index=cifar10_class2idx.keys())

st.header("Trained model output")
st.bar_chart(
    data=df,
    x_label="Probability",
    y_label="Class",
    horizontal=True,
)

# Unlearned model
st.header("Unlearned model output")
st.write("Model has been tuned to forget `airplane` and `ship`.")

unlearned_model = torch.load(UNLEARNED_MODEL_PATH, map_location=device).eval()
unlearned_pred = unlearned_model(x)
unlearned_probs = unlearned_pred.softmax(dim=1).squeeze().cpu().detach().numpy()

df = pd.DataFrame({"Value": unlearned_probs}, index=cifar10_class2idx.keys())

st.bar_chart(
    data=df,
    x_label="Probability",
    y_label="Class",
    horizontal=True,
)

# Show results (static)
st.header("Unlearning process")
st.write("The unlearning process was evaluated during each epoch.")

# MIA
st.header("Membership Inference Attack")
st.write("We will use membership inference attacks to evaluate the extent to which the model unlearned a class.")
