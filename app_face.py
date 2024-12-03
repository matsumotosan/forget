import torch
import pandas as pd
import random
import streamlit as st
from streamlit_image_select import image_select
from unlearning_datamodule import mufac_transform
from mufac_dataset import MUFAC
from utils import mufac_class2idx


APP_DIR = "app_data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EXPERIMENT_DIR = f"{APP_DIR}/2024-11-30-17-37-13"
TRAINED_MODEL_PATH = f"{EXPERIMENT_DIR}/ckpt/epoch-0.pt"
UNLEARNED_MODEL_PATH = f"{EXPERIMENT_DIR}/ckpt/epoch-20.pt"

N_CHOICES = 12


@st.cache_resource
def load_trained_model():
    return torch.load(TRAINED_MODEL_PATH, map_location=device)


@st.cache_resource
def load_unlearned_model():
    return torch.load(UNLEARNED_MODEL_PATH, map_location=device)


@st.cache_resource
def load_dataset():
    with st.spinner("Downloading MUFAC dataset"):
        dataset = MUFAC(
            root=APP_DIR,
            stage="test",
        )
    return dataset


trained_model = load_trained_model()
unlearned_model = load_unlearned_model()
dataset = load_dataset()

st.title("Face Attribute Classification")
st.write(
    "The application for which we are interested in exploring machine unlearning is facial attribute classification, specifically age classification. "
    "The task inherently involves sensitive information in the form of images of individuals "
    "that they may or may not have consented to the use of. Furthermore, "
    "once an image of a person has been used in training a facial recognition system, it is unclear "
    "if and how the individual's image can be eliminated from the model. For this reason, we are "
    "interested in extending our findings from our experiments with the CIFAR-10 dataset to the a facial recognition dataset."
)

st.write(
    "The facial recognition dataset that we are evaluating our unlearning strategy on is known as the Machine Unlearning for Facial Age Classification (MUFAC) dataset. "
    "It is comprises of over 13,000 facial images of Asian individuals of various ages. Each image is labeled with the picture individual's age and identity, "
    "allowing us to evaluate the effectiveness of both class- and instance-unlearning."
)

if "rand_idx" not in st.session_state:
    st.session_state.rand_idx = random.sample(range(len(dataset.data)), N_CHOICES)
if st.button("Get new choices"):
    st.session_state.rand_idx = random.sample(range(len(dataset.data)), N_CHOICES)

img = image_select(
    "Select an image to classify",
    [dataset[i][0] for i in st.session_state.rand_idx],
    use_container_width=False,
)

x = mufac_transform(img).unsqueeze(0).to(device)

trained_logits = trained_model(x)
trained_probs = trained_logits.softmax(dim=1).squeeze().cpu().detach().numpy()

unlearned_logits = unlearned_model(x)
unlearned_probs = unlearned_logits.softmax(dim=1).squeeze().cpu().detach().numpy()

df = pd.DataFrame(
    {
        "class": mufac_class2idx.keys(),
        "trained": trained_probs,
        "unlearned": unlearned_probs,
    }
)

st.bar_chart(
    data=df,
    x="class",
    y=["trained", "unlearned"],
    x_label="Class",
    y_label="Probability",
    horizontal=False,
    stack=False,
)

# Membership inference attack
st.header("Membership Inference Attack")
st.write(
    "Although we have some evidence that the model is successfully unlearning, it is unclear if the model "
    "has actually unlearned classes or if it is concealing its knowledge. Machine learning has demonstrated in numerous "
    "scenarios that it is capable of exploiting features in the data to learn shortcuts instead of the intended solution."
    "We used membership inference attack (MIA) to evaluate the extent to which a model scrubbed itself of memory of the classes intended to be forgotten.\n"
    "The idea behind an MIA is to train an attacker (in our case a simple logistic regressor) on the losses of the model on inputs that were and were not part of the original training dataset. "
    "If the attacker is able to distinguish between the two types of samples, that may be indicative of the model failing to adequately forget "
    "classes to be forgotten."
)

st.image("figs/mia_cifar10_unlearned.png")
