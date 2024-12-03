import random
import pandas as pd
import torch
import streamlit as st
from streamlit_image_select import image_select
from torchvision.datasets import CIFAR10
from unlearning_datamodule import cifar10_transform
from models import load_resnet18
from utils import cifar10_idx2class, cifar10_class2idx, read_json

APP_DIR = "app_data"

TRAINED_MODEL_PATH = f"{APP_DIR}/weights_resnet18_cifar10.pt"
UNLEARNED_FORGET_RETAIN_DIR = f"{APP_DIR}/2024-11-28-20-30-08"
UNLEARNED_FORGET_DIR = f"{APP_DIR}/2024-11-28-20-30-08"

N_CHOICES = 12

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
st.write(
    "The model has been tuned to forget `airplane` and `ship`. Select various images to see how the model's output changes"
    "based on the class. You should see that both the trained and unlearned models output similar probability distributions"
    "for classes not in the forget set. For classes in the forget set (`airplane` and `ship`), the model's output shifts "
    "noticeably. In this case, the predicted probability distribution is more evenly distributed among the classes indicating "
    "that the model is not confident in its predictions."
)

trained_model = load_model(TRAINED_MODEL_PATH).eval()
UNLEARNED_MODEL_PATH = f"{UNLEARNED_FORGET_RETAIN_DIR}/ckpt/epoch-20.pt"
unlearned_model = torch.load(UNLEARNED_MODEL_PATH, map_location=device).eval()

with st.spinner("Downloading dataset"):
    dataset = load_dataset()

if "rand_idx" not in st.session_state:
    st.session_state.rand_idx = random.sample(range(len(dataset)), N_CHOICES)
if st.button("Get new choices"):
    st.session_state.rand_idx = random.sample(range(len(dataset)), N_CHOICES)

img = image_select(
    "Select an image to classify",
    [dataset[i][0] for i in st.session_state.rand_idx],
    use_container_width=True,
)

x = cifar10_transform(img).unsqueeze(0).to(device)
trained_pred = trained_model(x)
trained_probs = trained_pred.softmax(dim=1).squeeze().cpu().detach().numpy()

unlearned_pred = unlearned_model(x)
unlearned_probs = unlearned_pred.softmax(dim=1).squeeze().cpu().detach().numpy()

df = pd.DataFrame(
    {
        "class": cifar10_class2idx.keys(),
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

# Show results (static)
st.header("Unlearning process")
st.write(
    "The performance of the model on the validation set was logged each unlearning epoch. "
    "The loss over the course of unlearning demonstrates that the model is forgetting the `airplane` and `ship` classes, "
    "whilst maintaining high accuracy on the remaining classes."
)

params = read_json(f"{UNLEARNED_FORGET_RETAIN_DIR}/params.json")
metrics = read_json(f"{UNLEARNED_FORGET_RETAIN_DIR}/metrics.json")

data = []
for epoch_idx, accuracies in enumerate(metrics["val_acc"]):
    for class_idx, accuracy in enumerate(accuracies):
        data.append(
            {
                "Unlearning Epoch": epoch_idx,
                "Class": f"{cifar10_idx2class[class_idx]}",
                "Accuracy": accuracy,
            }
        )

df = pd.DataFrame(data)
if params["forget_step"] and not params["retain_step"]:
    setting = "forget"
elif params["retain_step"] and not params["forget_step"]:
    setting = "retain"
elif params["retain_step"] and params["forget_step"]:
    setting = "forget+retain"
else:
    raise ValueError("Cannot have neither.")

st.line_chart(df, x="Unlearning Epoch", y="Accuracy", color="Class")
