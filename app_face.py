import torch
import random
import numpy as np
import streamlit as st
from PIL import Image
from face_classifier import FaceClassifier
from lfw_classification_dataset import LFWClassificationDataset
from facenet_pytorch import MTCNN
from torchvision import transforms as T
from streamlit_image_select import image_select
from torch.nn.functional import softmax


DATASET_DIR = "./data"
CKPT_PATH = "lightning_logs/version_38/checkpoints/epoch=19-step=520.ckpt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = T.Compose(
    [
        # T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


def load_classifier():
    model = FaceClassifier.load_from_checkpoint(CKPT_PATH).eval().to(device)
    return model


def load_mtcnn():
    mtcnn = MTCNN(
        image_size=160,
        margin=0,
        min_face_size=20,
        thresholds=[0.6, 0.7, 0.7],
        factor=0.709,
        post_process=True,
        device=device,
    )
    return mtcnn


def load_dataset():
    dataset = LFWClassificationDataset(
        root=DATASET_DIR,
        split="train",
        download=True,
        people=["Ariel_Sharon", "Colin_Powell", "George_W_Bush"],
        transform=transform,
        preprocessed=False,
    )
    return dataset


def preprocess_image(image):
    print(image)
    img = Image.open(image)
    cropped = mtcnn(img)
    st.image(T.ToPILImage()(cropped))
    if cropped is None:
        st.error("Did not detect a face in the image.")
    return transform(cropped)


model = load_classifier()
mtcnn = load_mtcnn()
dataset = load_dataset()

if 'rand_idx' not in st.session_state:
    st.session_state.rand_idx = random.sample(range(len(dataset.data)), 4)

st.title("Face Identity Prediction")
st.write(
    "For our final project, we are interested in exploring "
    "machine unlearning for facial identification. Machine unlearning refers "
    "to the idea of remove a specific subset of data points from a model whilst maintaining "
    "the model's performance on the remaining data. This concept has gained traction recently "
    "admist growing concerns about privacy and security. They are also of great interest "
    "from an engineering perspective as they could be used to mitigate the effects of outliers and "
    "incorrect data after the model has been trained - an increasingly expensive process."
)

if st.button("Get new choices"):
    st.session_state.rand_idx = random.sample(range(len(dataset.data)), 4)

img = image_select(
    "Select an image to classify",
    [dataset.data[i] for i in st.session_state.rand_idx],
    use_container_width=False,
)

idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
preprocessed_img = preprocess_image(img)
logits = model(preprocessed_img.unsqueeze(0)).squeeze()
probs = softmax(logits, dim=0).detach().numpy()
topk = np.argsort(probs)

st.bar_chart(
    # x=[idx_to_class[k] for k in topk],
    data=probs[topk[:5]],
    x_label="Probability",
    horizontal=True,
)

st.success(f"Predicted Identity: **{idx_to_class[topk[0]]}**")

st.subheader("Why train a model from scratch?")
st.write(
    "Although a pretrained model would provide us with a better starting point, "
    "we thought that it would interfere with our future investigations into the unlearning process. "
    "As such, we chose a popular architecture for facial recognition and decided to train it from scratch as performance is not our end goal."
)

st.subheader("Why isn't the model trained on the entire dataset?")
st.write(
    "The LFW dataset was originally intended for facial verification instead of facial identification. "
    ""
)
