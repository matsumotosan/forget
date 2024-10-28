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


DATASET_DIR = "forget_data"
CKPT_PATH = "forget_downloads/version_38/checkpoints/epoch=19-step=520.ckpt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = T.Compose(
    [
        # T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)


@st.cache_resource
def load_classifier():
    return FaceClassifier.load_from_checkpoint(CKPT_PATH).eval().to(device)


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


@st.cache_resource
def load_dataset():
    with st.spinner("Downloading LFW dataset"):
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
    img = Image.open(image)
    cropped = mtcnn(img)

    st.write("What the model sees")
    st.image(T.ToPILImage()(cropped))

    if cropped is None:
        st.error("Did not detect a face in the image.")
    return transform(cropped)


model = load_classifier()
mtcnn = load_mtcnn()
dataset = load_dataset()

if "rand_idx" not in st.session_state:
    st.session_state.rand_idx = random.sample(range(len(dataset.data)), 4)

st.title("Face Identity Prediction")
st.write(
    "The application for which we are interested in exploring "
    "machine unlearning is facial identification. The task inherently involves sensitive information "
    "(images of people) that individuals may or may not have consented to the use of. Furthermore, "
    "once an image of a person has been used in training a facial recognition system, it is unclear "
    "if and how the individual's image can be eliminated from the model. For this reason, we are "
    "interested in extending our findings from experimenting with the MNIST dataset to commonly used "
    "facial recognition benchmarks such as Labeled Faces in the Wild (LFW)."
)

st.write(
    "We chose to use the InceptionResnetV1 as our architecture for facial recognition given the ease "
    "with which it can be adapted for a variety of face recognition tasks. The images are first "
    "preprocessed by a Multitask Cascaded Convolutional Network (MTCNN) to detect and crop out the "
    "face in the image (if it exists) prior to being passed to the InceptionResnetV1 model. We "
    "trained the model from scratch on a modified version of the LFW dataset for classification."
)

st.markdown(
    ":red[If you see the error `IndexError: list index out of range`, please try refreshing the page or "
    "interacting with a button on this page.]"
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
    data=probs[topk[:5]],
    x_label="Probability",
    horizontal=True,
)

st.success(f"Predicted Identity: **{idx_to_class[topk[0]]}**")

st.write(
    "Note: Our model is collapsing during training to predict the class with the largest number of samples (Colin Powell). "
    "We were unable to determine the cause of this but intend to address the issue for the final project."
)

st.subheader("Why train a model from scratch?")
st.write(
    "Although a pretrained model would provide us with a better starting point, "
    "we thought that it would interfere with our future investigations into the unlearning process as a finetuned "
    "model may retain additional knowledge compared to model we trained from scratch. "
    "As such, we chose a popular architecture for facial recognition and decided to train it from "
    "scratch as achieving state-of-the-art performance is not our end goal."
)
