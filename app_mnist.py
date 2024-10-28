import random
import numpy as np
import torch
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_image_select import image_select
from mnist_classifier import MNISTClassifier
from torchvision import transforms as T
from torchvision.datasets import MNIST


CKPT_PATH = "forget_downloads/version_43/checkpoints/epoch=2-step=5157.ckpt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
transform = T.Compose(
    [
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
    ]
)


def load_classifier():
    return MNISTClassifier.load_from_checkpoint(CKPT_PATH).eval().to(device)


model = load_classifier()

st.title("Digit Recognition")
st.write("For our final project, we are interested in exploring ")

# Test on MNIST test set
mnist_dataset = MNIST(root="data", train=False, download=True)

if "rand_idx" not in st.session_state:
    st.session_state.rand_idx = random.sample(range(len(mnist_dataset)), 4)
if st.button("Get new choices"):
    st.session_state.rand_idx = random.sample(range(len(mnist_dataset)), 4)

img = image_select(
    "Select an image to classify",
    [mnist_dataset[i][0] for i in st.session_state.rand_idx],
    use_container_width=False,
)

test_pred = model(transform(img).to(device))
test_probs = test_pred.softmax(dim=1).squeeze().cpu().detach().numpy()

st.bar_chart(
    data=test_probs,
    x_label="Probability",
    y_label="Digit",
    horizontal=True,
)

# Test on user drawing
st.header("Try your own handwritten digits")

canvas_result = st_canvas(
    fill_color="white",  # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color="white",
    background_color="black",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

img = Image.fromarray(canvas_result.image_data.astype("uint8"))
img = img.convert("L")
img = img.resize((28, 28))

st.write(
    "The image below shows the preprocessed image passed to the model for classification."
    "Note that the scale has been increased 10-fold for visualization purposes."
)

st.image(img, caption="Preprocessed Digit", width=280)

img = np.array(img) / 255.0
img = torch.tensor(img, dtype=torch.float32)
img = img.unsqueeze(0)
probs = model(img).softmax(dim=1).detach().numpy()

st.bar_chart(
    data=probs.squeeze(),
    x_label="Probability",
    y_label="Digit",
    horizontal=True,
)

st.subheader("Why MNIST?")
st.write(
    "Given the time constraints of our project, we thought it would be wise to test unlearning methods "
    "on a simple dataset like MNIST before moving on to larger datasets requiring larger models. We "
    'realize that MNIST is a "solved problem", however, we intend for the focus of our project to be '
    "on the machine unlearning aspect and believe MNIST will be a suitable first dataset for that."
)
