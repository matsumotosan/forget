from facenet_pytorch import InceptionResnetV1
from torchvision.datasets import LFWPeople
import matplotlib.pyplot as plt

model = InceptionResnetV1(pretrained='vggface2').eval()
test_dataset = LFWPeople(root="data", split="test", download=True)

print(test_dataset[0])

# print(model(test_dataset[0]))
