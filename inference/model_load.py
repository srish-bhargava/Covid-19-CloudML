import os
import torch
from torchvision import models
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as datasets
from flask import Flask, jsonify, request
import io
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

app = Flask(__name__)

port = 8001

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model = Net().to(device)

model.load_state_dict(torch.load("/tmp/saved_models.pth"))
model.eval()

def prepare_image(image, target_size):
    image = image.convert('L')
    # Resize the input image nad preprocess it.
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)
    # Convert to Torch.Tensor and normalize.
    image = transforms.Normalize((0.1307,), (0.3081,))(image)
    # Add batch_size axis.
    image = image[None]
    with torch.no_grad():
        molded_images = Variable(image)
        return molded_images

def mnist_prediction(image_bytes):
    image = prepare_image(image_bytes, target_size=(28, 28))
    preds = F.softmax(model(image), dim=1)
    results = torch.topk(preds.cpu().data, k=5, dim=1)
    print("HELLO")
    print(results)
    return results

@app.route('/predict', methods=["POST"]) #allow POST requests
def predict():
    if request.method == "POST":
        if request.files.get("image"):
            data = {"success": False}
            image=Image.open(request.files["image"])
            image = prepare_image(image, target_size=(28, 28))
            preds = F.softmax(model(image), dim=1)
            results = torch.topk(preds.cpu().data, k=5, dim=1)
            # print(results)
            data['predictions'] = list()

            for prob, label in zip(results[0][0], results[1][0]):
                r = {"label": label.item(), "probability": float(prob)}
                data['predictions'].append(r)
            data["success"] = True
            return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port)
