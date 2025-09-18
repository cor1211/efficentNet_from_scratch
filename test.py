import torch
import numpy as np
import PIL
from PIL import Image
import argparse
from torchvision.transforms import ToTensor, Resize, Normalize, Compose
from efficentnet import EfficentNet

# Read image, transform 
# Load model
# Predict

label_map={
   0:'bus',
   1:'family sedan',
   2:'fire engine',
   3:'heavy truck',
   4:'jeep',
   5:'minibus',
   6:'racing car',
   7:'SUV',
   8:'taxi',
   9:'truck'
}

parser = argparse.ArgumentParser()
parser.add_argument('--image_path', type = str, default=r'D:\học\Tài Liệu CNTT\efficentNet\vehicle_dataset\test\1b9471eefb6f3951f127beb69ef5a584.jpg')
parser.add_argument('--resize', type = int, default=224)
parser.add_argument('--version', type = str, default='b0')
parser.add_argument('--num_classes', type = int, default=10)
parser.add_argument('--weight_path', type = str, default=r'D:\học\Tài Liệu CNTT\efficentNet\weights\vehicle\b0\b0-epoch85-0_815.pth')
args = parser.parse_args()
version = args.version
weight_path = args.weight_path
num_classes = args.num_classes
image_path = args.image_path
resize = (args.resize, args.resize)

if torch.cuda.is_available():
   device = torch.device('cuda')
else:
   device = torch.device('cpu')
   
image = Image.open(image_path).convert('RGB')

transform = Compose(
   transforms=[
      Resize(resize),
      ToTensor(),
      Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5]),
   ]
)

input_tensor = transform(image).unsqueeze(0).to(device)
print(input_tensor.shape)

# initialize model
model = EfficentNet(version=version, num_classes=num_classes).to(device)
model.load_state_dict(torch.load(weight_path))
model.eval()

# predict
with torch.no_grad():
   output = model(input_tensor)
probs = (torch.softmax(output, 1))
idx = torch.argmax(probs, 1)
print(probs[0][idx.item()].item())
print(label_map[idx.item()])
