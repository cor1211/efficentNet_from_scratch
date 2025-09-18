from torch.utils.data import Dataset
import os
from torchvision.transforms import Resize, Compose, RandomAffine, ToTensor, Normalize
import PIL
from PIL import Image

label_map = {
   'bus': 0,
   'family sedan':1,
   'fire engine':2,
   'heavy truck':3,
   'jeep':4,
   'minibus': 5,
   'racing car': 6,
   'SUV':7,
   'taxi':8,
   'truck':9
}

class VehicleDataset(Dataset):
   def __init__(self, root, train = True, transform = None):
      root = os.path.join(root, 'train' if train else 'val')
      self.image = []
      self.label = []
      self.transform = transform
      for folder in os.listdir(root):
         folder_path = os.path.join(root, folder)
         for image in os.listdir(folder_path):
            image_path = os.path.join(folder_path, image)
            self.image.append(image_path)
            self.label.append(label_map[folder])
   
   def __len__(self):
      return len(self.label)
   
   def __getitem__(self, idx):
      image_path, label = self.image[idx], self.label[idx]
      image = Image.open(image_path).convert('RGB')
      if self.transform:
         image = self.transform(image)
      
      return image, label
         
         
if __name__ == '__main__':
   train_transform= Compose(
      transforms=[
         RandomAffine(
            degrees=10,
            translate=(0.1,0.1),
            scale=(0.8,1.2),
            shear=10,
            
         ),
         Resize((224,224)),
         ToTensor(),
         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
      ]
   )
   train_set = VehicleDataset(root=r'D:\học\Tài Liệu CNTT\efficentNet\vehicle_dataset', train=True, transform=train_transform)
   print(f'Number of training samples: {len(train_set)}')