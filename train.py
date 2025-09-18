import torch
from tqdm import tqdm
import argparse
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, RandomAffine
from efficentnet import EfficentNet
# from emotion_dataset import EmotionDataset
from torchmetrics.classification import MulticlassAccuracy
# from sklearn.metrics import accuracy_score
from vehicle_dataset import VehicleDataset


if __name__ == '__main__':
   
   # Check GPU
   if torch.cuda.is_available():
      device = torch.device('cuda')
      print(f"Using GPU: {torch.cuda.get_device_name(0)}.")
   else:
      device = torch.device('cpu')
      print('No GPU available, using CPU instead.')

   # Arg parser
   parser = argparse.ArgumentParser()
   parser.add_argument('--epochs', type=int, default=10)
   parser.add_argument('--batch_size', type=int, default=32)
   parser.add_argument('--dataset_path', type=str, default=r'D:\học\Tài Liệu CNTT\efficentNet\emotion_dataset')
   parser.add_argument('--load_weight_path', type=str, default='')
   parser.add_argument('--save_weight_path', type=str, default='')
   parser.add_argument('--version', type=str, default='b0')
   parser.add_argument('--num_classes', type=int, default=7)
   
   args = parser.parse_args()
   epochs = args.epochs
   batch_size = args.batch_size
   dataset_path = args.dataset_path
   load_weight_path = args.load_weight_path
   save_weight_path = args.save_weight_path
   version = args.version
   num_classes = args.num_classes
   
   # Input image size by version
   if version == 'b0':
      resize = (224, 224)
   elif version == 'b1':
      resize = (240, 240)
   elif version == 'b2':
      resize = (288, 288)
   elif version == 'b3':
      resize = (320, 320)
   elif version == 'b4':
      resize = (384, 384)
   elif version == 'b5':
      resize = (456, 456)
   elif version == 'b6':
      resize = (528, 528)
   elif version == 'b7':
      resize = (600, 600)
   print(f'Image size: {resize}')
   
   # Transform data
   train_transform= Compose(
      transforms=[
         RandomAffine(
            degrees=10,
            translate=(0.1,0.1),
            scale=(0.8,1.2),
            shear=10,
            
         ),
         Resize(resize),
         ToTensor(),
         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
      ]
   )
   
   test_transform= Compose(
      transforms=[
         Resize(resize),
         ToTensor(),
         Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
      ]
   )
   
   # # DataLoader
   # training_set = EmotionDataset(root = dataset_path, train=True, transform=train_transform)
   # test_set = EmotionDataset(root=dataset_path, train=False, transform=test_transform)

   training_set = VehicleDataset(root = dataset_path, train=True, transform=train_transform)
   test_set = VehicleDataset(root=dataset_path, train=False, transform=test_transform)
   
   training_loader = DataLoader(
      dataset=training_set,
      batch_size=batch_size,
      shuffle=True,
      num_workers=2,
      drop_last=False
   )
    
   test_loader = DataLoader(
      dataset=test_set,
      batch_size=batch_size,
      shuffle=False,
      num_workers=2,
      drop_last=False
   )
   num_iter = len(training_loader)
   num_iter_test = len(test_loader)
   # Model
   model = EfficentNet(version=version, num_classes=num_classes)
   if load_weight_path:
      model.load_state_dict(torch.load(load_weight_path))
   model.to(device)
   criterion = torch.nn.CrossEntropyLoss()
   # optimizer = torch.optim.SGD(model.parameters(), lr= 0.001, momentum=0.9, weight_decay=1e-4)
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
   # optimizer = torch.optim.RMSprop(model.parameters(), lr = 1e-4, momentum=0.9, weight_decay=0.9)
   accuracy_metric = MulticlassAccuracy(num_classes=num_classes).to(device)
   
   best_acc = 0.0 
   # Training
   for epoch in range(epochs):
      model.train()
      loss_total = 0.0
      train_tqdm = tqdm(training_loader)
      for iter, (image_tensor, label) in enumerate(train_tqdm):
         image_tensor, label = image_tensor.to(device), label.to(device)
         # Predict
         output = model(image_tensor)
         # Compute loss
         loss_value = criterion(output, label)      
         loss_total+=loss_value.item()
         # Clear old gradients
         optimizer.zero_grad()
         # Compute new gradients (Back prog)
         loss_value.backward()
         # update weights
         optimizer.step()
         train_tqdm.set_description(f'Epoch [{epoch+1}/{epochs}], Iter[{iter+1}/{num_iter}], Loss: {(loss_value.item()):.5f}')
      
      print(f'Epoch [{epoch+1}/{epochs}], Loss_avg: {(loss_total/num_iter):.5f}')
      # Save weight
      torch.save(model.state_dict(), f'/{save_weight_path}/{version}-epoch{epoch+1}.pth')
      print(f'Save /{save_weight_path}/{version}-epoch{epoch+1}.pth successfully!')
      
      # Eval
      model.eval()
      accuracy_metric.reset()
      # accuracy_total = 0.0
      with torch.no_grad():
         test_tqdm = tqdm(test_loader)
         for iter, (image_tensor, label) in enumerate(test_tqdm):
            image_tensor, label = image_tensor.to(device), label.to(device)
            output = model(image_tensor)
            # probs = torch.softmax(output, dim=1)
            label_pred = output.argmax(dim=1)
            # accuracy_total += accuracy_score(label.cpu(), label_pred.cpu())
            accuracy_metric.update(label_pred,label)
      acc_avg = accuracy_metric.compute()
      # acc_avg = accuracy_total/num_iter_test
      # print(f'Epoch [{epoch+1}/{epochs}], Accuracy: {acc_avg.item():.3f}')
      print(f'Epoch [{epoch+1}/{epochs}], Accuracy: {acc_avg.item():.3f}')

      if acc_avg.item() > best_acc:
         best_acc = acc_avg.item()
         torch.save(model.state_dict(), f'/{save_weight_path}/{version}-epoch{epoch+1}-{str(round(best_acc, 3)).replace(".", "_")}.pth')
         print(f'Save best /{save_weight_path}/{version}-epoch{epoch+1}-{str(round(best_acc, 3)).replace(".", "_")}.pth successfully!')
         
      
      
   