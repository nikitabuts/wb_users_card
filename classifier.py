import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import os
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2



class SubjectsDataset(Dataset):

    def __init__(self, image_path, labels_path, transform=None):
        """
        Args:
            image_path: Путь до папки с изображениями
            labels_path: Путь до csv файла с метками
            transform(опционально): трансформации, которые будут проводиться с изображением
        """
        self.image_path = image_path
        self.labels_path = labels_path
        self.transform = transform
        
        le = LabelEncoder()
        labels_frame = pd.read_csv(self.labels_path)
        labels = le.fit_transform(labels_frame['subject'])
        nm = labels_frame['nmid'].map(int).tolist()
        vals = dict(zip(nm, labels))

        dataset = dict()
        for item in tqdm(os.listdir(self.image_path), desc='matching dataset'):
            id = int(item[6:14])    #Вычленяет nmid из имени изображения
            if id in nm:
                dataset[item] = vals[id]

        self.keys = list(dataset.keys())
        self.values = list(dataset.values())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        img_name = self.keys[idx]
        image = cv2.cvtColor(
                            cv2.imread(
                                    os.path.join(self.image_path, img_name)
                            ),
                            cv2.COLOR_BGR2RGB
                )
        label = self.values[idx]

        if self.transform:
            sample = {'image': self.transform(image),
                      'label': label            
            }
        else:
            sample = {'image': torch.Tensor(image), 'label': label}
        
        return sample


class ConvNet(nn.Module):
    
    def __init__(self, n_classes):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_loop(model, train_loader, optimizer, loss_fn, batch_size, device):
    model = model.train()
    train_loss = 0.0
    for i, data in tqdm(enumerate(train_loader, 0), desc='train'):
        images, labels = data['image'], data['label']
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / batch_size

def eval_loop(model, val_loader, loss_fn, batch_size, device):
    model = model.eval()
    eval_loss = 0.0
    for i, data in tqdm(enumerate(val_loader, 0), desc='validation'):
        images, labels = data['image'], data['label']
        outputs = model(images.to(device))
        loss = loss_fn(outputs.to(device), labels.to(device))
        eval_loss += loss.item()
    return eval_loss / batch_size

def full_train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, batch_size, device):
    logs = defaultdict()
    best_eval_loss = 100000000000
    for _ in range(n_epochs):
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, batch_size, device)
        eval_loss = eval_loop(model, val_loader, loss_fn, batch_size, device)
        
        if eval_loss < best_eval_loss:
            torch.save(model.state_dict())
            best_eval_loss = eval_loss

        logs['train_loss'].append(train_loss)
        logs['eval_loss'].append(eval_loss)
    
    return model, logs


def main():
    paths = {
        'image_path': r'C:\Users\Buc.Nikita\Desktop\wb_images\data',
        'desc_path': r'C:\Users\Buc.Nikita\Desktop\wb_images\files_description.csv'
    }

    params = {
        'n_classes': 10,
        'batch_size': 64,
        'lr': 0.001,
        'n_epochs': 5
    }

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    transform = transforms.Compose(
        [   
            transforms.ToTensor(),
            transforms.Normalize(
                                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
                            ),
            transforms.ToPILImage(),
            #transforms.Grayscale(1),
            transforms.ToTensor()
        ]    
    )

    subject_dataset = SubjectsDataset(
        image_path=paths['image_path'],
        labels_path=paths['desc_path'],
        transform=transform
    )

    data_loader = DataLoader(subject_dataset, batch_size=params['batch_size'],
                            shuffle=True)

    model = ConvNet(n_classes=params['n_classes']).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    model, logs = full_train(
        model=model, train_loader=data_loader, val_loader=data_loader,
        optimizer=optimizer, loss_fn=loss_fn, n_epochs=params['n_epochs'], 
        batch_size=params['batch_size'], device=device
    )

    plt.plot(range(len(logs['train_loss'])), logs['train_loss'])
    plt.show()