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
import numpy as np
from sklearn.model_selection import train_test_split



def build_split(labels_path, threshold=10, test_size=0.1):
    """
    Создает 2 csv файла, содержащих nmid и лейблы поспличенного csv файла, содержащего все изображения и лейблы. 
    Такое разделение нужно для того, чтобы сохранялся баланс классов между трейном и валидацией

    Args:
        labels_path: Путь до дирректории, созданной при скачивании изображений в файле downloader.py
        threshold: Убирает из выборки классы, где счетчик объектов данных классов меньше порога
        test_size: Размер тестовой выборки

    Returns:
        X_train: Pandas DataFrame - тренировочный датафрейм
        X_test: Pandas DataFrame - валидационный датафрейм
        dict(str, int) - словарь, где ключ: название класса, а значение: закодированный номер класса
    """

    
    le = LabelEncoder()
    labels_frame = pd.read_csv(labels_path)
    labels = pd.Series(le.fit_transform(labels_frame['subject']))
    mask = labels.map(labels.value_counts()) > threshold
    labels = labels[mask]
    labels_frame = labels_frame[mask]

    X_train, X_test, y_train, y_test = train_test_split(labels_frame, labels, 
        stratify=labels, test_size=test_size)

    X_train['subject'] = y_train
    X_test['subject'] = y_test
    
    return X_train, X_test, dict(zip(labels_frame['subject'], labels))



class SubjectsDataset(Dataset):

    def __init__(self, image_path, dataframe, transform=None):
        """
        Args:
            image_path: Путь до папки с изображениями
            dataframe: Pandas DataFrame, содержащий nmid и имена классов
            transform(опционально): трансформации, которые будут проводиться с изображением
        """
        self.image_path = image_path
        self.transform = transform
    
        labels_frame = dataframe
        labels = dataframe['subject']
        nm = dataframe['nmid'].map(int).tolist()
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
                                    os.path.join(
                                        self.image_path, img_name
                                    )
                            ),
                            cv2.COLOR_BGR2RGB
                )
        label = self.values[idx]

        if self.transform:
            sample = {'image': self.transform(image),
                      'label': torch.from_numpy(np.array(label))            
            }
        else:
            sample = {'image': torch.Tensor(image), 'label': torch.from_numpy(np.array(label))}
        
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
    for i, data in tqdm(enumerate(train_loader), desc='train loop'):
        images, labels = data['image'], data['label']
        images, labels = images.to(device), labels.type(torch.long).to(device)
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
    for i, data in tqdm(enumerate(val_loader), desc='eval loop'):
        images, labels = data['image'], data['label']
        images, labels = images.to(device), labels.type(torch.long).to(device)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        eval_loss += loss.item()
    return eval_loss / batch_size

def full_train(model, train_loader, val_loader, optimizer, loss_fn, n_epochs, batch_size, device, save_path):
    train_losses, eval_losses = [], []
    best_eval_loss = 100000000000
    for _ in tqdm(range(n_epochs), desc='full_train'):
        train_loss = train_loop(model, train_loader, optimizer, loss_fn, batch_size, device)
        eval_loss = eval_loop(model, val_loader, loss_fn, batch_size, device)

        print("-----------------")
        print(train_loss)
        print(eval_loss)
        print("-----------------")
        
        if eval_loss < best_eval_loss:
            torch.save(model.state_dict(), save_path)
            best_eval_loss = eval_loss

        train_losses.append(train_loss)
        eval_losses.append(eval_loss)
    
    return model, train_losses, eval_losses


def main():
    paths = {
        'image_path': r'C:\Users\Buc.Nikita\Desktop\wb_images\data',
        'desc_path': r'C:\Users\Buc.Nikita\Desktop\wb_images\files_description.csv',
        'save_path': r'trained_subject_model.pth'
    }

    params = {
        'batch_size': 128,
        'lr': 0.001,
        'n_epochs': 5,
        'threshold': 10,
        'test_size': 0.1
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
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]    
    )

    train_df, test_df, class_names = build_split(
        labels_path=paths['desc_path'], 
        threshold=params['threshold'], 
        test_size=params['test_size']
    )

    n_classes = len(class_names)

    print(f'number of classes: {n_classes}')

    train_subject_dataset = SubjectsDataset(
        image_path=paths['image_path'],
        dataframe=train_df,
        transform=transform
    )

    test_subject_dataset = SubjectsDataset(
        image_path=paths['image_path'],
        dataframe=test_df,
        transform=transform
    )

    train_data_loader = DataLoader(
        train_subject_dataset, 
        batch_size=params['batch_size'],
        shuffle=True
    )

    test_data_loader = DataLoader(
        test_subject_dataset, 
        batch_size=params['batch_size'],
        shuffle=False
    )    

    model = ConvNet(n_classes=n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=params['lr'])

    model, train_losses, eval_losses = full_train(
        model=model, train_loader=train_data_loader, val_loader=test_data_loader,
        optimizer=optimizer, loss_fn=loss_fn, n_epochs=params['n_epochs'], 
        batch_size=params['batch_size'], device=device, save_path=paths['save_path']
    )

    plt.plot(range(train_losses), train_losses, label='train_loss')
    plt.plot(range(eval_losses), eval_losses, label='eval_loss')
    plt.legend()
    plt.grid(True)
    plt.show()

main()