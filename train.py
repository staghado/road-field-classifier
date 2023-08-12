import glob
import os
import time
import mlflow

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import transforms

from model import create_classifier
from dataset import ClassificationDataset, CLASS_MAP
from utils import random_split_paths
from config import *

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25, save_dir='./checkpoints'):
    since = time.time()

    best_model_path = os.path.join(save_dir, 'best.pt')

    torch.save(model.state_dict(), best_model_path)
    best_acc = 0.0

    # start run logging
    mlflow.start_run()
    mlflow.log_param("num_epochs", num_epochs)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  
            else:
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for sample in dataloaders[phase]:
                inputs, labels = sample['image'], sample['label']
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            mlflow.log_metric(f"{phase}_loss", epoch_loss)
            mlflow.log_metric(f"{phase}_acc", epoch_acc)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    mlflow.end_run()


if __name__ == '__main__':

    # model
    model = create_classifier(pretrained=True)
    model = model.to(DEVICE)

    # loss
    criterion = nn.CrossEntropyLoss()

    # optimizer
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    # Decay LR by a factor of GAMMA every STEP_SIZE epochs
    lr_schedule = lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # prepare the image paths
    image_paths = glob.glob(os.path.join(ROOT_DIR, 'roads', '*')) + glob.glob(os.path.join(ROOT_DIR, 'fields', '*'))
    train_paths, validation_paths = random_split_paths(image_paths, validation_ratio=VALIDATION_RATIO, random_seed=RANDOM_SEED)

    # init the train/val datasets
    train_transform = transforms.Compose([
        transforms.RandAugment(num_ops=NUM_OPS, magnitude=MAGNITUDE),
        transforms.RandomHorizontalFlip(),
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])
    train_dataset = ClassificationDataset(image_paths=train_paths, transform=train_transform)

    val_transform = transforms.Compose([
        transforms.Resize(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(MEANS, STDS)
    ])
    val_dataset = ClassificationDataset(image_paths=validation_paths, transform=val_transform)

    dataloaders = {
        'train': DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS),
        'val': DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
        }
    dataset_sizes = {
        'train': train_dataset.__len__(),
        'val': val_dataset.__len__()
        }

    # train the model
    train_model(model, criterion, optimizer, lr_schedule, dataloaders, dataset_sizes, device=DEVICE, num_epochs=NUM_EPOCHS, save_dir=SAVE_DIR)