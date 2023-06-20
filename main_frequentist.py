from __future__ import print_function

import os
import argparse

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler
# from torch.utils.tensorboard import SummaryWriter
import data
import utils
import metrics
import config_frequentist as cfg
from models.NonBayesianModels.AlexNet import AlexNet
from models.NonBayesianModels.LeNet import LeNet
from models.NonBayesianModels.ThreeConvThreeFC import ThreeConvThreeFC
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn
import pandas as pd
# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def cm_plot(data_loader,net):
    y_pred = []
    y_true = []
    # iterate over test data
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        output = net(data) # Feed Network
        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        target = target.data.cpu().numpy()
        y_true.extend(target) # Save Truth

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], 
                         index = [i for i in class_names],
                         columns = [i for i in class_names])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('figure/cm_freq_ep200.png')


def getModel(net_type, inputs, outputs):
    if (net_type == 'lenet'):
        return LeNet(outputs, inputs)
    elif (net_type == 'alexnet'):
        return AlexNet(outputs, inputs)
    elif (net_type == '3conv3fc'):
        return ThreeConvThreeFC(outputs, inputs)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')


def train_model(net, optimizer, criterion, train_loader):
    train_loss = 0.0
    net.train()
    accs = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return train_loss, np.mean(accs)


def validate_model(net, criterion, valid_loader):
    valid_loss = 0.0
    net.eval()
    accs = []
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return valid_loss, np.mean(accs)

def test_model(net, criterion, test_loader):
    test_loss = 0.0
    net.eval()
    accs = []
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        test_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return test_loss, np.mean(accs)

tra_acc = []
tes_acc = []
tra_loss = []
tes_loss = []
epoch_num = []

def pred(data_loader,net):
    testiter = iter(data_loader)
    images, labels = testiter.next()
    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        preds = net(images)
    fig = plt.figure(figsize=(15, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    images_np = [i.mean(dim=0).cpu().numpy() for i in images]

    for i in range(50):
        ax = fig.add_subplot(5, 10, i + 1, xticks=[], yticks=[])
        ax.imshow(images_np[i], cmap=plt.cm.gray_r, interpolation='nearest')
        if labels[i] == torch.max(preds[i], 0)[1]:
            ax.text(0, 3, class_names[torch.max(preds[i], 0)[1]], color='blue')
        else:
            ax.text(0, 3, class_names[torch.max(preds[i], 0)[1]], color='red')
    plt.savefig("figure/pred_freq_ep200.png")

def run(dataset, net_type):

    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs).to(device)

    ckpt_dir = f'checkpoints/{dataset}/frequentist'
    ckpt_name = f'checkpoints/{dataset}/frequentist/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    # lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    # valid_loss_min = np.Inf
    # writer = SummaryWriter()
    for epoch in range(1, n_epochs+1):

        train_loss, train_acc = train_model(net, optimizer, criterion, train_loader)
        # valid_loss, valid_acc = validate_model(net, criterion, valid_loader)
        test_loss, test_acc = test_model(net, criterion, valid_loader)
        # lr_sched.step(valid_loss)
        # lr_sched.step(train_loss)
        tra_acc.append(train_acc)
        tes_acc.append(test_acc)
        
        train_loss = train_loss/len(train_loader.dataset)
        # valid_loss = valid_loss/len(valid_loader.dataset)
        test_loss = test_loss/len(test_loader.dataset)
        
        tra_loss.append(train_loss)
        tes_loss.append(test_loss)
        epoch_num.append(epoch)
        
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tTesting Loss: {:.4f} \tTesting Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, test_loss, test_acc))
        
        # writer.add_scalar(tag="Loss/train",scalar_value=train_loss,global_step=epoch)
        # writer.add_scalar(tag="Loss/test",scalar_value=test_loss,global_step=epoch)
        # writer.add_scalar(tag="Accuracy/train",scalar_value=train_acc,global_step=epoch)
        # writer.add_scalar(tag="Accuracy/test",scalar_value=test_acc,global_step=epoch)
        # img_grid = torchvision.utils.make_grid(images)
        # # save model if validation loss has decreased
        # if valid_loss <= valid_loss_min:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        #         valid_loss_min, valid_loss))
        #     torch.save(net.state_dict(), ckpt_name)
        #     valid_loss_min = valid_loss
    # print(len(tra_loss),len(tes_loss))
    # print(tra_loss)
    # print(tes_loss)
    print(test_loader.data.shape)
    print(train_loader.data.shape)
    cm_plot(test_loader,net)
    pred(test_loader,net)
    # trainloader_iterator = iter(train_loader)
    # images, labels = trainloader_iterator.next()
    # SummaryWriter.add_graph(net,images)
    # writer.close()
    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Frequentist Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='FAMNIST', type=str, help='dataset = [FAMNIST/MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
    plt.figure()
    plt.plot(tra_acc)
    plt.plot(tes_acc)
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy of model')
    plt.savefig("figure/acc_freq_ep200.png")
    plt.clf()
    plt.figure()
    plt.plot(tra_loss)
    plt.plot(tes_loss)
    plt.legend(['Train Loss', 'Test Loss'])
    plt.xlabel('epochs')
    plt.ylabel('Log Loss')
    plt.savefig("figure/loss_preq_ep200.png")