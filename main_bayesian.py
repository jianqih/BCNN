from __future__ import print_function

import os
import argparse
# pip install torchsummary
# from pytorchsum.torchsummary.torchsummary import summary
from collections import OrderedDict

import torch
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F
import torch.nn as nn

import data
import utils
import metrics
import config_bayesian as cfg
from models.BayesianModels.Bayesian3Conv3FC import BBB3Conv3FC
from models.BayesianModels.BayesianAlexNet import BBBAlexNet
from models.BayesianModels.BayesianLeNet import BBBLeNet
import matplotlib.pyplot as plt

# CUDA settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



# def summary(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
#     result, params_info = summary_string(
#         model, input_size, batch_size, device, dtypes)
#     print(result)

#     return params_info


# def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
#     if dtypes == None:
#         dtypes = [torch.FloatTensor]*len(input_size)

#     summary_str = ''

#     def register_hook(module):
#         def hook(module, input, output):
#             class_name = str(module.__class__).split(".")[-1].split("'")[0]
#             module_idx = len(summary)

#             m_key = "%s-%i" % (class_name, module_idx + 1)
#             summary[m_key] = OrderedDict()
#             summary[m_key]["input_shape"] = list(input[0].size())
#             summary[m_key]["input_shape"][0] = batch_size
#             if isinstance(output, (list, tuple)):
#                 summary[m_key]["output_shape"] = [
#                     [-1] + list(o.size())[1:] for o in output
#                 ]
#             else:
#                 summary[m_key]["output_shape"] = list(output.size())
#                 summary[m_key]["output_shape"][0] = batch_size

#             params = 0
#             if hasattr(module, "weight") and hasattr(module.weight, "size"):
#                 params += torch.prod(torch.LongTensor(list(module.weight.size())))
#                 summary[m_key]["trainable"] = module.weight.requires_grad
#             if hasattr(module, "bias") and hasattr(module.bias, "size"):
#                 params += torch.prod(torch.LongTensor(list(module.bias.size())))
#             summary[m_key]["nb_params"] = params

#         if (
#             not isinstance(module, nn.Sequential)
#             and not isinstance(module, nn.ModuleList)
#         ):
#             hooks.append(module.register_forward_hook(hook))

#     # multiple inputs to the network
#     if isinstance(input_size, tuple):
#         input_size = [input_size]

#     # batch_size of 2 for batchnorm
#     x = [torch.rand(2, *in_size).type(dtype).to(device=device)
#          for in_size, dtype in zip(input_size, dtypes)]

#     # create properties
#     summary = OrderedDict()
#     hooks = []

#     # register hook
#     model.apply(register_hook)

#     # make a forward pass
#     # print(x.shape)
#     model(*x)

#     # remove these hooks
#     for h in hooks:
#         h.remove()

#     summary_str += "----------------------------------------------------------------" + "\n"
#     line_new = "{:>20}  {:>25} {:>15}".format(
#         "Layer (type)", "Output Shape", "Param #")
#     summary_str += line_new + "\n"
#     summary_str += "================================================================" + "\n"
#     total_params = 0
#     total_output = 0
#     trainable_params = 0
#     for layer in summary:
#         # input_shape, output_shape, trainable, nb_params
#         line_new = "{:>20}  {:>25} {:>15}".format(
#             layer,
#             str(summary[layer]["output_shape"]),
#             "{0:,}".format(summary[layer]["nb_params"]),
#         )
#         total_params += summary[layer]["nb_params"]

#         total_output += np.prod(summary[layer]["output_shape"])
#         if "trainable" in summary[layer]:
#             if summary[layer]["trainable"] == True:
#                 trainable_params += summary[layer]["nb_params"]
#         summary_str += line_new + "\n"

#     # assume 4 bytes/number (float on cuda).
#     total_input_size = abs(np.prod(sum(input_size, ()))
#                            * batch_size * 4. / (1024 ** 2.))
#     total_output_size = abs(2. * total_output * 4. /
#                             (1024 ** 2.))  # x2 for gradients
#     total_params_size = abs(total_params * 4. / (1024 ** 2.))
#     total_size = total_params_size + total_output_size + total_input_size

#     summary_str += "================================================================" + "\n"
#     summary_str += "Total params: {0:,}".format(total_params) + "\n"
#     summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
#     summary_str += "Non-trainable params: {0:,}".format(total_params -
#                                                         trainable_params) + "\n"
#     summary_str += "----------------------------------------------------------------" + "\n"
#     summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
#     summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
#     summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
#     summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
#     summary_str += "----------------------------------------------------------------" + "\n"
#     # return summary
#     return summary_str, (total_params, trainable_params)


def getModel(net_type, inputs, outputs, priors, layer_type, activation_type):
    if (net_type == 'lenet'):
        return BBBLeNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == 'alexnet'):
        return BBBAlexNet(outputs, inputs, priors, layer_type, activation_type)
    elif (net_type == '3conv3fc'):
        return BBB3Conv3FC(outputs, inputs, priors, layer_type, activation_type)
    else:
        raise ValueError('Network should be either [LeNet / AlexNet / 3Conv3FC')




def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)




def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(validloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
        valid_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)


def test_model(net, criterion, testloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    test_loss = 0.0
    accs = []

    for i, (inputs, labels) in enumerate(testloader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = torch.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1).data

        log_outputs = utils.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(testloader), beta_type, epoch, num_epochs)
        test_loss += criterion(log_outputs, labels, kl, beta).item()
        accs.append(metrics.acc(log_outputs, labels))

    return test_loss/len(testloader), np.mean(accs)


tra_acc_list = []
tes_acc_list = []
tra_loss_list = []
tes_loss_list = []
epoch_num_list = []
tra_kl_list = []

def run(dataset, net_type):

    # Hyper Parameter settings
    layer_type = cfg.layer_type
    activation_type = cfg.activation_type
    priors = cfg.priors

    train_ens = cfg.train_ens
    valid_ens = cfg.valid_ens
    test_ens = cfg.test_ens
    n_epochs = cfg.n_epochs
    lr_start = cfg.lr_start
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size
    beta_type = cfg.beta_type

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = getModel(net_type, inputs, outputs, priors, layer_type, activation_type).to(device)

    ckpt_dir = f'checkpoints/{dataset}/bayesian'
    ckpt_name = f'checkpoints/{dataset}/bayesian/model_{net_type}_{layer_type}_{activation_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = metrics.ELBO(len(trainset)).to(device)
    optimizer = Adam(net.parameters(), lr=lr_start)
    # lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    # valid_loss_max = np.Inf
    for epoch in range(n_epochs):  # loop over the dataset multiple times

        train_loss, train_acc, train_kl = train_model(net, optimizer, criterion, train_loader, num_ens=train_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        
        # valid_loss, valid_acc = validate_model(net, criterion, valid_loader, num_ens=valid_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)
        
        test_loss, test_acc = test_model(net, criterion, valid_loader, num_ens=test_ens, beta_type=beta_type, epoch=epoch, num_epochs=n_epochs)

        tra_acc_list.append(train_acc)
        tes_acc_list.append(test_acc)
        # lr_sched.step(valid_loss)
        tra_loss_list.append(train_loss)
        tes_loss_list.append(test_loss)
        epoch_num_list.append(epoch)
        tra_kl_list.append(train_kl)

        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tTest Loss: {:.4f} \tTest Accuracy: {:.4f} \ttrain_kl_div: {:.4f}'.format(
            epoch, train_loss, train_acc, test_loss, test_acc, train_kl))

        # save model if validation accuracy has increased
        # if valid_loss <= valid_loss_max:
        #     print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        #         valid_loss_max, valid_loss))
        #     torch.save(net.state_dict(), ckpt_name)
        #     valid_loss_max = valid_loss
    print(len(tra_loss_list),len(tes_loss_list))
    print(tra_loss_list)
    print(tes_loss_list)
    # test_data = enumerate(test_loader)
    # _, (images, _) = next(test_data)
    # print(summary(net().to(device), (1,32,32)))
    # cm_plot(test_loader,net)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "PyTorch Bayesian Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='FAMNIST', type=str, help='dataset = [FAMNIST/MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
    plt.figure()
    plt.plot(tra_acc_list)
    plt.plot(tes_acc_list)
    plt.legend(['Train Accuracy', 'Test Accuracy'])
    plt.xlabel('epochs')
    plt.ylabel('accuracy of model')
    plt.savefig("figure/acc_bay_ep50.png")
    plt.clf()
    plt.figure()
    plt.plot(tra_loss_list)
    plt.plot(tes_loss_list)
    plt.legend(['Train Loss', 'Test Loss'])
    plt.xlabel('epochs')
    plt.ylabel('Log Loss')
    plt.savefig("figure/loss_bay_ep50.png")
    plt.clf()
    plt.figure()
    plt.plot(tra_kl_list)
    plt.savefig("figure/tra_kl_bay_ep50.png")
