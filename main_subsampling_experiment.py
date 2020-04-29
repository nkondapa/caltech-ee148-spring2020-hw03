from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt

import os

'''
This code is adapted from two sources:
(i) The official PyTorch MNIST example (https://github.com/pytorch/examples/blob/master/mnist/main.py)
(ii) Starter code from Yisong Yue's CS 155 Course (http://www.yisongyue.com/courses/cs155/2020_winter/)
'''


class fcNet(nn.Module):
    '''
    Design your model with fully connected layers (convolutional layers are not
    allowed here). Initial model is designed to have a poor performance. These
    are the sample units you can try:
        Linear, Dropout, activation layers (ReLU, softmax)
    '''

    def __init__(self):
        # Define the units that you will use in your model
        # Note that this has nothing to do with the order in which operations
        # are applied - that is defined in the forward function below.
        super(fcNet, self).__init__()
        self.fc1 = nn.Linear(in_features=784, out_features=20)
        self.fc2 = nn.Linear(20, 10)
        self.dropout1 = nn.Dropout(p=0.5)

    def forward(self, x):
        # Define the sequence of operations your model will apply to an input x
        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output


class ConvNet(nn.Module):
    '''
    Design your model with convolutional layers.
    '''

    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 3), stride=1)
        self.conv2 = nn.Conv2d(8, 8, 3, 1)
        self.dropout1 = nn.Dropout2d(0.5)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(200, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


class Net(nn.Module):
    '''
    Build the best MNIST classifier.
    '''

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=(3, 3), stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(3, 3), stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(10)
        self.dropout1 = nn.Dropout2d(0.5)
        self.max_pool1 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(4000, 256)
        self.fc2 = nn.Linear(256, 10)
        self.print_on = False

    def my_print(self, p):
        if self.print_on:
            print(p)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.bn1(x)
        x = self.max_pool1(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    '''
    This is your training function. When you call this function, the model is
    trained for 1 epoch.
    '''
    model.train()  # Set the model to training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()  # Clear the gradient
        output = model(data)  # Make predictions
        loss = F.nll_loss(output, target)  # Compute loss
        loss.backward()  # Gradient computation
        optimizer.step()  # Perform a single optimization step
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, set_name='Test'):
    model.eval()  # Set the model to inference mode
    test_loss = 0
    correct = 0
    incorrect = 0
    total = 0
    with torch.no_grad():  # For the inference step, gradient is not computed
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += len(output)
            incorrect += len(data) - pred.eq(target.view_as(pred)).sum().item()

    test_loss /= total

    print('\n' + set_name + ' set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total,
        100. * correct / total))

    return test_loss


def main():
    # Training settings
    # Use the command line to modify the default settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--step', type=int, default=1, metavar='N',
                        help='number of epochs between learning rate reductions (default: 1)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--evaluate', action='store_true', default=False,
                        help='evaluate your model on the official test set')
    parser.add_argument('--load-model', type=str,
                        help='model file path')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    # Evaluate on the official test set
    if args.evaluate:
        assert os.path.exists(args.load_model)

        # Set the test model
        model = Net().to(device)
        model.load_state_dict(torch.load(args.load_model))

        test_dataset = datasets.MNIST('../data', train=False,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize((0.1307,), (0.3081,))
                                      ]))

        # Pytorch has default MNIST dataloader which loads data at each iteration
        train_dataset_augmented = datasets.MNIST('../data', train=True, download=True,
                                                 transform=transforms.Compose([  # Data preprocessingls
                                                     transforms.RandomRotation(degrees=(-15, 15)),
                                                     transforms.ToTensor(),  # Add data augmentation here
                                                     transforms.Normalize((0.1307,), (0.3081,))
                                                 ]))
        train_loader = torch.utils.data.DataLoader(
            train_dataset_augmented, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(range(len(train_dataset_augmented)))
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.test_batch_size, shuffle=True, **kwargs)

        test(model, device, test_loader)
        test(model, device, train_loader, 'Train')

        return

    # Pytorch has default MNIST dataloader which loads data at each iteration
    train_dataset = datasets.MNIST('../data', train=True, download=True,
                                   transform=transforms.Compose([  # Data preprocessingls
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.1307,), (0.3081,))
                                   ]))

    train_dataset_augmented = datasets.MNIST('../data', train=True, download=True,
                                             transform=transforms.Compose([  # Data preprocessingls
                                                 transforms.RandomRotation(degrees=(-15, 15)),
                                                 transforms.ToTensor(),  # Add data augmentation here
                                                 transforms.Normalize((0.1307,), (0.3081,))
                                             ]))

    # display augmentation
    # index = 0
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(np.squeeze(np.array(train_dataset[index][0])))
    # plt.subplot(122)
    # plt.imshow(np.squeeze(np.array(train_dataset_augmented[index][0])))
    # plt.show()

    subsamples = [0.5, 0.25, 0.125, 0.0625]
    dsizes = []
    final_train_losses = []
    final_test_losses = []
    for ss in subsamples:
        # You can assign indices for training/validation or use a random subset for
        # training by using SubsetRandomSampler. Right now the train and validation
        # sets are built from the same indices - this is bad! Change it so that
        # the training and validation sets are disjoint and have the correct relative sizes.
        np.random.seed(10)
        sort_dict = dict(zip(range(10), [[], [], [], [], [], [], [], [], [], [], []]))
        for c, targ in enumerate(train_dataset.targets):
            sort_dict[targ.item()].append(c)

        subset_indices_valid = []
        subset_indices_train = []
        tot_size = 0
        for key in sort_dict.keys():
            msize = int(round(len(sort_dict[key]) * ss))
            tot_size += msize
            ss_size = int(round(msize * 0.15))
            shuffled_inds = np.random.permutation(sort_dict[key])
            subset_indices_valid.extend(shuffled_inds[0:ss_size])
            subset_indices_train.extend(shuffled_inds[ss_size:msize])
        dsizes.append(tot_size)


        # check balance
        # a = np.histogram(train_dataset.targets[subset_indices_valid], bins=np.linspace(-0.5, 9.5, 11))
        # b = np.histogram(train_dataset.targets[subset_indices_train], bins=np.linspace(-0.5, 9.5, 11))
        # print(a[0] / (a[0] + b[0]))

        train_loader = torch.utils.data.DataLoader(
            train_dataset_augmented, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(subset_indices_train)
        )
        val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size,
            sampler=SubsetRandomSampler(subset_indices_valid)
        )

        # Load your model [fcNet, ConvNet, Net]
        # model = fcNet().to(device)
        # model = ConvNet().to(device)
        model = Net().to(device)

        # Try different optimzers here [Adam, SGD, RMSprop]
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

        # Set your learning rate scheduler
        scheduler = StepLR(optimizer, step_size=args.step, gamma=args.gamma)

        # Training loop
        train_losses = []
        test_losses = []
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            train_loss = test(model, device, train_loader, 'Train')
            test_loss = test(model, device, val_loader)
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            scheduler.step()  # learning rate scheduler

            # You may optionally save your model at each epoch here

        if args.save_model:
            torch.save(model.state_dict(), "mnist_model_subsample_size=" + str(ss) + ".pt")

        final_train_loss = test(model, device, train_loader, 'Train')
        final_test_loss = test(model, device, val_loader)
        final_train_losses.append(final_train_loss)
        final_test_losses.append(final_test_loss)
    # Plotting
    plt.title('Reduced Dataset Experiment')
    plt.plot(np.log(dsizes), np.log(final_train_losses))
    plt.plot(np.log(dsizes), np.log(final_test_losses), '--')
    plt.xlabel('Log Dataset Size')
    plt.ylabel('Log Loss')
    plt.legend(['train', 'test'])
    plt.savefig('./reduced_dataset_experiment')


if __name__ == '__main__':
    main()
