#!/usr/bin/env python
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from cnn import CNN, VGG
import configparser
from data.dataset import MFCCDataset
import os
from tensorboardX import SummaryWriter


def train(model, device, train_loader, test_loader, optimizer, epoch, metric, log_interval, save_path, log_path):
    model.train()
    acc_best, epoch_best = 0., 0
    batch_cnt = 0
    # loss_list, acc_list = [], []
    writer = SummaryWriter(log_path)
    for e in range(1, epoch + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = metric(output, target)
            loss.backward()
            optimizer.step()
            batch_cnt += 1
            # print('before loss', loss.item())
            # loss_list.append(loss.item() / len(data))
            writer.add_scalar('loss', loss.item() / len(data), batch_cnt)
            # print('after loss', loss.item())
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    e, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))
        acc = test(model, device, test_loader, metric)
        # print('before acc', acc)
        # acc_list.append(acc)
        writer.add_scalar('acc', acc, e)
        # print('after acc', acc)
        if acc > acc_best:
            ckpt = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            torch.save(ckpt, save_path)
            acc_best, epoch_best = acc, e
        print('best acc', acc_best, 'in epoch', epoch_best)
    writer.close()
    # pickle.dump({'acc': acc_list, 'loss': loss_list}, open(log_path, 'wb'))


def test(model, device, test_loader, metric):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += metric(output, target)
            pred = output.argmax(dim=1)
            # print(pred)
            # print(pred.shape)
            # print(target)
            # print(len(target))
            # print(sum(pred==target))
            correct += pred.eq(target.view_as(pred)).sum().item()
            # print(correct)

    test_loss /= len(test_loader.dataset)

    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), acc))
    return acc


def main():
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str)
    parser.add_argument('--data', type=str)
    args = parser.parse_args()
    model_choice = args.model
    data_choice = args.data

    # config
    config = configparser.ConfigParser()
    config.read("config.ini")
    batch_size = int(config[model_choice]['batch_size'])
    test_batch_size = int(config[model_choice]['test_batch_size'])
    epoch = int(config[model_choice]['epoch'])
    lr = float(config[model_choice]['lr'])
    seed = int(config[model_choice]['seed'])
    log_interval = int(config[model_choice]['log_interval'])
    save_path = config[model_choice]['save_path']
    log_path = config[model_choice]['log_path']

    # setting
    torch.manual_seed(seed)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # data loader
    train_data = MFCCDataset(os.path.join(config[data_choice]['data_path'], config['data']['train_file']))
    dev_data = MFCCDataset(os.path.join(config[data_choice]['data_path'], config['data']['dev_file']))
    print('dataset', len(train_data), len(dev_data))
    kwargs = {'num_workers': 32, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(dev_data, batch_size=test_batch_size, shuffle=True, **kwargs)

    # model setting
    Model = {
        'cnn': CNN,
        'vgg': VGG
    }
    model = Model[model_choice]().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    metric = nn.CrossEntropyLoss().to(device)

    # train and test
    train(model, device, train_loader, test_loader, optimizer, epoch, metric, log_interval, save_path, log_path)


if __name__ == '__main__':
    main()
