import argparse
from torch import optim
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import warnings
import time
from torch.utils.data import DataLoader
from utils import *
import sys,os
sys.path.append('.')
# from model.unet_model_small import UNet
from model.unet_model import UNet
# from model.unet_model_tiny import UNet
# from model.unet_model_deep import UNet


warnings.filterwarnings("ignore")


def train_net(net, device, opt, logger):
    setup_seed(20)
    best_loss = 0
    patience_count = 0
    train_loss = np.zeros(opt.epochs)
    val_loss = np.zeros(opt.epochs)
    # -----------------del-----------------------
    train_closs = np.zeros(opt.epochs)
    train_sloss = np.zeros(opt.epochs)

    best_Importance = 0
    epochs_range = range(opt.epochs)

    if opt.task_id == 0:
        save_path = os.path.join(opt.save_path, 'task{}/'.format(opt.task_id))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
    else:
        save_path = os.path.join(opt.save_path, 'task{}_{}/'.format(opt.task_id, int(opt.beta)))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

    if opt.dataset == 'fundus':
        from data.fundus.dataset import Data_loader
    elif opt.dataset == 'cardiac':
        from data.cardiac.dataset import Data_loader

    criterion = DiceLoss()
    dataset_train = Data_loader(opt.data_path)
    loader_train = DataLoader(dataset=dataset_train, batch_size=opt.batch_size, shuffle=True)
    dataset_val = Data_loader(opt.data_path, train=False)
    loader_val = DataLoader(dataset=dataset_val, batch_size=1, shuffle=False)
    # optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0.01)
    optimizer = optim.SGD(net.parameters(), lr=opt.lr, weight_decay=0.01, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.1)

    if opt.task_id > 0:
        print('Resume Training...')
        if opt.task_id == 1:
            last_path = os.path.join(opt.save_path, 'task{}'.format(opt.task_id - 1))
        else:
            last_path = os.path.join(opt.save_path, 'task{}_{}'.format(opt.task_id - 1, int(opt.beta)))
        net.load_state_dict(torch.load(os.path.join(last_path, 'net_best.pth')))
        All_Importance = torch.load(os.path.join(last_path, 'All_Importance.pth'))
        old_Star_vals = torch.load(os.path.join(last_path, 'Star.pth'))
        Star_vals = []
        for w in net.parameters():
            Star_vals.append(torch.zeros_like(w))
    else:
        print('First task...')
        All_Importance = []
        Star_vals = []
        initial_vals = []
        for w in net.parameters():
            All_Importance.append(torch.zeros_like(w))
            Star_vals.append(torch.zeros_like(w))
            initial_vals.append(w.data.clone())

    def compute_I(batch_num):
        with torch.no_grad():
            for i, w in enumerate(net.parameters()):
                # Importance[i].mul_(batch_num / (batch_num + 1))
                weight[i] += -torch.mul(w.grad.data, w.data - old_w[i])
                # Importance[i].add_(torch.maximum(weight, torch.zeros_like(weight))/(batch_num + 1))
                Importance[i] = torch.maximum(weight[i], torch.zeros_like(weight[i]))
                # Importance[i] = weight[i]
                old_w[i] = w.data.clone()

    def compute_allI(task_id):
        l = len(best_Importance)
        epsilon = 0.1
        with torch.no_grad():
            for i in range(l):
                if task_id > 0:
                    delta_w = Star_vals[i] - old_Star_vals[i]
                else:
                    delta_w = 0
                All_Importance[i] += best_Importance[i]/(delta_w ** 2 + epsilon)

    def create_npy():
        # create npy
        if opt.task_id == 0:
            net.load_state_dict(torch.load(os.path.join(opt.save_path, 'task0', 'net_best.pth')))
        else:
            net.load_state_dict(torch.load(os.path.join(opt.save_path, f'task{opt.task_id}_{int(opt.beta)}', 'net_best.pth')))
        with torch.no_grad():
            for i, w in enumerate(net.parameters()):
                Star_vals[i].copy_(w.data)
            compute_allI(opt.task_id)
        torch.save(All_Importance, os.path.join(save_path, 'All_Importance.pth'))
        torch.save(Star_vals, os.path.join(save_path, 'Star.pth'))

        np.save(save_path + 'train_loss.npy', train_loss)
        np.save(save_path + 'val_loss.npy', val_loss)

        # -----------------del-----------------------
        np.save(save_path + 'train_closs.npy', train_closs)
        np.save(save_path + 'train_sloss.npy', train_sloss)

    old_w = []
    weight = []
    for w in net.parameters():
        old_w.append(w.data.clone())
        weight.append(torch.zeros_like(w))

    for epoch in epochs_range:
        logger.log(f'Epoch {epoch + 1} Start')

        batch_num = 0
        net.train()
        total_loss = 0
        # -----------------del-----------------------
        closs = 0
        sloss = 0

        step = 0
        start_time = time.time()

        Importance = []
        
        for w in net.parameters():
            Importance.append(torch.zeros_like(w))

        for image, label in loader_train:

            if opt.dataset == 'cardiac':
                if 'M&M' in opt.data_path:
                    label[label == 3] = 0
                if 'ACDC' in opt.data_path:
                    label[label == 1] = 0
                label[label > 0] = 1

            optimizer.zero_grad()
            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            pred = net(image)
            # label = torch.squeeze(label, dim=1)
            # oh_label = nn.functional.one_hot(label, num_classes=2)
            # oh_label = oh_label.swapaxes(1, 3)
            # oh_label = oh_label.swapaxes(2, 3)
            # loss = Mul_Fbeta(pred.softmax(dim=1), oh_label)
            loss, _ = criterion(pred, label.squeeze())
            total_loss += loss.data.cpu().numpy().item()
            # -----------------del-----------------------
            closs += loss.data.cpu().numpy().item()

            tmp = torch.tensor(0, dtype=torch.float32)
            tmp = tmp.to(device=device)
            if opt.task_id > 0:
                for i, w in enumerate(net.parameters()):
                    tmp += opt.beta / 2 * torch.sum(torch.mul(All_Importance[i], torch.square(w - old_Star_vals[i])))
            # -----------------del-----------------------
            sloss += tmp.data.cpu().numpy().item()
            # print(tmax)
            # print(tmp)
            loss += tmp
            total_loss += loss.data.cpu().numpy().item()
            loss.backward()
            optimizer.step()
            compute_I(batch_num)

            if step % 50 == 0:
                logger.log(f'epoch:{epoch + 1} step:{step + 1} total_loss:{total_loss}')

            batch_num += 1
            step += 1

        end_time = time.time()
        scheduler.step(epoch)

        # -----------------del-----------------------
        closs /= len(loader_train)
        sloss /= len(loader_train)
        train_closs[epoch] = closs
        train_sloss[epoch] = sloss

        total_loss /= len(loader_train)
        train_loss[epoch] = total_loss
        logger.log('Epoch{}_train_loss = {}, closs = {}, sloss = {}, using {}s'.format(epoch + 1, total_loss, closs, sloss, end_time - start_time))

        # ------validation------
        total_val = 0
        net.eval()
        with torch.no_grad():
            for image, label in loader_val:

                if opt.dataset == 'cardiac':
                    if 'M&M' in opt.data_path:
                        label[label == 3] = 0
                    if 'ACDC' in opt.data_path:
                        label[label == 1] = 0
                    label[label > 0] = 1

                image = image.to(device=device, dtype=torch.float32)
                label = label.to(device=device, dtype=torch.long)
                pred = net(image)
                # result = pred.argmax(dim=1)
                _, loss = criterion(pred, label.squeeze())
                total_val += loss.data.cpu().numpy().item()
        total_val /= len(loader_val)
        logger.log('Epoch{}_val_DICE = {}'.format(epoch + 1, total_val))
        val_loss[epoch] = total_val

        if total_val > best_loss:
            best_loss = total_val
            torch.save(net.state_dict(), os.path.join(save_path, 'net_best.pth'))
            best_Importance = Importance
            torch.save(best_Importance, os.path.join(save_path, 'best_Importance.pth'))
            with torch.no_grad():
                for i, w in enumerate(net.parameters()):
                    Star_vals[i].copy_(w.data)
            patience_count = 0
        else:
            patience_count += 1
            logger.log('Early stopping patience {}'.format(patience_count))

        if patience_count >= opt.patience:
            logger.log('Early stopping at epoch {}'.format(epoch + 1))
            break

    create_npy()

if __name__ == "__main__":
    setup_seed(20)
    parser = argparse.ArgumentParser(description='Unet_train')
    parser.add_argument('--method_name', type=str, default='SI-SGD', help='Method name')
    parser.add_argument('--order', type=int, default=3, help='Training order')
    parser.add_argument('--dataset', type=str, default='cardiac',
                        help='Dataset type. It could be "fundus" or "cardiac".')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--milestone', type=int, default=[20, 50, 100], help='When to decay learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--save_path', type=str, default='../../parameters/SI', help='Path to save models')
    parser.add_argument('--data_path', type=str, default='../../data/fundus/Chase/', help='Path of training data')
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--beta', type=int, default=5)
    parser.add_argument('--patience', type=int, default=20,
                        help='The number of epochs to wait before early stopping')
    opt = parser.parse_args()

    # train_exp = ['SI', 'SI-tiny', 'SI-small', 'SI-large']
    # for exp in train_exp:
    #     opt.method_name = exp

    #     try:
    #         cap = opt.method_name.split('-')[1]
    #     except IndexError:
    #         cap = ''
    #     if cap == 'tiny':
    #         from model.unet_model_tiny import UNet
    #     elif cap == 'small':
    #         from model.unet_model_small import UNet
    #     elif cap == '' or cap == 'SGD':
    #         from model.unet_model import UNet
    #     elif cap == 'large':
    #         from model.unet_model_deep import UNet
        
    episode_train(opt, UNet, train_net, Logger)

    # respective_train(opt, UNet, train_net, Logger)
