import argparse
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
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
    patience_count = 0
    best_loss = 0
    train_loss = np.zeros(opt.epochs)
    val_loss = np.zeros(opt.epochs)
    epochs_range = range(opt.epochs)

    save_path = os.path.join(opt.save_path, 'task{}/'.format(opt.task_id))
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
        last_path = os.path.join(opt.save_path, 'task{}'.format(opt.task_id - 1))
        last_path.replace('\\', '/')
        net.load_state_dict(torch.load(os.path.join(last_path, 'net_best.pth')))
        old_net = UNet(n_channels=1, n_classes=2).cuda()
        old_net.load_state_dict(torch.load(os.path.join(last_path, 'net_best.pth')))

    def kl_loss(image, output):
        net.eval()
        with torch.no_grad():
            old_output = old_net(image)
        return nn.MSELoss()(old_output, output)

    for epoch in epochs_range:
        logger.log(f'Epoch {epoch + 1} Start')

        total_loss = 0
        step = 0
        start_time = time.time()

        for image, label in loader_train:
            net.train()
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
            loss, _ = criterion(pred, label.squeeze())
            total_loss += loss.data.cpu().numpy().item()
            tmp = torch.tensor(0, dtype=torch.float32)
            tmp = tmp.to(device=device)
            if opt.task_id > 0:
                tmp = kl_loss(image, pred)
            
            net.train()
            loss += tmp

            total_loss += loss.data.cpu().numpy().item()
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                logger.log(f'epoch:{epoch + 1} step:{step + 1} total_loss:{total_loss}')

            step += 1

        end_time = time.time()
        scheduler.step(epoch)
        total_loss /= len(loader_train)
        train_loss[epoch] = total_loss
        logger.log('Epoch{}_train_loss = {}, using {}s'.format(epoch + 1, total_loss, end_time - start_time))

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
                # loss = Mul_dice(result, label.squeeze())
                # total_val += loss.data.cpu().numpy().item()
                _, loss = criterion(pred, label.squeeze())
                total_val += loss.data.cpu().numpy().item()
        total_val /= len(loader_val)
        logger.log('Epoch{}_val_DICE = {}'.format(epoch + 1, total_val))
        val_loss[epoch] = total_val
        
        if total_val > best_loss:
            best_loss = total_val
            torch.save(net.state_dict(), os.path.join(save_path, 'net_best.pth'))
            patience_count = 0
        else:
            patience_count += 1
            logger.log('Early stopping patience {}'.format(patience_count))

        if patience_count >= opt.patience:
            logger.log('Early stopping at epoch {}'.format(epoch + 1))
            break

    np.save(save_path + 'train_loss.npy', train_loss)
    np.save(save_path + 'val_loss.npy', val_loss)


if __name__ == "__main__":
    setup_seed(20)
    parser = argparse.ArgumentParser(description='Unet_train')
    parser.add_argument('--method_name', type=str, default='LwF-SGD', help='Method name')
    parser.add_argument('--order', type=int, default=3, help='Training order')
    parser.add_argument('--dataset', type=str, default='fundus', help='Dataset type. It could be "fundus" or "cardiac".')
    parser.add_argument('--batch_size', type=int, default=16, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=150, help='Number of training epochs')
    parser.add_argument('--milestone', type=int, default=[20, 50, 100], help='When to decay learning rate')
    parser.add_argument('--lr', type=float, default=1e-3, help='Initial learning rate')
    parser.add_argument('--save_path', type=str, default='../../parameters/LwF', help='Path to save models')
    parser.add_argument('--data_path', type=str, default='../../data/fundus/Chase/', help='Path of training data')
    parser.add_argument('--task_id', type=int, default=0)
    parser.add_argument('--patience', type=int, default=20,
                        help='The number of epochs to wait before early stopping')
    opt = parser.parse_args()
    
    # train_exp = ['LwF', 'LwF-tiny', 'LwF-small', 'LwF-large']
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