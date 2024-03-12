import os
import random
import numpy as np
import torch
import logging
import torch.nn.functional as F
import torch.nn as nn


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['PYTHONHASHSEED'] = str(seed)


class DiceLoss(nn.Module):
    """Dice Loss PyTorch
        dice_loss = 1 - 2*p*t / (p^2 + t^2). p and t represent predict and target.
    Args:
        weight: An array of shape [C,]
        predict: A float32 tensor of shape [N, C, *], for Semantic segmentation task is [N, C, H, W]
        target: A int64 tensor of shape [N, *], for Semantic segmentation task is [N, H, W]
    Return:
        diceloss, dice coefficient
    """

    def __init__(self, weight=None, device='cuda'):
        super(DiceLoss, self).__init__()
        if weight is not None:
            weight = torch.Tensor(weight)
            self.weight = weight / torch.sum(weight)  # Normalized weight
        self.smooth = 1e-5
        self.device = device

    def forward(self, predict, target):
        N, C = predict.size()[:2]
        predict = predict.view(N, C, -1)  # (N, C, *)
        target = target.view(N, 1, -1)  # (N, 1, *)

        predict = F.softmax(predict, dim=1)  # (N, C, *) ==> (N, C, *)
        ## convert target(N, 1, *) into one hot vector (N, C, *)
        # (N, 1, *) ==> (N, C, *)
        target_onehot = torch.zeros(predict.size()).to(self.device)
        target_onehot.scatter_(1, target, 1)  # (N, C, *)
        # print(predict.shape, target_onehot.shape)
        predict = predict[:, 1:, :] # remove the background
        target_onehot = target_onehot[:, 1:, :] # remove the background
        # print(predict.shape, target_onehot.shape)   
        intersection = torch.sum(predict * target_onehot, dim=2)  # (N, C)
        union = torch.sum(predict.pow(2), dim=2) + \
            torch.sum(target_onehot, dim=2)  # (N, C)
        ## p^2 + t^2 >= 2*p*t, target_onehot^2 == target_onehot
        dice_coef = (2 * intersection + self.smooth) / \
            (union + self.smooth)  # (N, C)

        if hasattr(self, 'weight'):
            if self.weight.type() != predict.type():
                self.weight = self.weight.type_as(predict)
                dice_coef = dice_coef * self.weight * C  # (N, C)
        dice_loss = 1 - torch.mean(dice_coef)  # 1

        return dice_loss, torch.mean(dice_coef)


def dice(player_A, player_B, beta):
    inter = player_A.flatten() * player_B.flatten()
    if player_A.sum() == 0 and player_B.sum() == 0:
        return torch.tensor(0.0)
    return ((1 + beta ** 2) * inter.sum() / (player_A.sum() + (beta ** 2) * player_B.sum())).mean()


def Mul_dice(pred, label, class_id=1, beta=1):
    player_A = (pred == class_id).to(dtype=torch.int)
    player_B = (label == class_id).to(dtype=torch.int)
    return dice(player_A, player_B, beta)


def test_dice(pred, label, seen_classes, beta=1):
    loss = 0
    for class_id in range(1, seen_classes + 1):
        player_A = (pred == class_id).to(dtype=torch.int)
        player_B = (label == class_id).to(dtype=torch.int)
        loss += dice(player_A, player_B, beta)
    loss /= seen_classes
    return loss


def Fbeta(player_A, player_B, beta):
    player_A = player_A.view(player_A.shape[0], -1)
    player_B = player_B.view(player_B.shape[0], -1)
    inter = player_A * player_B
    if player_A.sum() == 0 and player_B.sum() == 0:
        return torch.tensor(0.0)
    return (1 - (1 + beta ** 2) * inter.sum(dim=1) / (
            (player_A).sum(dim=1) + (beta ** 2) * (player_B).sum(dim=1))).mean()


def Mul_Fbeta(pred, label, class_id=1, beta=1):
    player_A = pred[:, class_id, :, :]
    player_B = label[:, class_id, :, :]
    return Fbeta(player_A, player_B, beta)


def episode_train(opt, UNet, train_net, Logger):
    trian_tj = [(2, 'fundus'), (3, 'fundus'), (2, 'cardiac'), (3, 'cardiac')]

    for tj in trian_tj:
        opt.order, opt.dataset = tj[0], tj[1]

        if opt.dataset == 'fundus':
            if opt.order == 0:
                # easy to hard
                datasets = {0: 'Chase', 1: 'Stare', 2: 'Rite', 3: 'Drhagis'}
            elif opt.order == 1:
                # hard to easy
                datasets = {0: 'Drhagis', 1: 'Rite', 2: 'Stare', 3: 'Chase'}
            elif opt.order == 2:
                # easy to hard
                datasets = {0: 'Rite', 1: 'Chase', 2: 'Stare', 3: 'Drhagis'}
            elif opt.order == 3:
                # hard to easy
                datasets = {0: 'Drhagis', 1: 'Stare', 2: 'Chase', 3: 'Rite'}
            else:
                # randomly
                datasets = {0: 'Stare', 1: 'Drhagis', 2: 'Chase', 3: 'Rite'}
        elif opt.dataset == 'cardiac':
            if opt.order == 0:
                # easy to hard
                datasets = {0: 'Sunnybrook', 1: 'CAP', 2: 'ACDC', 3: 'M&M'}
            elif opt.order == 1:
                # hard to easy
                datasets = {0: 'M&M', 1: 'ACDC', 2: 'CAP', 3: 'Sunnybrook'}
            elif opt.order == 2:
                # easy to hard
                datasets = {0: 'M&M', 1: 'Sunnybrook', 2: 'ACDC', 3: 'CAP'}
            elif opt.order == 3:
                # hard to easy
                datasets = {0: 'CAP', 1: 'ACDC', 2: 'Sunnybrook', 3: 'M&M'}
            else:
                # randomly
                datasets = {0: 'CAP', 1: 'M&M', 2: 'Sunnybrook', 3: 'ACDC'}

        for i in datasets.keys():
            opt.save_path = f'parameters/{opt.dataset}/{opt.method_name}_order{opt.order}'
            opt.data_path = f'data/{opt.dataset}/{datasets[i]}/'
            opt.task_id = i

            if not os.path.exists(opt.save_path):
                os.makedirs(opt.save_path)

            logger = Logger(opt.save_path, f'{opt.method_name}_order{opt.order}_{opt.dataset}_task{opt.task_id}')
            logger.log(f'(order {opt.order}) train in {datasets[i]}')

            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

            net = UNet(n_channels=1, n_classes=2)

            net.to(device=device)

            train_net(net, device, opt, logger)


def respective_train(opt, UNet, train_net, Logger):
    if opt.dataset == 'fundus':
        if opt.order == 0:
            # easy to hard
            datasets = {0: 'Chase', 1: 'Stare', 2: 'Rite', 3: 'Drhagis'}
        elif opt.order == 1:
            # hard to easy
            datasets = {0: 'Drhagis', 1: 'Rite', 2: 'Stare', 3: 'Chase'}
        elif opt.order == 2:
            # easy to hard
            datasets = {0: 'Rite', 1: 'Chase', 2: 'Stare', 3: 'Drhagis'}
        elif opt.order == 3:
            # hard to easy
            datasets = {0: 'Drhagis', 1: 'Stare', 2: 'Chase', 3: 'Rite'}
        else:
            # randomly
            datasets = {0: 'Stare', 1: 'Drhagis', 2: 'Chase', 3: 'Rite'}
    elif opt.dataset == 'cardiac':
        if opt.order == 0:
            # easy to hard
            datasets = {0: 'Sunnybrook', 1: 'CAP', 2: 'ACDC', 3: 'M&M'}
        elif opt.order == 1:
            # hard to easy
            datasets = {0: 'M&M', 1: 'ACDC', 2: 'CAP', 3: 'Sunnybrook'}
        elif opt.order == 2:
            # easy to hard
            datasets = {0: 'M&M', 1: 'Sunnybrook', 2: 'ACDC', 3: 'CAP'}
            # datasets = {0: 'CAP'}
        elif opt.order == 3:
            # hard to easy
            datasets = {0: 'CAP', 1: 'ACDC', 2: 'Sunnybrook', 3: 'M&M'}
        else:
            # randomly
            datasets = {0: 'CAP', 1: 'M&M', 2: 'Sunnybrook', 3: 'ACDC'}

    for i in datasets.keys():
        opt.save_path = f'parameters/{opt.dataset}/{opt.method_name}_order{opt.order}'
        opt.data_path = f'data/{opt.dataset}/{datasets[i]}/'
        opt.task_id = i

        if not os.path.exists(opt.save_path):
            os.makedirs(opt.save_path)

        logger = Logger(opt.save_path, f'{opt.method_name}_order{opt.order}_{opt.dataset}_task{opt.task_id}')
        logger.log(f'(order {opt.order}) train in {datasets[i]}')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        net = UNet(n_channels=1, n_classes=2)

        net.to(device=device)

        train_net(net, device, opt, logger)


class Logger:
    def __init__(self, save_path, name):

        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        file_handler = logging.FileHandler(filename=os.path.join(save_path, f'{name}.log'), mode='w', encoding='UTF-8')

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(message)s')
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)

        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def log(self, text):
        self.logger.info(text)

if __name__ == '__main__':
    logger = Logger('test')
    logger.log('epoch 1')
