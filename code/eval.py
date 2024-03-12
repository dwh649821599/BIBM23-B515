import argparse
import warnings
from torch.utils.data import DataLoader
from utils import *
import sys,os
sys.path.append('..')
# from model.unet_model_small import UNet
# from model.unet_model import UNet
from model.unet_model_tiny import UNet
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore")


def eval_net(net, device, opt):
    setup_seed(20)
    test_loss = []
    if 'Naive' in opt.method_name or 'Single' in opt.method_name or 'LwF' in opt.method_name or 'Joint' in opt.method_name:
        load_path = os.path.join(opt.load_path, 'task{}/'.format(opt.task_id))
    else:
        if opt.task_id == 0:
            load_path = os.path.join(opt.load_path, 'task{}/'.format(opt.task_id))
        else:
            load_path = os.path.join(opt.load_path, 'task{}_{}/'.format(opt.task_id, opt.beta))

    if opt.dataset == 'fundus':
        from data.fundus.dataset import Data_loader
    elif opt.dataset == 'cardiac':
        from data.cardiac.dataset import Data_loader

    print('Loading dataset ...\n')
    dataset_name = os.path.basename(os.path.dirname(opt.data_path))
    dataset_test = Data_loader(opt.data_path, test=True)
    loader_test = DataLoader(dataset=dataset_test, batch_size=1, shuffle=False)
    net.load_state_dict(torch.load(os.path.join(load_path, 'net_best.pth')))
    net.eval()
    print(f'test model trained on task{opt.task_id} using {dataset_name}')
    with torch.no_grad():
        for image, label in loader_test:

            if opt.dataset == 'cardiac':
                if 'M&M' in opt.data_path:
                    label[label == 3] = 0
                if 'ACDC' in opt.data_path:
                    label[label == 1] = 0
                label[label > 0] = 1

            image = image.to(device=device, dtype=torch.float32)
            label = label.to(device=device, dtype=torch.long)
            pred = net(image)
            result = pred.argmax(dim=1)
            loss = Mul_dice(result, label.squeeze())
            test_loss.append(loss.data.cpu().numpy().item())

    print(f'test_loss:{np.mean(test_loss)}')
    np.save(opt.save_path + f'/{opt.method_name}_order{opt.order}_{opt.task_id}_{dataset_name}_loss.npy', test_loss)


if __name__ == "__main__":
    setup_seed(20)
    parser = argparse.ArgumentParser(description='Unet_eval')
    parser.add_argument('--method_name', type=str, default='Joint-tiny', help='Method name')
    parser.add_argument('--order', type=int, default=3, help='Training order')
    parser.add_argument('--dataset', type=str, default='cardiac',
                        help='Dataset type. It could be "fundus" or "cardiac".')
    parser.add_argument('--save_path', type=str, default='../result/Naive_order0', help='Path to save models')
    parser.add_argument('--load_path', type=str, default='../parameters/Naive_order0')
    parser.add_argument('--data_path', type=str, default='../data/fundus/Chase/', help='Path of training data')
    parser.add_argument('--beta', type=float, default=5)
    parser.add_argument('--task_id', type=int, default=0)
    opt = parser.parse_args()

    eval_tj = [(2, 'fundus'), (3, 'fundus'), (2, 'cardiac'), (3, 'cardiac')]

    for tj in eval_tj:
        opt.order, opt.dataset = tj[0], tj[1]

        print(f'test {opt.method_name}_order{opt.order} on {opt.dataset}')

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

        opt.save_path = f'../result/{opt.dataset}/{opt.method_name}_order{opt.order}'
        for i in datasets.keys():
            for j in range(len(datasets)):
                opt.load_path = f'../parameters/{opt.dataset}/{opt.method_name}_order{opt.order}'
                opt.data_path = f'../data/{opt.dataset}/{datasets[i]}/'
                # opt.beta = 5
                opt.task_id = j

                if not os.path.exists(opt.save_path):
                    os.makedirs(opt.save_path)

                device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
                net = UNet(n_channels=1, n_classes=2)
                net.to(device=device)
                eval_net(net, device, opt)
