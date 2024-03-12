import argparse
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR
import warnings
import time
from torchsummary import summary
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
    optimizer = optim.Adam(net.parameters(), lr=opt.lr, weight_decay=0.01)
    # optimizer = optim.SGD(net.parameters(), lr=opt.lr, weight_decay=0.01, momentum=0.9)
    scheduler = MultiStepLR(optimizer, milestones=opt.milestone, gamma=0.1)

    if opt.task_id > 0:
        last_path = os.path.join(opt.save_path, 'task{}'.format(opt.task_id - 1))
        last_path.replace('\\', '/')
        net.load_state_dict(torch.load(os.path.join(last_path, 'net_best.pth')))
        old_net = UNet(n_channels=1, n_classes=2).cuda()
        old_net.load_state_dict(torch.load(os.path.join(last_path, 'net_best.pth')))

    # if opt.task_id == 0:
    #     return
    
    def grad_cam_loss(image, pred, label):
        # from pytorch_grad_cam import GradCAM

        # class SemanticSegmentationTarget:
        #     def __init__(self, category, mask):
        #         self.category = category
        #         self.mask = mask
        
        #     def __call__(self, model_output):
        #         # print(model_output.shape)
        #         return (model_output[self.category, :, :] * self.mask).sum()
        
        # print(net)
        old_feature_map = [None]
        new_feature_map = [None]
        old_grad = [None]
        new_grad = [None]
        
        def old_forward_hook(module, inp, outp):
            old_feature_map[0] = outp
        
        def new_forward_hook(module, inp, outp):
            new_feature_map[0] = outp
        
        def old_backward_hook(module, grad_in, grad_out):
            old_grad[0] = grad_out[0]
            
        def new_backward_hook(module, grad_in, grad_out):
            new_grad[0] = grad_out[0]

        # old_net.down4.maxpool_conv[1].double_conv[3].register_forward_hook(old_forward_hook)
        # old_net.down4.maxpool_conv[1].double_conv[3].register_full_backward_hook(old_backward_hook)
        
        # net.down4.maxpool_conv[1].double_conv[3].register_forward_hook(new_forward_hook)
        # net.down4.maxpool_conv[1].double_conv[3].register_full_backward_hook(new_backward_hook)
        
        hooks = []
        
        '''hooks.append(old_net.up4.conv.double_conv[3].register_forward_hook(old_forward_hook))
        hooks.append(old_net.up4.conv.double_conv[3].register_backward_hook(old_backward_hook))
        
        hooks.append(net.up4.conv.double_conv[3].register_forward_hook(new_forward_hook))
        hooks.append(net.up4.conv.double_conv[3].register_backward_hook(new_backward_hook))'''
        hooks.append(net.down4.maxpool_conv[1].double_conv[3].register_forward_hook(new_forward_hook))
        hooks.append(net.down4.maxpool_conv[1].double_conv[3].register_backward_hook(new_backward_hook))
        
        hooks.append(old_net.down4.maxpool_conv[1].double_conv[3].register_forward_hook(old_forward_hook))
        hooks.append(old_net.down4.maxpool_conv[1].double_conv[3].register_backward_hook(old_backward_hook))
        
        # new_target_layers = [net.up4.conv.double_conv[3]]
        # old_target_layers = [old_net.up4.conv.double_conv[3]]
        # new_target_layers = [net.outc.conv]
        # old_target_layers = [old_net.outc.conv]
        
        gc_loss = torch.tensor(0, dtype=torch.float32).to(device=device)
        
        net.eval()
        old_net.eval()
        
        # new_normalized_masks = torch.nn.functional.softmax(pred, dim=1).cpu()
        # old_normalized_masks = torch.nn.functional.softmax(old_net(image), dim=1).cpu()

        for idx in label.unique():
            if idx == 0:
                continue
            
            net.zero_grad()
            old_net.zero_grad()
            
            pred = net(image)
            yc_new = torch.max(pred, dim=1)[0][pred.argmax(dim=1) == idx].sum()
            old_pred = old_net(image)
            yc_old = torch.max(old_pred, dim=1)[0][old_pred.argmax(dim=1) == idx].sum()
            
            yc_old.backward(retain_graph=True)
            yc_new.backward(retain_graph=True)
            # print(net.down4.maxpool_conv[1].double_conv[3].weight.grad)
            # for name, param in net.named_parameters():
            #     if name == 'up4.conv.double_conv.3.weight':
            #         print(param.grad)
            
            
            # print(torch.unique(old_feature_map[0]))
            # print(torch.unique(old_grad[0]))
            # print(torch.unique(new_feature_map[0]))
            # print(torch.unique(new_grad[0]))
            # exit()
            
            old_cam_map = torch.sum(old_feature_map[0] * old_grad[0], dim=1)
            old_cam_map = torch.mean(torch.maximum(old_cam_map, torch.tensor(0, dtype=torch.float32).to(device=device)), dim=0)
            # print(old_cam_map.shape)
            new_cam_map = torch.sum(new_feature_map[0] * new_grad[0], dim=1)
            new_cam_map = torch.mean(torch.maximum(new_cam_map, torch.tensor(0, dtype=torch.float32).to(device=device)), dim=0)
            # print(new_cam_map.shape)
            
            net.zero_grad()
            old_net.zero_grad()
            
            # category = idx
            # old_mask = old_normalized_masks.argmax(axis=1).detach().cpu().numpy()
            # new_mask = new_normalized_masks.argmax(axis=1).detach().cpu().numpy()
            # old_mask_float = np.float32(old_mask == category)
            # new_mask_float = np.float32(new_mask == category)
            
            # old_target_layers = [old_net.up4.conv.double_conv[3]]
            # old_targets = [SemanticSegmentationTarget(category, old_mask_float)]
            # with GradCAM(model=old_net,
            #              target_layers=old_target_layers,
            #              use_cuda=torch.cuda.is_available()) as cam:
            #     old_grad_cam = cam(input_tensor=image,
            #                         targets=old_targets)
                
            # new_target_layers = [net.up4.conv.double_conv[3]]
            # new_targets = [SemanticSegmentationTarget(category, new_mask_float)]
            # with GradCAM(model=net,
            #              target_layers=new_target_layers,
            #              use_cuda=torch.cuda.is_available()) as cam:
            #     new_grad_cam = cam(input_tensor=image,
            #                         targets=new_targets)

            # print(new_grad_cam.shape)
            # old_grad_cam = torch.from_numpy(old_grad_cam).to(device=device, dtype=torch.float32)
            # new_grad_cam = torch.from_numpy(new_grad_cam).to(device=device, dtype=torch.float32)
            gc_loss += nn.L1Loss()(new_cam_map, old_cam_map)
            # gc = nn.L1Loss()(new_grad_cam, old_grad_cam)
            # if torch.isnan(gcl).int().sum() == 0:
            # gc_loss += gc
            # print(gc_loss)
            
            for hook in hooks:
                hook.remove()
            
            
        return gc_loss
    
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
            label = torch.squeeze(label, dim=1)
            oh_label = nn.functional.one_hot(label, num_classes=2)
            oh_label = oh_label.swapaxes(1, 3)
            oh_label = oh_label.swapaxes(2, 3)
            loss = Mul_Fbeta(pred.softmax(dim=1), oh_label)
            total_loss += loss.data.cpu().numpy().item()
            # loss, _ = criterion(pred, label.squeeze())
            # total_loss += loss.data.cpu().numpy().item()
            tmp = torch.tensor(0, dtype=torch.float32)
            tmp = tmp.to(device=device)
            if opt.task_id > 0:
                kl = kl_loss(image, pred)
                grad_cam = grad_cam_loss(image, pred, label)
                
                # print(kl, grad_cam)
                tmp = 0.05*kl + 0.05*grad_cam
            
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
                result = pred.argmax(dim=1)
                loss = Mul_dice(result, label.squeeze())
                total_val += loss.data.cpu().numpy().item()
                # _, loss = criterion(pred, label.squeeze())
                # total_val += loss.data.cpu().numpy().item()
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
    parser.add_argument('--method_name', type=str, default='LwM', help='Method name')
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

    # train_exp = ['LwM', 'LwM-tiny', 'LwM-small', 'LwM-large']
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