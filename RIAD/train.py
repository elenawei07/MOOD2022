import os
import random
import argparse
import time
import torch
import torch.optim as optim
import torch.nn as nn
from torchvision.utils import save_image
from tqdm import tqdm
from datasets.mvtec import MVTecDataset
# from datasets.preprocessing import generate_image_list, augment_images
from utils.funcs import EarlyStop, denorm
from utils.utils import time_file_str, time_string, convert_secs2time, AverageMeter, print_log
from utils.gen_mask import gen_mask
from models.unet import UNet
from losses.gms_loss import MSGMS_Loss
from losses.ssim_loss import SSIM_Loss
from data import *
from pathlib import Path
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')


def main():
    parser = argparse.ArgumentParser(description='RIAD anomaly detection')
    parser.add_argument('--data_type', type=str, default='brain')
    parser.add_argument('--data_path', type=str, default='/home/data/WJ/MOOD2022/data/train/process_brain_train')
    parser.add_argument('--epochs', type=int, default=100, help='maximum training epochs')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--validation_ratio', type=float, default=0.2)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--belta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=1.0)
    parser.add_argument('--lr', type=float, default=0.00001, help='learning rate of Adam')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='decay of Adam')
    parser.add_argument('--seed', type=int, default=None, help='manual seed')
    parser.add_argument('--k_value', type=int, nargs='+', default=[2, 4, 8, 16])
    args = parser.parse_args()

    args.input_channel = 3
    test_data_path = '/home/data/WJ/MOOD2022/data/test_anomaly/abdomen/data'
    test_gt_path = '/home/data/WJ/MOOD2022/data/test_anomaly/abdomen/label'

    if args.seed is None:
        args.seed = random.randint(1, 10000)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    args.prefix = time_file_str()
    args.save_dir = './' + args.data_type + '/seed_{}/'.format(args.seed)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    log = open(os.path.join(args.save_dir, 'model_training_log_{}.txt'.format(args.prefix)), 'w')
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)

    # load model and dataset
    model = UNet(n_channels=args.input_channel).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
    train_dataset, val_dataset = get_train_val_datasets(datapath=Path(args.data_path), data=args.data_type)
    test_dataset = get_test_dataset(datapath=Path(test_data_path), gtpath=Path(test_gt_path), data=args.data_type)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)

    # fetch fixed data for debugging
    x_normal_fixed, _ = iter(val_loader).next()
    x_normal_fixed = x_normal_fixed.to(device)

    x_test_fixed, _ = iter(test_loader).next()
    x_test_fixed = x_test_fixed.to(device)

    # start training
    save_name = None
    early_stop = EarlyStop(patience=10, save_name=save_name)
    start_time = time.time()
    epoch_time = AverageMeter()
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)
        print_log(' {:3d}/{:3d} ----- [{:s}] {:s}'.format(epoch, args.epochs, time_string(), need_time), log)
        train(args, model, epoch, train_loader, optimizer, log)
        val_loss = val(args, model, epoch, val_loader, log)

        if epoch % 10 == 0:
            save_sample = os.path.join(args.save_dir, '{}-images.jpg'.format(epoch))
            save_sample2 = os.path.join(args.save_dir, '{}test-images.jpg'.format(epoch))
            save_snapshot(x_normal_fixed, x_test_fixed, model, save_sample, save_sample2, log)

        save_name = os.path.join(args.save_dir, '{}_{}_{}_model.pt'.format(args.data_type, args.prefix, epoch))
        if (early_stop(val_loss, model, optimizer, log, save_name)):
            break

        epoch_time.update(time.time() - start_time)
        start_time = time.time()
    log.close()


def train(args, model, epoch, train_loader, optimizer, log):
    model.train()
    l2_losses = AverageMeter()
    gms_losses = AverageMeter()
    ssim_losses = AverageMeter()
    ssim = SSIM_Loss()
    mse = nn.MSELoss(reduction='mean')
    msgms = MSGMS_Loss()
    for (data, _,) in tqdm(train_loader):
        optimizer.zero_grad()

        data = data.to(device)
        # generator mask
        k_value = random.sample(args.k_value, 1)
        Ms_generator = gen_mask(k_value, 3, args.img_size)
        Ms = next(Ms_generator)

        inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
        outputs = [model(x) for x in inputs]
        output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))

        l2_loss = mse(data, output)
        gms_loss = msgms(data, output)
        ssim_loss = ssim(data, output)

        loss = args.gamma * l2_loss + args.alpha * gms_loss + args.belta * ssim_loss

        l2_losses.update(l2_loss.item(), data.size(0))
        gms_losses.update(gms_loss.item(), data.size(0))
        ssim_losses.update(ssim_loss.item(), data.size(0))

        loss.backward()
        optimizer.step()

    print_log(('Train Epoch: {} L2_Loss: {:.6f} GMS_Loss: {:.6f} SSIM_Loss: {:.6f}'.format(
        epoch, l2_losses.avg, gms_losses.avg, ssim_losses.avg)), log)


def val(args, model, epoch, val_loader, log):
    model.eval()
    losses = AverageMeter()
    ssim = SSIM_Loss()
    mse = nn.MSELoss(reduction='mean')
    msgms = MSGMS_Loss()
    for (data, _,) in tqdm(val_loader):
        data = data.to(device)
        # generator mask
        k_value = random.sample(args.k_value, 1)
        Ms_generator = gen_mask(k_value, 3, args.img_size)
        Ms = next(Ms_generator)
        inputs = [data * (torch.tensor(mask, requires_grad=False).to(device)) for mask in Ms]
        with torch.no_grad():
            outputs = [model(x) for x in inputs]
            output = sum(map(lambda x, y: x * (torch.tensor(1 - y, requires_grad=False).to(device)), outputs, Ms))

            l2_loss = mse(data, output)
            gms_loss = msgms(data, output)
            ssim_loss = ssim(data, output)

            loss = args.gamma * l2_loss + args.alpha * gms_loss + args.alpha * ssim_loss
            losses.update(loss.item(), data.size(0))
    print_log(('Valid Epoch: {} loss: {:.6f}'.format(epoch, losses.avg)), log)

    return losses.avg


def save_snapshot(x, x2, model, save_dir, save_dir2, log):
    model.eval()
    with torch.no_grad():
        x_fake_list = x
        recon = model(x)
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(denorm(x_concat.data.cpu()), save_dir, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir)), log)

        x_fake_list = x2
        recon = model(x2)
        x_concat = torch.cat((x_fake_list, recon), dim=3)
        save_image(denorm(x_concat.data.cpu()), save_dir2, nrow=1, padding=0)
        print_log(('Saved real and fake images into {}...'.format(save_dir2)), log)


def adjust_learning_rate(args, optimizer, epoch):
    if epoch == 50 or epoch == 75:
        lr = args.lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


if __name__ == '__main__':
    main()
