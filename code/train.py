import argparse
import time

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from model_03 import Net
from dataset import *
import matplotlib.pyplot as plt
from evaluation import psnr
import numpy as np
from tqdm import tqdm
import sys
import os

parser = argparse.ArgumentParser(description="PyTorch D3Dnet")
parser.add_argument("--save", default='./log', type=str, help="Save path")
parser.add_argument("--resume", default="", type=str, help="Resume path (default: none)")
parser.add_argument("--scale_factor", type=int, default=4, help="scale")
parser.add_argument("--train_dataset_dir", default='E:/Desktop/D3Dnet-master/code/data/Vid4', type=str, help="train_dataset")
parser.add_argument("--inType", type=str, default='y', help="RGB input or y input")
parser.add_argument("--batchSize", type=int, default=1, help="Training batch size")
parser.add_argument("--nEpochs", type=int, default=1, help="Number of epochs to train for")
parser.add_argument("--gpu", default=0, type=int, help="gpu ids (default: 0)")
parser.add_argument("--lr", type=float, default=4e-4, help="Learning Rate. Default=4e-4")
parser.add_argument('--gamma', type=float, default=0.5, help='gamma')
parser.add_argument("--step", type=int, default=6, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=6")
parser.add_argument("--threads", type=int, default=1, help="Number of threads for data loader to use, Default: 1")

global opt, model
opt = parser.parse_args()
torch.cuda.set_device(opt.gpu)


def train(num_files, train_loader, scale_factor, epoch_num):

    net = Net(scale_factor).cuda()
    if num_files != 1:
        net.load_state_dict(torch.load("./model.pth"))

    epoch_state = 0
    loss_list = []
    psnr_list = []
    loss_epoch = []
    psnr_epoch = []

    if opt.resume:
        ckpt = torch.load(opt.resume)
        net.load_state_dict(ckpt['state_dict'])
        epoch_state = ckpt['epoch']
        loss_list = ckpt['loss']
        psnr_list = ckpt['psnr']

    optimizer = torch.optim.Adam(net.parameters(), lr=opt.lr)
    criterion_MSE = torch.nn.MSELoss().cuda()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.step, gamma=opt.gamma)
    for idx_epoch in range(epoch_state, epoch_num):
        train_bar = tqdm(train_loader, file=sys.stdout)
        for idx_iter, (LR, HR) in enumerate(train_bar):
            LR, HR = Variable(LR).cuda(), Variable(HR).cuda()
            SR = net(LR)

            loss = criterion_MSE(SR, HR[:, :, 3, :, :])
            loss_epoch.append(loss.detach().cpu())
            psnr_epoch.append(psnr(SR, HR[:, :, 3, :, :]))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bar.desc = f"train epoch[{idx_epoch + 1}/{epoch_num}] loss:{loss:.3f}"

        scheduler.step()
        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            print(time.ctime()[4:-5] + ' Epoch---%d, loss_epoch---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))
            torch.save(net.state_dict(), "./model.pth")
            loss_epoch = []
            psnr_epoch = []


def main():
    for num_files in range(1, 5):
        train_set = TrainSetLoader(num_files, opt.train_dataset_dir, scale_factor=opt.scale_factor,
                                   inType=opt.inType)
        print(f"第{num_files}个数据集预处理完成，准备打包！")
        train_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize,
                                  shuffle=True)
        print(f"第{num_files}个数据集打包完成，开始训练！")
        train(num_files, train_loader, opt.scale_factor, opt.nEpochs)


if __name__ == '__main__':
    main()

