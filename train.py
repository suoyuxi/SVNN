import argparse
import logging
import os
import sys
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader

from dataset import BasicDataset
from svnn import svnn, LeNet
from eval import eval

class trainLoss(nn.Module):
    def __init__(self, alpha=0.10, beta=0.70, gamma=0.70):
        super(trainLoss, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def structureWeight(self, gt):
        return 1 / (self.alpha + torch.abs(gt))

    def suppressWeight(self, error):
        return torch.pow(error,2) + error

    def forward(self, pre, gt, sam):

        ErrorLoss = torch.sum( self.structureWeight(gt) * torch.abs((pre-gt)) ) / torch.sum( self.structureWeight(gt) )
        StrucLoss = torch.sum( torch.relu(torch.abs(gt)-torch.abs(sam)) * torch.abs(pre-gt) ) / torch.sum( torch.relu(torch.abs(gt)-torch.abs(sam)) )
        # SupprLoss = torch.sum( torch.pow(torch.relu(torch.abs(sam)-torch.abs(gt)),2) * torch.abs(pre-gt) ) / torch.sum( torch.pow(torch.relu(torch.abs(sam)-torch.abs(gt)),2) )
        SupprLoss = torch.sum( self.suppressWeight(torch.relu(torch.abs(sam)-torch.abs(gt))) * torch.abs(pre-gt) ) / torch.sum( self.suppressWeight(torch.relu(torch.abs(sam)-torch.abs(gt))) )


        return ErrorLoss + self.beta*StrucLoss + self.gamma*SupprLoss

def train_net(net,
              device,
              epoch_start=0,
              epochs=5,
              batch_size=1,
              lr=0.001,
              save_cp=True,
              dir_checkpoint='workdir/',
              val=True):

    #set up trainset and valset
    trainset = BasicDataset()
    valset = BasicDataset(is_train=False)
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)

    logger.info('Starting training:\
        Start Epoch:     {}\
        Epochs:          {}\
        Batch size:      {}\
        Learning rate:   {}\
        Device:          {}'.format(epoch_start,epochs,batch_size,lr,device.type))

    # optimizer = optim.Adam(net.parameters(), lr=lr, betas=(0.9,0.99)) # Adam for LeNet
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9) # SGD for svnn
    scheduler = lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1) 
    
    # criterion = nn.MSELoss() # MSE for LeNet
    criterion = trainLoss() # specially designed loss for svnn
    logger.info('trainLoss-(alpha:{})-(beta:{})-(gamma:{})'.format(criterion.alpha, criterion.beta, criterion.gamma))

    for epoch in range(epoch_start, epoch_start+epochs):
        # net.train()
        epoch_loss = 0
        with tqdm(total=len(trainset), desc='Epoch {}/{}'.format(epoch+1,epochs+epoch_start)) as pbar:
            cnt_sample = 0
            for sample,gt in train_loader:
                # load data to GPU or CPU
                sample = sample.to(device=device, dtype=torch.float32)
                gt = gt.to(device=device, dtype=torch.float32)
                # forward inference
                out_result = net(sample)
                loss = criterion(out_result, gt, sample)
                epoch_loss += loss.item()
                # show the loss and process
                pbar.set_postfix({'loss for one sample': loss.item()})
                # bp
                optimizer.zero_grad()
                loss.backward()
                
                optimizer.step()
                # update par
                pbar.update(sample.shape[0])
                # val for each 1500 iterations
                cnt_sample = cnt_sample + 1
        
        scheduler.step()
        logger.info('epoch : {} learninf rate : {}'.format(epoch + 1, scheduler.get_lr()))

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logger.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       os.path.join(dir_checkpoint, 'CP_epoch{}.pth'.format(epoch + 1)))
            if val:    
                esr = eval(net, val_loader, device)                    
                logger.info('Validation error to signal ratio {}'.format(esr))
            logger.info('Checkpoint {} saved !'.format(epoch + 1))
            logger.info('epoch_loss: {}'.format(epoch_loss))

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=100,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-s', '--epoch_start', metavar='ES', type=int, default=0,
                        help='start of epoch', dest='epoch_start')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=64,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.01,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-c', '--checkpoint', dest='checkpoint', type=str, default='workdir/',
                        help='save model in a file folder')
    parser.add_argument('-v', '--validation', dest='val', type=bool, default=True,
                        help='whether validation or not')
    parser.add_argument('-fl', '--filter-length', dest='filter_length', type=int, default=5,
                        help='set filter filter length')
    return parser.parse_args()


if __name__ == '__main__':
    # parse the args
    args = get_args()
    #make checkpoint dir
    os.makedirs(args.checkpoint, exist_ok=True)
    # set up log
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    logger = logging.getLogger()
    # log out
    txt_handler = logging.FileHandler(os.path.join(args.checkpoint, 'log.txt'), mode='a')
    txt_handler.setLevel(logging.INFO)
    logger.addHandler(txt_handler)
    # lock GPU or CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Using device {}'.format(device))

    #load the net
    net = svnn(filter_length=args.filter_length) # svnn
    # net = LeNet(seqLength=512) # shallow LeNet
    logger.info('Starting training:\
                filter_length:  {}'.format(args.filter_length))
    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logger.info('Model loaded from {}'.format(args.load))

    net.to(device=device)

    try:
        train_net(net=net,
                  epoch_start=args.epoch_start,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  dir_checkpoint=args.checkpoint,
                  val=args.val
                  )
    except KeyboardInterrupt:
        interrupt_path = args.checkpoint + 'INTERRUPTED.pth'
        torch.save(net.state_dict(), interrupt_path)
        logger.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
