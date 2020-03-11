import argparse
from collections import OrderedDict
from pathlib import Path
import sys

import numpy as np
import pretrainedmodels
import pretrainedmodels.utils
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.optim
from torch.optim.lr_scheduler import StepLR
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import FaceDatasets
from model import get_model


def print_fun(s):
    print(s)
    sys.stdout.flush()


def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--limit", type=int, default=0, help="Limit steps")
    parser.add_argument("--utk-dir", type=str, required=False, help="UTK Data root directory")
    parser.add_argument("--appa-real-dir", type=str, required=False, help="APPA-REAL Data root directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint if any")
    parser.add_argument("--checkpoint", type=str, default="checkpoint", help="Checkpoint directory")
    parser.add_argument("--tensorboard", type=str, default=None, help="Tensorboard log directory")
    parser.add_argument('--multi_gpu', action="store_true", help="Use multi GPUs (data parallel)")
    parser.add_argument("--aug", action='store_true', default=False)

    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--opt", default="adam")  # adam or sgd)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr_decay_step", type=float, default=5)
    parser.add_argument("--lr_decay_rate", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--age_stddev", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--arch", default="se_resnext50_32x4d")
    parser.add_argument("--img_size", type=int, default=224)
    args = parser.parse_args()
    return args


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count


def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()

    with tqdm(train_loader, disable=None) as _tqdm:
        for i, (x, y) in enumerate(_tqdm):
            x = x.to(device)
            y = y.to(device)

            # compute output
            outputs = model(x)

            # calc loss
            loss = criterion(outputs, y)
            cur_loss = loss.item()

            # calc accuracy
            _, predicted = outputs.max(1)
            correct_num = predicted.eq(y).sum().item()

            # measure accuracy and record loss
            sample_num = x.size(0)
            loss_monitor.update(cur_loss, sample_num)
            accuracy_monitor.update(correct_num, sample_num)

            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            data = OrderedDict(
                stage="train",
                epoch=epoch,
                loss=loss_monitor.avg,
                acc=accuracy_monitor.avg,
                correct=correct_num,
                sample_num=sample_num
            )
            _tqdm.set_postfix(data)
            if _tqdm.disable and i % 10 == 0:
                data_str = ', '.join([f'{k}={v}' for k, v in data.items()])
                msg = f'Step {i}/{len(train_loader)} [{data_str}]'
                print_fun(msg)

    return loss_monitor.avg, accuracy_monitor.avg


def validate(validate_loader, model, criterion, epoch, device):
    model.eval()
    loss_monitor = AverageMeter()
    accuracy_monitor = AverageMeter()
    preds = []
    gt = []

    with torch.no_grad():
        with tqdm(validate_loader, disable=None) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)

                # compute output
                outputs = model(x)
                preds.append(F.softmax(outputs, dim=-1).cpu().numpy())
                gt.append(y.cpu().numpy())

                # valid for validation, not used for test
                if criterion is not None:
                    # calc loss
                    loss = criterion(outputs, y)
                    cur_loss = loss.item()

                    # calc accuracy
                    _, predicted = outputs.max(1)
                    correct_num = predicted.eq(y).sum().item()

                    # measure accuracy and record loss
                    sample_num = x.size(0)
                    loss_monitor.update(cur_loss, sample_num)
                    accuracy_monitor.update(correct_num, sample_num)
                    data = OrderedDict(
                        stage="val", epoch=epoch, loss=loss_monitor.avg,
                        acc=accuracy_monitor.avg, correct=correct_num, sample_num=sample_num
                    )
                    _tqdm.set_postfix(data)
                    if _tqdm.disable and i % 10 == 0:
                        data_str = ', '.join([f'{k}={v}' for k, v in data.items()])
                        msg = f'Step {i}/{len(validate_loader)} [{data_str}]'
                        print_fun(msg)

    preds = np.concatenate(preds, axis=0)
    gt = np.concatenate(gt, axis=0)
    ages = np.arange(0, 101)
    ave_preds = (preds * ages).sum(axis=-1)
    diff = ave_preds - gt
    mae = np.abs(diff).mean()

    return loss_monitor.avg, accuracy_monitor.avg, mae


def main():
    args = get_args()

    start_epoch = 0
    checkpoint_dir = Path(args.checkpoint)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = get_model(model_name=args.arch)

    if args.opt == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # optionally resume from a checkpoint
    resume_path = args.resume

    if resume_path:
        if Path(resume_path).is_file():
            print("=> loading checkpoint '{}'".format(resume_path))
            checkpoint = torch.load(resume_path, map_location="cpu")
            start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(resume_path, checkpoint['epoch']))
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        else:
            print("=> no checkpoint found at '{}'".format(resume_path))

    if args.multi_gpu:
        model = nn.DataParallel(model)

    if device == "cuda":
        cudnn.benchmark = True

    criterion = nn.CrossEntropyLoss().to(device)
    train_dataset = FaceDatasets(
        args.appa_real_dir,
        # None,
        args.utk_dir,
        "train",
        img_size=args.img_size,
        augment=args.aug,
        age_stddev=args.age_stddev
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=True
    )

    val_dataset = FaceDatasets(
        args.appa_real_dir,
        None,
        "valid",
        img_size=args.img_size, augment=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, drop_last=False
    )

    scheduler = StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_rate,
        last_epoch=start_epoch - 1
    )
    best_val_mae = 10000.0
    train_writer = None

    if args.tensorboard is not None:
        train_writer = SummaryWriter(log_dir=args.tensorboard + "/" + "_train")
        val_writer = SummaryWriter(log_dir=args.tensorboard + "/" + "_val")

    for epoch in range(start_epoch, args.epochs):
        # train
        train_loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, device)

        # validate
        val_loss, val_acc, val_mae = validate(val_loader, model, criterion, epoch, device)

        if args.tensorboard is not None:
            train_writer.add_scalar("loss", train_loss, epoch)
            train_writer.add_scalar("acc", train_acc, epoch)
            val_writer.add_scalar("loss", val_loss, epoch)
            val_writer.add_scalar("acc", val_acc, epoch)
            val_writer.add_scalar("mae", val_mae, epoch)

        # checkpoint
        if val_mae < best_val_mae:
            print(f"=> [epoch {epoch:03d}] best val mae was improved from {best_val_mae:.3f} to {val_mae:.3f}")
            model_state_dict = model.module.state_dict() if args.multi_gpu else model.state_dict()
            torch.save(
                {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model_state_dict,
                    'optimizer_state_dict': optimizer.state_dict()
                },
                str(checkpoint_dir.joinpath("epoch{:03d}_{:.5f}_{:.4f}.pth".format(epoch, val_loss, val_mae)))
            )
            best_val_mae = val_mae
        else:
            print(f"=> [epoch {epoch:03d}] best val mae was not improved from {best_val_mae:.3f} ({val_mae:.3f})")

        # adjust learning rate
        scheduler.step(epoch=epoch)

    print("=> training finished")
    print(f"best val mae: {best_val_mae:.3f}")


if __name__ == '__main__':
    main()
