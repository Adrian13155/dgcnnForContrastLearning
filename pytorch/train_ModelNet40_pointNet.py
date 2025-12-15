from __future__ import annotations
import argparse
import os
import random
from typing import Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import ModelNet40
from datetime import datetime
from model import DGCNN
from util import cal_loss
import sklearn.metrics as metrics
import numpy as np
from models.PointNet import ContrastPointNet

import logging

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


def parse_args():
    p = argparse.ArgumentParser("Classification Training on ModelNet40 (PointNet)")
    p.add_argument("--batchSize", type=int, default=32)
    p.add_argument("--nepoch", type=int, default=250)
    p.add_argument("--npoints", type=int, default=1024)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_cpu", action="store_true", default=False)
    p.add_argument("--gpu_id", type=str, default="7")
    p.add_argument("--feature_transform", action="store_true", default=True)

    p.add_argument('--exp_name', type=str, default='PointNetClsModelNet40_Contrast', help='')
    p.add_argument('--save_dir', help='日志保存路径', default='/data/cjj/projects/PointCloudLearning/dgcnn/experiment', type=str)
    p.add_argument('--checkpoint_path', type=str, default="/data/cjj/projects/ContrastLearning/experiment/11-14_15:32_Fusion360/epoch196.pth", help="checkpoint path")
    
    return p.parse_args()

def feature_transform_regularizer(trans):
    d = trans.size(1)
    I = torch.eye(d, device=trans.device).unsqueeze(0)
    AAT = torch.bmm(trans, trans.transpose(2, 1))
    return torch.mean(torch.norm(I - AAT, dim=(1, 2)))


def make_loaders(args):
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.npoints), num_workers=8,
                              batch_size=args.batchSize, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.npoints), num_workers=8,
                             batch_size=args.batchSize // 2, shuffle=False, drop_last=False)
    return train_loader, test_loader


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()

    # 创建验证进度条
    eval_pbar = tqdm(loader, desc="[Eval]", leave=False, ncols=80)
    test_pred = []
    test_true = []
    for batch in eval_pbar:
        data, label = batch

        data = data.to(device)
        data = data.permute(0, 2, 1)
        label = label.to(device)

        pred, point_feat, trans_feat = net(data)

        preds = pred.max(dim=1)[1]
        test_true.append(label.cpu().numpy())
        test_pred.append(preds.detach().cpu().numpy())

        # 计算当前累积准确率并更新进度条
        current_true = np.concatenate(test_true)
        current_pred = np.concatenate(test_pred)
        current_acc = metrics.accuracy_score(current_true, current_pred)
        eval_pbar.set_postfix({'Acc': f'{current_acc:.4f}'})

    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return metrics.accuracy_score(test_true, test_pred)


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    now = datetime.now()
    formatted_time = now.strftime("%m-%d_%H:%M")
    save_dir = os.path.join(args.save_dir, f'{formatted_time}_{args.exp_name}')

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    if not args.use_cpu:
        torch.cuda.set_device(int(args.gpu_id))

    train_loader, test_loader = make_loaders(args)

    num_classes = 40
    net = ContrastPointNet(k=num_classes, feature_transform=args.feature_transform).cuda()
    optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(os.path.join(save_dir, f'run_{args.exp_name}.log'))
    logger.info(f"Arguments: {args}")
    logger.info(f"model params: {sum(p.numel() for p in net.parameters()) / 1e6} M")
    logger.info(f"Network Structure: {str(net)}")
    best_acc = 0.0

    if isinstance(args.checkpoint_path, str):
        if os.path.exists(args.checkpoint_path):
            net.load_state_dict(torch.load(args.checkpoint_path, map_location=torch.device('cpu')), strict=False)
            net.to(device)
            logger.info(f"Loaded checkpoint from {args.checkpoint_path}")
        else:
            logger.info(f"Checkpoint file not found at {args.checkpoint_path}")

    # 初始测试
    test_acc = evaluate(net, test_loader, device)
    logger.info(f"Epoch [Init, 0] Test Acc: {test_acc:.4f} ")

    for epoch in range(args.nepoch):
        net.train()
        running = {"loss": 0.0}
        correct, total = 0, 0

        # 创建训练进度条
        train_pbar = tqdm(train_loader,
                         desc=f"Epoch {epoch+1:03d}/{args.nepoch:03d} [Train]",
                         leave=False,
                         ncols=100)
        train_pred = []
        train_true = []
        for batch_data in train_pbar:
            data, label = batch_data
            data = data.to(device)
            data = data.permute(0, 2, 1)
            label = label.to(device)
            label = label[:,0]
            batch_size = data.size()[0]
            optimizer.zero_grad()

            # 前向传播
            pred, point_feat, trans_feat = net(data)

            # 分类损失
            loss = F.nll_loss(pred, label)
            if args.feature_transform:
                loss += feature_transform_regularizer(trans_feat) * 0.001

            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.item())

            preds = pred.max(dim=1)[1]

            train_true.append(label.cpu().numpy())
            train_pred.append(preds.detach().cpu().numpy())

            # 更新进度条显示
            current_batch = train_pbar.n
            current_loss = running["loss"] / max(1, current_batch)
            # 计算当前累积准确率
            current_true = np.concatenate(train_true)
            current_pred = np.concatenate(train_pred)
            current_acc = metrics.accuracy_score(current_true, current_pred)
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        train_true = np.concatenate(train_true)
        train_pred = np.concatenate(train_pred)
        train_loss = running["loss"] / max(1, len(train_loader))
        train_acc = metrics.accuracy_score(train_true, train_pred)

        # 验证
        test_acc = evaluate(net, test_loader, device)
        scheduler.step()

        logger.info(f"Epoch [{epoch+1:03d}/{args.nepoch:03d}] "
                       f"Train Loss: {train_loss:.4f} "
                       f"| Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | LR: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最后一个和最好的模型
        torch.save(net.state_dict(), os.path.join(save_dir, "model_last.pth"))
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(net.state_dict(), os.path.join(save_dir, "model_best.pth"))

    logger.info(f"Training done. Best test acc: {best_acc:.4f}")


if __name__ == "__main__":
    main()
