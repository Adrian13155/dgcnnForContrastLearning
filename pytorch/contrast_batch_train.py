
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

from model import DGCNN
from models.GDANet_cls import GDANET
from contrast_loss import region_contrastive_loss, info_nce_loss
from datetime import datetime
from Fusion360Dataset import BatchFusion360Dataset

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
    p = argparse.ArgumentParser("Batch Contrastive + Classification (DGCNN) - Fusion360")
    p.add_argument("--dataset", type=str, default="/data/cjj/dataset/Fusion360GalleryDataset/a1.0.0_10",
                   help="Fusion360 Dataset Path")
    p.add_argument("--batchSize", type=int, default=1, 
                   help="DataLoader batch size (should be 1 for BatchFusion360Dataset, each item is already a batch)")
    p.add_argument("--group_batch_size", type=int, default=32,
                   help="Number of parts per group (B=20, parts from same assembly)")
    p.add_argument("--nepoch", type=int, default=200)
    p.add_argument("--npoints", type=int, default=1024)
    p.add_argument("--lr", type=float, default= 1e-3)  # 原来是1e-3
    p.add_argument("--step_size", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.5)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_cpu", action="store_true", default=False)
    p.add_argument("--gpu_id", type=str, default="7")
    # augmentation & contrastive
    p.add_argument("--rotation_mode", type=str, default="xyz", choices=["xyz", "yaw", "none"])
    p.add_argument("--no_jitter", action="store_true", default=False)
    p.add_argument("--lambda_con", type=float, default=0.1, help="weight for contrastive loss")
    p.add_argument("--lambda_region", type=float, default=0.1, help="weight for region contrastive loss")
    p.add_argument("--temperature", type=float, default=0.07)
    p.add_argument("--use_region_contrast", action="store_true", default=True, help="use region contrastive learning")
    # graphcut segmentation parameters
    p.add_argument("--use_graphcut_segmentation", action="store_true", default=True, help="use graphcut segmentation for grouping")
    p.add_argument("--graphcut_k", type=int, default=8, help="number of neighbors for graphcut")
    p.add_argument("--graphcut_c", type=float, default=0.5, help="merging threshold for graphcut")
    p.add_argument("--graphcut_min_size", type=int, default=20, help="minimum region size for graphcut")

    p.add_argument('--exp_name', type=str, default='GDANetFusion360BatchGroupContrast', help='')
    p.add_argument('--save_dir', help='日志保存路径', default='/data/cjj/projects/PointCloudLearning/dgcnn/experiment', type=str)
    # p.add_argument('--checkpoint_path', type=str, default='/data/cjj/projects/PointCloudLearning/dgcnn/experiment/12-08_10:25_Fusion360BatchGroupContrast/epoch119.pth', help="checkpoint path")

    # p.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    # p.add_argument('--emb_dims', type=int, default=1024, help='Dimension of embeddings')
    # p.add_argument('--k', type=int, default=20, help='Num of nearest neighbors to use')
    return p.parse_args()


def make_loaders(args):
    """
    创建使用BatchFusion360Dataset的数据加载器
    注意：DataLoader的batch_size应该设置为1，因为BatchFusion360Dataset的每个__getitem__已经返回一个批次
    """
    train_ds = BatchFusion360Dataset(
        root=args.dataset,
        batch_size=args.group_batch_size,  # 每个整体文件夹内的零件数量
        npoints=args.npoints,
        rotation_mode=args.rotation_mode,
        jitter=not args.no_jitter,
        center_unit=True,
        use_graphcut_segmentation=args.use_graphcut_segmentation,
        graphcut_k=args.graphcut_k,
        graphcut_c=args.graphcut_c,
        graphcut_min_size=args.graphcut_min_size
    )
    # DataLoader的batch_size设置为1，因为每个样本已经是一个批次了
    train_loader = DataLoader(train_ds, batch_size=args.batchSize, shuffle=True, num_workers=args.workers)
    return train_loader

@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    correct, total = 0, 0
    
    # 创建验证进度条
    eval_pbar = tqdm(loader, desc="[Eval]", leave=False, ncols=80)
    
    for batch in eval_pbar:
        pts, normals, label = batch

        pts = pts.to(device).transpose(2, 1)
        normals = normals.to(device).transpose(2, 1)
        label = label.to(device)

        logp = net(pts, normals)

        pred = logp.data.max(1)[1]
        correct += pred.eq(label.data).cpu().sum().item()
        total += label.size(0)
        
        # 更新验证进度条
        current_acc = correct / total if total > 0 else 0.0
        eval_pbar.set_postfix({'Acc': f'{current_acc:.4f}'})
    
    return (correct / total if total > 0 else 0.0)


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

    train_loader = make_loaders(args)

    # net = DGCNN(args).to(device)
    net = GDANET().to(device)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    os.makedirs(save_dir, exist_ok=True)
    logger = get_logger(os.path.join(save_dir, f'run_{args.exp_name}.log'))
    logger.info(f"Arguments: {args}")
    logger.info(f"model params: {sum(p.numel() for p in net.parameters()) / 1e6} M")
    logger.info(f"Network Structure: {str(net)}")
    logger.info(f"Dataset size: {len(train_loader.dataset)} groups")
    logger.info(f"Group batch size (parts per group): {args.group_batch_size}")

    # 安全地获取checkpoint_path，如果参数被注释掉则返回空字符串
    checkpoint_path = getattr(args, 'checkpoint_path', '')
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            net.load_state_dict(checkpoint['model_state_dict'], strict=False)
            if 'optimizer_state_dict' in checkpoint:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info("Loaded optimizer state from checkpoint")
            if 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info("Loaded scheduler state from checkpoint")
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                logger.info(f"Resuming from epoch {start_epoch}")
            else:
                start_epoch = 0
        else:
            net.load_state_dict(checkpoint, strict=False)
            logger.info(f"Loaded checkpoint from {checkpoint_path}")
            start_epoch = 0
        net.to(device)
    else:
        start_epoch = 0
        if checkpoint_path:
            logger.info(f"Checkpoint file not found at {checkpoint_path}, starting from scratch")

    for epoch in range(start_epoch, args.nepoch):
        net.train()
        running = {"loss": 0.0, "con": 0.0, "region_con": 0.0, "reg": 0.0}

        # 创建训练进度条
        train_pbar = tqdm(train_loader, 
                         desc=f"Epoch {epoch+1:03d}/{args.nepoch:03d} [Train]",
                         leave=False,
                         ncols=100)
        
        for batch_data in train_pbar:
            # BatchFusion360Dataset返回的数据格式：
            # 如果DataLoader的batch_size=1，则batch_data是一个tuple
            # 如果DataLoader的batch_size>1，则每个元素会被额外堆叠一层
            # 为了简化，我们假设batch_size=1，所以直接解包
            if len(batch_data) == 6:  # 包含分组信息
                batch_v1, batch_v2, batch_normal1, batch_normal2, batch_y, batch_segments = batch_data
            else:  # 不包含分组信息
                batch_v1, batch_v2, batch_normal1, batch_normal2, batch_y = batch_data
                batch_segments = None
            
            # 去除DataLoader添加的额外维度 (batch_size=1时添加的维度)
            batch_v1 = batch_v1.squeeze(0)
            batch_v2 = batch_v2.squeeze(0)
            batch_normal1 = batch_normal1.squeeze(0)
            batch_normal2 = batch_normal2.squeeze(0)
            batch_y = batch_y.squeeze(0)
            if batch_segments is not None:
                batch_segments = batch_segments.squeeze(0)
            
            # 数据形状: (group_batch_size, npoints, 3) -> (group_batch_size, 3, npoints)
            batch_v1 = batch_v1.to(device).transpose(2, 1)  # (B, 3, N)
            batch_v2 = batch_v2.to(device).transpose(2, 1)
            batch_normal1 = batch_normal1.to(device).transpose(2, 1)
            batch_normal2 = batch_normal2.to(device).transpose(2, 1)
            batch_y = batch_y.to(device)
            if batch_segments is not None:
                batch_segments = batch_segments.to(device)

            optimizer.zero_grad()

            # two views
            logp1, point_feat1 = net(batch_v1)
            logp2, point_feat2 = net(batch_v2)

            # point-level contrastive loss
            con = info_nce_loss(point_feat1, point_feat2, temperature=args.temperature)

            # region contrastive loss
            region_con = 0.0
            if args.use_region_contrast and batch_segments is not None:
                # 使用真正的点级特征 point_feat1, point_feat2 (B, 1024, N)
                region_con = region_contrastive_loss(point_feat1, point_feat2, batch_segments, temperature=args.temperature)

            # total loss
            loss = args.lambda_con * con + args.lambda_region * region_con
            loss.backward()
            optimizer.step()

            running["loss"] += float(loss.item())
            running["con"] += float(con.item())
            running["region_con"] += float(region_con if isinstance(region_con, torch.Tensor) else region_con)
            
            # 更新进度条显示
            current_batch = train_pbar.n
            current_loss = running["loss"] / max(1, current_batch)
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'LR': f'{scheduler.get_last_lr()[0]:.6f}'
            })

        train_loss = running["loss"] / max(1, len(train_loader))

        scheduler.step()

        logger.info(f"Epoch [{epoch+1:03d}/{args.nepoch:03d}] "
                       f"Train Loss: {train_loss:.4f} (Con {running['con']/max(1,len(train_loader)):.4f}, "
                       f"RegionCon {running['region_con']/max(1,len(train_loader)):.4f}) "
                       f"|LR: {scheduler.get_last_lr()[0]:.6f}")

        # save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
        }
        torch.save(checkpoint, os.path.join(save_dir, f"epoch{epoch+1:03d}.pth"))
        
        # 保存最新的checkpoint
        torch.save(checkpoint, os.path.join(save_dir, "latest.pth"))

    logger.info(f"Training done.")

if __name__ == "__main__":
    main()

