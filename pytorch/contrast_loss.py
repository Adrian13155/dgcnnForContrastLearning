import torch
import torch.nn.functional as F

def info_nce_loss(point_feat1: torch.Tensor, point_feat2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    点级别InfoNCE损失 - 正样本是不同视角的相同点，其他点为负样本
    
    Args:
        point_feat1: (B, D, N) 第一个视图的点级别特征
        point_feat2: (B, D, N) 第二个视图的点级别特征
        temperature: 温度参数
    
    Returns:
        loss: 点级别对比损失
    """
    B, D, N = point_feat1.shape
    device = point_feat1.device
    
    # 归一化特征
    feat1_norm = F.normalize(point_feat1, dim=1)  # (B, D, N)
    feat2_norm = F.normalize(point_feat2, dim=1)  # (B, D, N)
    
    total_loss = 0.0
    total_points = 0
    
    for b in range(B):
        # 当前点云的特征
        f1 = feat1_norm[b]  # (D, N)
        f2 = feat2_norm[b]  # (D, N)
        
        # 计算相似度矩阵: f1[i] 与 f2[j] 的相似度
        sim = torch.matmul(f1.t(), f2) / temperature  # (N, N)
        
        # 正样本：相同点的不同视角 (对角线元素)
        pos_sim = torch.diag(sim)  # (N,) - 第i个点与第i个点的相似度
        
        # 负样本：不同点的相似度 (非对角线元素)
        # 对于每个点i，负样本是与所有其他点的相似度
        pos_exp = torch.exp(pos_sim)  # (N,)
        
        # 计算每个点的负样本相似度
        # 对于点i，负样本是与所有其他点的相似度
        neg_exp_sum = torch.exp(sim).sum(dim=1) - pos_exp  # (N,) - 排除对角线
        
        # 计算损失
        denominator = pos_exp + neg_exp_sum
        losses = -torch.log(pos_exp / denominator)
        
        total_loss += losses.sum()
        total_points += N
    
    return total_loss / max(total_points, 1)

def region_contrastive_loss(point_feat1: torch.Tensor, point_feat2: torch.Tensor, 
                           segments: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """
    计算分组对比损失 - 超快速向量化版本
    
    Args:
        point_feat1: (B, D, N) 第一个视图的点级别特征
        point_feat2: (B, D, N) 第二个视图的点级别特征  
        segments: (B, N) 每个点的分组ID
        temperature: 温度参数
    
    Returns:
        loss: 分组对比损失
    """
    B, D, N = point_feat1.shape
    device = point_feat1.device
    
    # 归一化特征
    feat1_norm = F.normalize(point_feat1, dim=1)  # (B, D, N)
    feat2_norm = F.normalize(point_feat2, dim=1)  # (B, D, N)
    
    total_loss = 0.0
    total_pairs = 0
    
    for b in range(B):
        # 当前点云的特征和分组
        f1 = feat1_norm[b]  # (D, N)
        f2 = feat2_norm[b]  # (D, N)
        seg = segments[b]    # (N,)
        
        # 计算相似度矩阵
        sim = torch.matmul(f1.t(), f2) / temperature  # (N, N)
        
        # 获取唯一的分组ID
        unique_segments = torch.unique(seg)
        
        for seg_id in unique_segments:
            # 找到属于当前分组的点
            mask = (seg == seg_id)
            seg_indices = torch.where(mask)[0]
            
            if len(seg_indices) < 2:  # 分组中至少需要2个点
                continue
            
            # 完全向量化计算
            n_seg = len(seg_indices)
            seg_sim = sim[seg_indices][:, seg_indices]  # (n_seg, n_seg)
            
            # 创建上三角掩码
            triu_mask = torch.triu(torch.ones(n_seg, n_seg, device=device), diagonal=1).bool()
            
            if triu_mask.sum() == 0:
                continue
                
            # 正样本相似度
            pos_sim = seg_sim[triu_mask]  # (n_pairs,)
            
            # 负样本相似度（与其他分组的点）
            neg_mask = (seg != seg_id)
            if neg_mask.sum() == 0:
                continue
                
            neg_indices = torch.where(neg_mask)[0]
            neg_sim = sim[seg_indices][:, neg_indices]  # (n_seg, n_neg)
            
            # 完全向量化损失计算
            pos_exp = torch.exp(pos_sim)  # (n_pairs,)
            
            # 对每个正样本点，计算与所有负样本的相似度
            row_indices = torch.where(triu_mask)[0]  # 正样本的行索引
            neg_exp_sum = torch.exp(neg_sim[row_indices]).sum(dim=1)  # (n_pairs,)
            
            # 计算损失
            denominator = pos_exp + neg_exp_sum
            losses = -torch.log(pos_exp / denominator)
            
            total_loss += losses.sum()
            total_pairs += len(losses)
    
    return total_loss / max(total_pairs, 1)