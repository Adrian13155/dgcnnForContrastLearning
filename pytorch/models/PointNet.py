import torch
import torch.nn as nn
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        # x: (B, 3, N)
        b = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(3, device=x.device).view(1, 9).repeat(b, 1)
        x = x + iden
        return x.view(-1, 3, 3)

class STNkd(nn.Module):
    def __init__(self, k=64):
        super().__init__()
        self.conv1 = nn.Conv1d(k, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.k = k

    def forward(self, x):
        # x: (B, k, N)
        b = x.size(0)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2)[0]

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.eye(self.k, device=x.device).view(1, self.k*self.k).repeat(b, 1)
        x = x + iden
        return x.view(-1, self.k, self.k)

class PointNetfeat(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False):
        super().__init__()
        self.stn = STN3d()
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.conv3 = nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        # x: (B, 3, N)
        n_pts = x.size(2)
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))  # (B, 64, N)

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        x = F.relu(self.bn2(self.conv2(x)))  # (B, 128, N)
        x = self.bn3(self.conv3(x))  # (B, 1024, N)
        
        # 返回点级特征，不进行池化
        return x, trans, trans_feat  # (B, 1024, N)

class ContrastPointNet(nn.Module):

    def __init__(self, k: int = 40, feature_transform: bool = False):
        super().__init__()
        self.feature_transform = feature_transform
        self.feat = PointNetfeat(global_feat=True, feature_transform=feature_transform)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        # x: (B, 3, N)
        point_feat, trans, trans_feat = self.feat(x)  # point_feat: (B, 1024, N)
        
        # 对点级特征进行最大池化得到全局特征
        h = torch.max(point_feat, 2, keepdim=True)[0]  # (B, 1024, 1)
        h = h.view(-1, 1024)  # (B, 1024)
        
        # 分类
        xcls = F.relu(self.bn1(self.fc1(h)))
        xcls = F.relu(self.bn2(self.dropout(self.fc2(xcls))))
        logits = self.fc3(xcls)
        log_probs = F.log_softmax(logits, dim=1)


        return log_probs, point_feat, trans_feat