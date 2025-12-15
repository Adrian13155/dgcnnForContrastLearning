"""
来自:https://github.com/CentauriStar/Rotation-Invariant-Point-Cloud-Analysis/blob/main/Data/ModelNet.py
"""
import os
import sys
sys.path.append("/data/cjj/projects/PointCloudLearning/dgcnn")

import numpy as np
from torch.utils.data import Dataset
import torch
from torch_geometric.data import Data
import provider
from torch_geometric.loader import DataLoader
from sklearn.neighbors import NearestNeighbors
class DisjointSet:
    """并查集数据结构，用于图割算法"""
    def __init__(self, n):
        self.parent = np.arange(n)
        self.size = np.ones(n, dtype=int)

    def find(self, u):
        while self.parent[u] != u:
            self.parent[u] = self.parent[self.parent[u]]  # 路径压缩
            u = self.parent[u]
        return u

    def union(self, u, v):
        u = self.find(u)
        v = self.find(v)
        if u == v:
            return False
        if self.size[u] < self.size[v]:
            u, v = v, u
        self.parent[v] = u
        self.size[u] += self.size[v]
        return True


def segment_pointcloud(points, k=8, c=0.5, min_size=20):
    """
    基于Felzenszwalb图分割思想的点云分割
    
    Args:
        points: Nx3 numpy array 点云坐标
        k: 每个点邻居数量
        c: 合并阈值参数，越大区域越少
        min_size: 最小区域大小，太小的区域会被合并
    
    Returns:
        labels: 每个点的分割类别
    """
    N = points.shape[0]
    
    if N < k + 1:
        # 如果点数太少，直接返回单个分组
        return np.zeros(N, dtype=int)

    # 1. 计算k近邻图边
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(points)

    # 构建边列表：(点i, 点j, 权重)
    edges = []
    for i in range(N):
        for j in range(1, k+1):  # indices[i, 0]是自己，跳过
            neighbor = indices[i, j]
            dist = distances[i, j]
            edges.append((i, neighbor, dist))
    edges = np.array(edges, dtype=[('p1', int), ('p2', int), ('w', float)])

    # 2. 按权重排序边（距离越小越优先合并）
    edges.sort(order='w')

    # 3. 初始化并查集，每个点自己一个集合
    u = DisjointSet(N)

    # 4. 初始化阈值函数
    threshold = np.full(N, c)

    # 5. 依次遍历边，决定是否合并
    for edge in edges:
        a = u.find(edge['p1'])
        b = u.find(edge['p2'])
        if a != b:
            if edge['w'] <= threshold[a] and edge['w'] <= threshold[b]:
                merged = u.union(a, b)
                if merged:
                    a_new = u.find(a)
                    threshold[a_new] = edge['w'] + c / u.size[a_new]

    # 6. 合并小区域
    for edge in edges:
        a = u.find(edge['p1'])
        b = u.find(edge['p2'])
        if a != b:
            if u.size[a] < min_size or u.size[b] < min_size:
                u.union(a, b)

    # 7. 给每个点赋标签
    labels = np.zeros(N, dtype=int)
    label_map = {}
    label_id = 0
    for i in range(N):
        root = u.find(i)
        if root not in label_map:
            label_map[root] = label_id
            label_id += 1
        labels[i] = label_map[root]

    # print(f"分割得到 {label_id} 个区域")

    return labels

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: Nx3 '''
    dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def shuffle_points(data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(data.shape[-2])
    np.random.shuffle(idx)
    return data[idx,:]

class ModelNetNormal(Dataset):
    def __init__(self, root, npoints=1024, split='train', normalize=True, normal_channel=False,
                 modelnet10=False, cache_size=15000, shuffle=True, transform=None, drop_out=False):
        self.root = root
        # self.batch_size = batch_size
        self.npoints = npoints
        self.normalize = normalize
        self.transform = transform
        self.shuffle = shuffle
        self.drop_out = drop_out
        if modelnet10:
            self.catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
        else:
            self.catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
        self.cat = [line.rstrip() for line in open(self.catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))
        self.normal_channel = normal_channel

        shape_ids = {}
        if modelnet10:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet10_test.txt'))]
        else:
            shape_ids['train'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_train.txt'))]
            shape_ids['test'] = [line.rstrip() for line in open(os.path.join(self.root, 'modelnet40_test.txt'))]
        assert (split == 'train' or split == 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [(shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt') for i
                         in range(len(shape_ids[split]))]

        self.cache_size = cache_size  # how many data points to cache in memory
        self.cache = {}  # from index to (point_set, cls) tuple

        self.class_num = 40

    def _augment_data(self, rotated_data, rotate=False, shiver=False, translate=False, jitter=False,
                      shuffle=True):
        if rotate:
            if self.normal_channel:
                rotated_data = provider.rotate_point_cloud_with_normal(rotated_data)
                rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
            else:
                rotated_data = provider.rotate_point_cloud(rotated_data)
                rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        if shiver:
            rotated_data[:, 0:3] = provider.random_scale_point_cloud(rotated_data[:, 0:3])
        if translate:
            rotated_data[:, 0:3] = provider.shift_point_cloud(rotated_data)
        if jitter:
            rotated_data[:, 0:3] = provider.jitter_point_cloud(rotated_data)

        if shuffle:
            return shuffle_points(rotated_data)
        else:
            return rotated_data


    def rotate(self, batch_data):
        if self.normal_channel:
            rotated_data = provider.rotate_point_cloud_with_normal(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud_with_normal(rotated_data)
        else:
            rotated_data = provider.rotate_point_cloud(batch_data)
            rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)

        return provider.shuffle_points(rotated_data)

    def __getitem__(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
            point_set = point_set[0:self.npoints, :]

        else:
            fn = self.datapath[index]
            cls = self.classes[self.datapath[index][0]]
            cls = np.array([cls]).astype(np.int32)
            point_set = np.loadtxt(fn[1], delimiter=',').astype(np.float32)
            # Take the first npoints
            point_set = point_set[0:self.npoints, :]
            if self.normalize:
                point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])
            if len(self.cache) < self.cache_size:
                self.cache[index] = (point_set, cls)

        if self.drop_out:
            point_set = random_point_dropout(point_set)

        points = self._augment_data(point_set, shuffle=self.shuffle)              # only shuffle by default
        
        lra = compute_LRA(torch.from_numpy(points[:, :3]).float().unsqueeze(0), True, nsample = 32)
        
        if self.normal_channel:
            data = Data(pos=torch.from_numpy(points[:, :3]).float(), y=torch.from_numpy(cls).long(),
                    norm=torch.from_numpy(points[:, 3:]).float(), LRA=lra.squeeze(0))
        else:
            data = Data(pos=torch.from_numpy(points[:, :3]).float(), y=torch.from_numpy(cls).long(),
                        LRA=lra.squeeze(0))
            data.id = index

        if self.transform is not None:
            data = self.transform(data)
        
        return data

    def __len__(self):
        return len(self.datapath)


class ContrastLearningModelNetNormal(Dataset):
    """
    基于 ModelNetNormal 构造对比学习数据，每次返回两组增强视图及其法向量。
    use_normals表示是否信任/使用文件里自带的法向，如果为False则是用 LRA（局部 PCA）从坐标估计法向
    """

    def __init__(self,
                 root,
                 npoints: int = 1024,
                 split: str = 'train',
                 normalize: bool = True,
                 use_normals: bool = True,
                 modelnet10: bool = False,
                 cache_size: int = 15000,
                 drop_out: bool = False,
                 rotate: bool = True,
                 rotate_perturb: bool = True,
                 jitter: bool = True,
                 shuffle: bool = True,
                 lra_nsample: int = 32,
                 use_graphcut_segmentation: bool = False,
                 graphcut_k: int = 8,
                 graphcut_c: float = 0.5,
                 graphcut_min_size: int = 20):
        self.root = root
        self.npoints = npoints
        self.split = split
        self.normalize = normalize
        self.use_normals = use_normals
        self.modelnet10 = modelnet10
        self.cache_size = cache_size
        self.cache = {}
        self.drop_out = drop_out
        self.rotate = rotate
        self.rotate_perturb = rotate_perturb
        self.jitter = jitter
        self.shuffle = shuffle
        self.lra_nsample = lra_nsample
        self.use_graphcut_segmentation = use_graphcut_segmentation
        self.graphcut_k = graphcut_k
        self.graphcut_c = graphcut_c
        self.graphcut_min_size = graphcut_min_size

        if modelnet10:
            catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
            train_list = 'modelnet10_train.txt'
            test_list = 'modelnet10_test.txt'
        else:
            catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
            train_list = 'modelnet40_train.txt'
            test_list = 'modelnet40_test.txt'

        self.cat = [line.rstrip() for line in open(catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {
            'train': [line.rstrip() for line in open(os.path.join(self.root, train_list))],
            'test': [line.rstrip() for line in open(os.path.join(self.root, test_list))]
        }
        assert split in ('train', 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [
            (shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt')
            for i in range(len(shape_ids[split]))
        ]

    def __len__(self):
        return len(self.datapath)

    def _maybe_cache(self, index, point_set, cls):
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls)

    def _load_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            shape_name, file_path = self.datapath[index]
            cls = np.array([self.classes[shape_name]], dtype=np.int32)
            point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
            self._maybe_cache(index, point_set, cls)
        return point_set.copy(), cls

    def _dropout(self, coords: np.ndarray, normals: np.ndarray):
        dropout_ratio = np.random.random() * 0.875
        drop_idx = np.where(np.random.random(coords.shape[0]) <= dropout_ratio)[0]
        if drop_idx.size > 0:
            coords[drop_idx, :] = coords[0, :]
            if normals is not None:
                normals[drop_idx, :] = normals[0, :]
        return coords, normals

    def _maybe_sample(self, pts: np.ndarray, normals: np.ndarray):
        n = pts.shape[0]
        if n == self.npoints:
            return pts, normals
        if n > self.npoints:
            idx = np.random.choice(n, self.npoints, replace=False)
        else:
            pad_idx = np.random.choice(n, self.npoints - n, replace=True)
            idx = np.concatenate([np.arange(n), pad_idx], axis=0)
        pts_new = pts[idx, :]
        normals_new = normals[idx, :] if normals is not None else None
        return pts_new, normals_new

    def _compute_normals_if_needed(self, pts: np.ndarray) -> np.ndarray:
        if self.use_normals:
            return None
        with torch.no_grad():
            lra = compute_LRA(torch.from_numpy(pts).float().unsqueeze(0),
                              weighting=True, nsample=self.lra_nsample)
        normals = lra.squeeze(0).cpu().numpy().astype(np.float32)
        return normals

    def _augment_once(self, pts: np.ndarray, normals: np.ndarray):
        data = np.concatenate([pts, normals], axis=1)[np.newaxis, ...]  # (1, N, 6)
        do_rotate = self.rotate and self.split == 'train'
        do_rotate_perturb = self.rotate_perturb and self.split == 'train'
        do_jitter = self.jitter and self.split == 'train'
        do_shuffle = self.shuffle

        if do_rotate:
            data = provider.rotate_point_cloud_with_normal(data)
        if do_rotate_perturb:
            data = provider.rotate_perturbation_point_cloud_with_normal(data)
        if do_jitter:
            jittered = provider.jitter_point_cloud(data[:, :, 0:3])
            data[:, :, 0:3] = jittered
        if do_shuffle:
            data = provider.shuffle_points(data)

        augmented_pts = data[0, :, 0:3]
        augmented_normals = data[0, :, 3:6]
        norm = np.linalg.norm(augmented_normals, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-6, None)
        augmented_normals = augmented_normals / norm

        return augmented_pts.astype(np.float32), augmented_normals.astype(np.float32)

    def __getitem__(self, index):
        point_set, cls = self._load_item(index)
        label = int(cls[0])

        coords = point_set[:, :3].astype(np.float32)
        normals = point_set[:, 3:6].astype(np.float32) if (self.use_normals and point_set.shape[1] >= 6) else None

        if self.normalize:
            coords = pc_normalize(coords)

        # 如果需要分组，先对原始点云进行分组
        segments = None
        if self.use_graphcut_segmentation:
            segments = segment_pointcloud(coords, k=self.graphcut_k,
                                       c=self.graphcut_c,
                                       min_size=self.graphcut_min_size)

        if self.drop_out:
            coords, normals = self._dropout(coords, normals)

        if normals is None:
            normals = self._compute_normals_if_needed(coords)
        if normals is None:
            raise ValueError("无法获取法向量：请将 use_normals=False 以启用 LRA 估计，或确保数据文件包含法向。")

        coords, normals = self._maybe_sample(coords, normals)

        v1_pts, v1_normals = self._augment_once(coords.copy(), normals.copy())
        v2_pts, v2_normals = self._augment_once(coords.copy(), normals.copy())

        v1 = torch.from_numpy(v1_pts).float()
        v2 = torch.from_numpy(v2_pts).float()
        normal1 = torch.from_numpy(v1_normals).float()
        normal2 = torch.from_numpy(v2_normals).float()
        y = torch.tensor(label, dtype=torch.long)

        if segments is not None:
            segments_tensor = torch.from_numpy(segments).long()
            return v1, v2, normal1, normal2, y, segments_tensor
        else:
            return v1, v2, normal1, normal2, y


class ModelNet40Normal(Dataset):
    """
    读取 ModelNet40 点云及法向量，返回单视图样本。
    """

    def __init__(self,
                 root,
                 split: str = "train",
                 npoints: int = 1024,
                 normalize: bool = True,
                 use_normals: bool = True,
                 modelnet10: bool = False,
                 cache_size: int = 15000,
                 drop_out: bool = False,
                 rotate: bool = True,
                 rotate_perturb: bool = True,
                 jitter: bool = True,
                 shuffle: bool = True,
                 lra_nsample: int = 32):
        self.root = root
        self.split = split
        self.npoints = npoints
        self.normalize = normalize
        self.use_normals = use_normals
        self.modelnet10 = modelnet10
        self.cache_size = cache_size
        self.cache = {}
        self.drop_out = drop_out
        self.rotate = rotate
        self.rotate_perturb = rotate_perturb
        self.jitter = jitter
        self.shuffle = shuffle
        self.lra_nsample = lra_nsample

        if modelnet10:
            catfile = os.path.join(self.root, 'modelnet10_shape_names.txt')
            train_list = 'modelnet10_train.txt'
            test_list = 'modelnet10_test.txt'
        else:
            catfile = os.path.join(self.root, 'modelnet40_shape_names.txt')
            train_list = 'modelnet40_train.txt'
            test_list = 'modelnet40_test.txt'

        self.cat = [line.rstrip() for line in open(catfile)]
        self.classes = dict(zip(self.cat, range(len(self.cat))))

        shape_ids = {
            'train': [line.rstrip() for line in open(os.path.join(self.root, train_list))],
            'test': [line.rstrip() for line in open(os.path.join(self.root, test_list))]
        }
        assert split in ('train', 'test')
        shape_names = ['_'.join(x.split('_')[0:-1]) for x in shape_ids[split]]
        self.datapath = [
            (shape_names[i], os.path.join(self.root, shape_names[i], shape_ids[split][i]) + '.txt')
            for i in range(len(shape_ids[split]))
        ]

    def __len__(self):
        return len(self.datapath)

    def _maybe_cache(self, index, point_set, cls):
        if len(self.cache) < self.cache_size:
            self.cache[index] = (point_set, cls)

    def _load_item(self, index):
        if index in self.cache:
            point_set, cls = self.cache[index]
        else:
            shape_name, file_path = self.datapath[index]
            cls = np.array([self.classes[shape_name]], dtype=np.int32)
            point_set = np.loadtxt(file_path, delimiter=',').astype(np.float32)
            self._maybe_cache(index, point_set, cls)
        return point_set.copy(), cls

    def _maybe_sample(self, pts: np.ndarray, normals: np.ndarray):
        n = pts.shape[0]
        if n == self.npoints:
            return pts, normals
        if n > self.npoints:
            idx = np.random.choice(n, self.npoints, replace=False)
        else:
            pad_idx = np.random.choice(n, self.npoints - n, replace=True)
            idx = np.concatenate([np.arange(n), pad_idx], axis=0)
        pts_new = pts[idx, :]
        normals_new = normals[idx, :] if normals is not None else None
        return pts_new, normals_new

    def _compute_normals_if_needed(self, pts: np.ndarray) -> np.ndarray:
        if self.use_normals:
            return None
        with torch.no_grad():
            lra = compute_LRA(torch.from_numpy(pts).float().unsqueeze(0),
                              weighting=True, nsample=self.lra_nsample)
        normals = lra.squeeze(0).cpu().numpy().astype(np.float32)
        return normals

    def _apply_dropout(self, coords: np.ndarray, normals: np.ndarray):
        dropout_ratio = np.random.random() * 0.875
        drop_idx = np.where(np.random.random(coords.shape[0]) <= dropout_ratio)[0]
        if drop_idx.size > 0:
            coords[drop_idx, :] = coords[0, :]
            if normals is not None:
                normals[drop_idx, :] = normals[0, :]
        return coords, normals

    def _augment(self, pts: np.ndarray, normals: np.ndarray):
        data = np.concatenate([pts, normals], axis=1)[np.newaxis, ...]  # (1, N, 6)
        do_rotate = self.rotate and self.split == 'train'
        do_rotate_perturb = self.rotate_perturb and self.split == 'train'
        do_jitter = self.jitter and self.split == 'train'
        do_shuffle = self.shuffle

        if do_rotate:
            data = provider.rotate_point_cloud_with_normal(data)
        if do_rotate_perturb:
            data = provider.rotate_perturbation_point_cloud_with_normal(data)
        if do_jitter:
            jittered = provider.jitter_point_cloud(data[:, :, 0:3])
            data[:, :, 0:3] = jittered
        if do_shuffle:
            data = provider.shuffle_points(data)

        pts_aug = data[0, :, 0:3]
        normals_aug = data[0, :, 3:6]
        norm = np.linalg.norm(normals_aug, axis=1, keepdims=True)
        norm = np.clip(norm, 1e-6, None)
        normals_aug = normals_aug / norm
        return pts_aug.astype(np.float32), normals_aug.astype(np.float32)

    def __getitem__(self, index):
        point_set, cls = self._load_item(index)
        label = int(cls[0])

        coords = point_set[:, :3].astype(np.float32)
        normals = point_set[:, 3:6].astype(np.float32) if (self.use_normals and point_set.shape[1] >= 6) else None

        if self.normalize:
            coords = pc_normalize(coords)

        if self.drop_out and self.split == 'train':
            coords, normals = self._apply_dropout(coords, normals)

        if normals is None:
            normals = self._compute_normals_if_needed(coords)
        if normals is None:
            raise ValueError("无法获取法向量：请将 use_normals=False 以启用 LRA 估计，或确保数据文件包含法向。")

        coords, normals = self._maybe_sample(coords, normals)

        pts_aug, normals_aug = self._augment(coords, normals)

        pts_tensor = torch.from_numpy(pts_aug).float()
        normals_tensor = torch.from_numpy(normals_aug).float()
        label_tensor = torch.tensor(label, dtype=torch.long)

        return pts_tensor, normals_tensor, label_tensor


def compute_LRA(xyz, weighting=False, nsample = 64):
    dists = torch.cdist(xyz, xyz)

    dists, idx = torch.topk(dists, nsample, dim=-1, largest=False, sorted=False)
    dists = dists.unsqueeze(-1)

    group_xyz = index_points(xyz, idx)
    group_xyz = group_xyz - xyz.unsqueeze(2)

    if weighting:
        dists_max, _ = dists.max(dim=2, keepdim=True)
        dists = dists_max - dists
        dists_sum = dists.sum(dim=2, keepdim=True)
        weights = dists / dists_sum
        weights[weights != weights] = 1.0
        M = torch.matmul(group_xyz.transpose(3,2), weights*group_xyz)
    else:
        M = torch.matmul(group_xyz.transpose(3,2), group_xyz)

    eigen_values, vec = torch.linalg.eigh(M)

    LRA = vec[:,:,:,0]
    LRA_length = torch.norm(LRA, dim=-1, keepdim=True)
    LRA = LRA / LRA_length
    return LRA # B N 3

def index_points(points, idx):
    """

    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)

    new_points = points[batch_indices, idx, :]      
    return new_points

if __name__ == "__main__":
    # dataset = ModelNet40Normal(root="/data/datasets/pointnet/modelnet40_normal_resampled", split="train", 
    # npoints=1024, normalize=True, use_normals=True,
    #  modelnet10=False, cache_size=15000, shuffle=True, drop_out=False)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for batch_data in dataloader:
    #     pts, normals, y = batch_data
    #     print(pts.shape, normals.shape, y.shape)
    #     break
    dataset = ContrastLearningModelNetNormal(root="/data/datasets/pointnet/modelnet40_normal_resampled", split="train", 
    npoints=1024, normalize=True, use_normals=True,
     modelnet10=False, cache_size=15000, shuffle=True, drop_out=False)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    for batch_data in dataloader:
        v1,v2,normal1,normal2,y = batch_data
        print(v1.shape, v2.shape, normal1.shape, normal2.shape, y.shape)
        break

    # dataset = ModelNetNormal(root="/data/datasets/pointnet/modelnet40_normal_resampled", split="train", 
    # npoints=1024, normalize=True, normal_channel=True,
    #  modelnet10=False, cache_size=15000, shuffle=True, transform=None, drop_out=False)
    # dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # for batch_data in dataloader:
    #     print(batch_data)
    #     print(batch_data.pos.shape, batch_data.norm.shape, batch_data.y.shape, batch_data.LRA.shape)
    #     batch_size = batch_data.batch.max() + 1
    #     xyz = batch_data.pos.view(batch_size, -1, 3).permute(0, 2, 1)
    #     normal = batch_data.norm.view(batch_size, -1, 3).permute(0, 2, 1)
    #     print("xyz shape:", xyz.shape)
    #     print("normal shape:", normal.shape)
    #     print("batch_size:", batch_size)
    #     print("batch_data.x:", batch_data.x)
    #     break