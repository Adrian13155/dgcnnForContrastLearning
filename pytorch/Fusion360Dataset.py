import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import open3d as o3d
import os.path as osp
import numpy as np
import torch
import torch.utils.data as data
from typing import Tuple, List
import glob
from torch.utils.data import Dataset
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

def _center_and_scale_unit_sphere(x: np.ndarray) -> np.ndarray:
    """将点云中心化并缩放到单位球面上"""
    # x: (N, 3)
    centroid = np.mean(x, axis=0, keepdims=True)
    x = x - centroid
    m = np.max(np.linalg.norm(x, axis=1))
    if m > 0:
        x = x / m
    return x

def _rotation_matrix_xyz(ax: float, ay: float, az: float) -> np.ndarray:
    """随机 xyz 轴随机角度增强（依次绕 X、Y、Z 旋转随机角度）"""
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(ax), -np.sin(ax)],
                   [0, np.sin(ax), np.cos(ax)]])
    Ry = np.array([[np.cos(ay), 0, np.sin(ay)],
                   [0, 1, 0],
                   [-np.sin(ay), 0, np.cos(ay)]])
    Rz = np.array([[np.cos(az), -np.sin(az), 0],
                   [np.sin(az), np.cos(az), 0],
                   [0, 0, 1]])
    return Rz @ Ry @ Rx

def random_rotate_xyz(points: np.ndarray) -> np.ndarray:
    """随机旋转点云"""
    angles = np.random.uniform(0, 2*np.pi, 3)
    R = _rotation_matrix_xyz(*angles)
    return points @ R.T

def jitter_points(points: np.ndarray, sigma: float = 0.01, clip: float = 0.05) -> np.ndarray:
    """对点云添加随机噪声"""
    N, C = points.shape
    assert clip > 0, "clip must be positive"
    jitter = np.random.randn(N, C) * sigma
    jitter = np.clip(jitter, -clip, clip)
    return points + jitter

def _estimate_normals(mesh: o3d.geometry.TriangleMesh, npoints: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    从三角网格计算顶点法向量，并返回采样后的点云和法向量。

    Args:
        mesh: Open3D三角网格对象
        npoints: 采样点数
    Returns:
        points: 采样后的点云坐标 (npoints, 3)
        normals: 对应的法向量 (npoints, 3)
    """
    # 计算顶点法向量
    mesh.compute_vertex_normals()

    # 获取顶点坐标和法向量
    points = np.asarray(mesh.vertices, dtype=np.float32)
    normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

    # 如果点数不够，进行重复采样；如果太多，进行随机采样
    N = points.shape[0]
    if N >= npoints:
        # 随机采样
        idx = np.random.choice(N, npoints, replace=False)
    else:
        # 重复采样
        idx = np.random.choice(N, npoints, replace=True)

    return points[idx], normals[idx]

def _collect_obj_files(root_dir: str) -> List[str]:
    """
    收集所有obj文件路径，排除assembly.obj

    Args:
        root_dir: 数据集根目录
    Returns:
        obj文件路径列表
    """
    obj_files = []
    # 递归查找所有obj文件
    for obj_path in glob.glob(osp.join(root_dir, "**", "*.obj"), recursive=True):
        # 排除assembly.obj文件
        if not osp.basename(obj_path).startswith("assembly"):
            obj_files.append(obj_path)
    return sorted(obj_files)

class Fusion360Dataset(data.Dataset):
    """
    Fusion360数据集类，用于加载Fusion360 Gallery数据集中的OBJ文件
    """
    def __init__(self,
                 root: str,
                 npoints: int = 1024,
                 augment: bool = True,
                 center_unit: bool = True,
                 rotation_mode: str = "xyz"):
        """
        Args:
            root: 数据集根目录路径
            npoints: 每个样本的点数
            augment: 是否进行数据增强
            center_unit: 是否中心化和单位化
            rotation_mode: 旋转增强模式 ("xyz" 或 None)
        """
        super().__init__()
        self.root = root
        self.npoints = npoints
        self.augment = augment
        self.center_unit = center_unit
        self.rotation_mode = rotation_mode

        # 收集所有obj文件
        self.obj_files = _collect_obj_files(root)
        # print(f"找到 {len(self.obj_files)} 个OBJ文件")

        if len(self.obj_files) == 0:
            raise ValueError(f"在 {root} 下没有找到OBJ文件")

    def __len__(self) -> int:
        return len(self.obj_files)

    def _maybe_sample_n(self, x: np.ndarray) -> np.ndarray:
        """采样或填充点云到指定点数"""
        N = x.shape[0]
        if N == self.npoints:
            return x
        if N > self.npoints:
            # 随机采样
            idx = np.random.choice(N, self.npoints, replace=False)
            return x[idx, :]
        # 填充：通过重复随机点
        pad_idx = np.random.choice(N, self.npoints - N, replace=True)
        return np.concatenate([x, x[pad_idx]], axis=0)

    def __getitem__(self, idx: int):
        """
        加载单个OBJ文件并返回处理后的点云

        Returns:
            pts: 点云坐标 (npoints, 3)
            normals: 点云法向量 (npoints, 3)
            label: 标签（Fusion360数据集没有标签，返回-1）
        """
        obj_path = self.obj_files[idx]

        try:
            # 读取OBJ文件
            mesh = o3d.io.read_triangle_mesh(obj_path)
            if not mesh.has_vertices():
                raise RuntimeError(f"网格无顶点: {obj_path}")

            # 从网格计算法向量并采样
            pts, normals = _estimate_normals(mesh, self.npoints)

            # 数据增强（训练时）
            if self.augment:
                if self.rotation_mode == "xyz":
                    # 对点云和法向量应用相同的旋转
                    angles = np.random.uniform(0, 2*np.pi, 3)
                    R = _rotation_matrix_xyz(*angles)
                    pts = pts @ R.T
                    normals = normals @ R.T
                pts = jitter_points(pts, sigma=0.01, clip=0.02)
                # 注意：法向量在添加噪声后不需要额外处理

            # 中心化和单位化
            if self.center_unit:
                pts = _center_and_scale_unit_sphere(pts)

            # Fusion360数据集没有标签，返回-1作为占位符
            label = -1

            return (
                torch.from_numpy(pts).float(),
                torch.from_numpy(normals).float(),
                torch.tensor(label, dtype=torch.long)
            )

        except Exception as e:
            print(f"加载文件失败 {obj_path}: {e}")
            # 返回零张量作为错误处理
            pts = torch.zeros(self.npoints, 3, dtype=torch.float32)
            normals = torch.zeros(self.npoints, 3, dtype=torch.float32)
            label = torch.tensor(-1, dtype=torch.long)
            return pts, normals, label


class ContrastiveFusion360Dataset(Dataset):
    """
    对比学习Fusion360数据集类，用于生成两个增强视图的点云数据
    """
    def __init__(self,
                 root: str,
                 npoints: int = 1024,
                 rotation_mode: str = "xyz",
                 jitter: bool = True,
                 center_unit: bool = True,
                 use_graphcut_segmentation: bool = False,
                 graphcut_k: int = 8,
                 graphcut_c: float = 0.5,
                 graphcut_min_size: int = 20):
        """
        Args:
            root: 数据集根目录路径
            npoints: 每个样本的点数
            rotation_mode: 旋转增强模式 ("xyz" 或 "none")
            jitter: 是否添加随机噪声
            center_unit: 是否中心化和单位化
            use_graphcut_segmentation: 是否使用图割进行点云分割
            graphcut_k: 图割算法中每个点的邻居数
            graphcut_c: 图割算法的合并阈值参数
            graphcut_min_size: 图割算法的最小区域大小
        """
        super().__init__()
        self.root = root
        self.npoints = npoints
        self.rotation_mode = rotation_mode
        self.jitter = jitter
        self.center_unit = center_unit
        self.use_graphcut_segmentation = use_graphcut_segmentation
        self.graphcut_k = graphcut_k
        self.graphcut_c = graphcut_c
        self.graphcut_min_size = graphcut_min_size

        # 收集所有obj文件
        self.obj_files = _collect_obj_files(root)

        if len(self.obj_files) == 0:
            raise ValueError(f"在 {root} 下没有找到OBJ文件")

    def __len__(self) -> int:
        return len(self.obj_files)

    def _prep(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理点云，返回处理后的点云和对应的分组标签

        Returns:
            pts_processed: 处理后的点云
            segments: 对应的分组标签
        """
        if self.center_unit:
            pts = _center_and_scale_unit_sphere(pts)

        N = pts.shape[0]
        segments = None

        # 如果需要分组，先对原始点云进行分组
        if self.use_graphcut_segmentation:
            segments = segment_pointcloud(pts, k=self.graphcut_k,
                                       c=self.graphcut_c,
                                       min_size=self.graphcut_min_size)

        # 处理点数
        if N > self.npoints:
            idx = np.random.choice(N, self.npoints, replace=False)
            pts = pts[idx, :]
            if segments is not None:
                segments = segments[idx]
        elif N < self.npoints:
            pad_idx = np.random.choice(N, self.npoints - N, replace=True)
            pts = np.concatenate([pts, pts[pad_idx]], axis=0)
            if segments is not None:
                segments = np.concatenate([segments, segments[pad_idx]], axis=0)

        return pts, segments

    def _augment_once(self, pts: np.ndarray) -> np.ndarray:
        """对点云应用一次数据增强"""
        # 旋转增强
        if self.rotation_mode == "xyz":
            pts = random_rotate_xyz(pts)
        elif self.rotation_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")

        # 添加噪声
        if self.jitter:
            pts = jitter_points(pts, sigma=0.01, clip=0.02)

        return pts

    def _augment_once_with_normals(self, pts: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对点云和法向量同时应用一次数据增强"""
        # 旋转增强
        if self.rotation_mode == "xyz":
            pts = random_rotate_xyz(pts)
            # 对法向量应用相同的旋转
            angles = np.random.uniform(0, 2*np.pi, 3)
            R = _rotation_matrix_xyz(*angles)
            normals = normals @ R.T
        elif self.rotation_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")

        # 添加噪声（只对点云添加噪声，法向量保持不变）
        if self.jitter:
            pts = jitter_points(pts, sigma=0.01, clip=0.02)

        return pts, normals

    def __getitem__(self, idx: int):
        """
        加载单个OBJ文件并返回两个增强视图的对比学习数据

        Returns:
            v1: 第一个视图的点云坐标 (npoints, 3)
            v2: 第二个视图的点云坐标 (npoints, 3)
            normal1: 第一个视图的法向量 (npoints, 3)
            normal2: 第二个视图的法向量 (npoints, 3)
            y: 标签（Fusion360数据集没有标签，返回-1）
            segments_tensor: 分组标签（如果启用分组，否则不返回）
        """
        obj_path = self.obj_files[idx]

        try:
            # 读取OBJ文件
            mesh = o3d.io.read_triangle_mesh(obj_path)
            if not mesh.has_vertices():
                raise RuntimeError(f"网格无顶点: {obj_path}")

            # 计算顶点法向量
            mesh.compute_vertex_normals()

            # 获取顶点坐标和法向量
            pts = np.asarray(mesh.vertices, dtype=np.float32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

            # 采样到指定点数
            N = pts.shape[0]
            if N >= self.npoints:
                # 随机采样
                idx = np.random.choice(N, self.npoints, replace=False)
            else:
                # 重复采样
                idx = np.random.choice(N, self.npoints, replace=True)
            pts = pts[idx]
            normals = normals[idx]

            # 预处理并获取分组信息
            base, segments = self._prep(pts)

            # 生成两个增强视图（点云和法向量同时增强）
            v1, normal1 = self._augment_once_with_normals(base.copy(), normals.copy())
            v2, normal2 = self._augment_once_with_normals(base.copy(), normals.copy())

            # 转换为torch张量
            v1 = torch.from_numpy(v1).float()
            v2 = torch.from_numpy(v2).float()
            normal1 = torch.from_numpy(normal1).float()
            normal2 = torch.from_numpy(normal2).float()
            y = torch.tensor(-1, dtype=torch.long)  # Fusion360没有标签

            # 返回分组信息
            if self.use_graphcut_segmentation and segments is not None:
                segments_tensor = torch.from_numpy(segments).long()
                return v1, v2, normal1, normal2, y, segments_tensor

            return v1, v2, normal1, normal2, y

        except Exception as e:
            print(f"加载文件失败 {obj_path}: {e}")
            # 返回零张量作为错误处理
            pts = torch.zeros(self.npoints, 3, dtype=torch.float32)
            normals = torch.zeros(self.npoints, 3, dtype=torch.float32)
            y = torch.tensor(-1, dtype=torch.long)
            if self.use_graphcut_segmentation:
                segments_tensor = torch.zeros(self.npoints, dtype=torch.long)
                return pts, pts, normals, normals, y, segments_tensor
            return pts, pts, normals, normals, y

class BatchFusion360Dataset(Dataset):
    """
    批量Fusion360数据集类，用于生成来自同一个整体（文件夹）的多个零件数据的对比学习数据
    一个batch内的数据都来自于同一个整体文件夹
    """
    def __init__(self,
                 root: str,
                 batch_size: int = 20,
                 npoints: int = 1024,
                 rotation_mode: str = "xyz",
                 jitter: bool = True,
                 center_unit: bool = True,
                 use_graphcut_segmentation: bool = False,
                 graphcut_k: int = 8,
                 graphcut_c: float = 0.5,
                 graphcut_min_size: int = 20):
        """
        Args:
            root: 数据集根目录路径
            batch_size: 每个整体的零件数量（B=20）
            npoints: 每个样本的点数
            rotation_mode: 旋转增强模式 ("xyz" 或 "none")
            jitter: 是否添加随机噪声
            center_unit: 是否中心化和单位化
            use_graphcut_segmentation: 是否使用图割进行点云分割
            graphcut_k: 图割算法中每个点的邻居数
            graphcut_c: 图割算法的合并阈值参数
            graphcut_min_size: 图割算法的最小区域大小
        """
        super().__init__()
        self.root = root
        self.batch_size = batch_size
        self.npoints = npoints
        self.rotation_mode = rotation_mode
        self.jitter = jitter
        self.center_unit = center_unit
        self.use_graphcut_segmentation = use_graphcut_segmentation
        self.graphcut_k = graphcut_k
        self.graphcut_c = graphcut_c
        self.graphcut_min_size = graphcut_min_size

        # 收集所有整体文件夹
        self.group_dirs = []
        self.group_obj_files = {}  # group_dir -> list of obj files

        if not osp.exists(root):
            raise ValueError(f"数据集根目录不存在: {root}")

        # 获取所有一级子目录
        for item in os.listdir(root):
            item_path = osp.join(root, item)
            if osp.isdir(item_path):
                # 收集该文件夹下的所有obj文件
                obj_files = []
                for obj_path in glob.glob(osp.join(item_path, "**", "*.obj"), recursive=True):
                    # 排除assembly.obj文件
                    if not osp.basename(obj_path).startswith("assembly"):
                        obj_files.append(obj_path)

                if len(obj_files) > 0:
                    self.group_dirs.append(item_path)
                    self.group_obj_files[item_path] = sorted(obj_files)

        if len(self.group_dirs) == 0:
            raise ValueError(f"在 {root} 下没有找到包含OBJ文件的文件夹")

        print(f"找到 {len(self.group_dirs)} 个整体文件夹")

    def __len__(self) -> int:
        return len(self.group_dirs)

    def _load_single_obj(self, obj_path: str):
        """
        加载单个OBJ文件并返回处理后的数据
        返回格式与ContrastiveFusion360Dataset.__getitem__相同
        """
        try:
            # 读取OBJ文件
            mesh = o3d.io.read_triangle_mesh(obj_path)
            if not mesh.has_vertices():
                raise RuntimeError(f"网格无顶点: {obj_path}")

            # 计算顶点法向量
            mesh.compute_vertex_normals()

            # 获取顶点坐标和法向量
            pts = np.asarray(mesh.vertices, dtype=np.float32)
            normals = np.asarray(mesh.vertex_normals, dtype=np.float32)

            # 采样到指定点数
            N = pts.shape[0]
            if N >= self.npoints:
                # 随机采样
                idx = np.random.choice(N, self.npoints, replace=False)
            else:
                # 重复采样
                idx = np.random.choice(N, self.npoints, replace=True)
            pts = pts[idx]
            normals = normals[idx]

            # 预处理并获取分组信息
            base, segments = self._prep(pts)

            # 生成两个增强视图（点云和法向量同时增强）
            v1, normal1 = self._augment_once_with_normals(base.copy(), normals.copy())
            v2, normal2 = self._augment_once_with_normals(base.copy(), normals.copy())

            # 转换为torch张量
            v1 = torch.from_numpy(v1).float()
            v2 = torch.from_numpy(v2).float()
            normal1 = torch.from_numpy(normal1).float()
            normal2 = torch.from_numpy(normal2).float()
            y = torch.tensor(-1, dtype=torch.long)  # Fusion360没有标签

            # 返回分组信息
            if self.use_graphcut_segmentation and segments is not None:
                segments_tensor = torch.from_numpy(segments).long()
                return v1, v2, normal1, normal2, y, segments_tensor

            return v1, v2, normal1, normal2, y

        except Exception as e:
            print(f"加载文件失败 {obj_path}: {e}")
            # 返回零张量作为错误处理
            pts = torch.zeros(self.npoints, 3, dtype=torch.float32)
            normals = torch.zeros(self.npoints, 3, dtype=torch.float32)
            y = torch.tensor(-1, dtype=torch.long)
            if self.use_graphcut_segmentation:
                segments_tensor = torch.zeros(self.npoints, dtype=torch.long)
                return pts, pts, normals, normals, y, segments_tensor
            return pts, pts, normals, normals, y

    def __getitem__(self, idx: int):
        """
        返回一个整体文件夹内的batch_size个零件的数据（堆叠为批次张量）

        Returns:
            batch_v1: (batch_size, npoints, 3) - 第一个视图的点云坐标
            batch_v2: (batch_size, npoints, 3) - 第二个视图的点云坐标
            batch_normal1: (batch_size, npoints, 3) - 第一个视图的法向量
            batch_normal2: (batch_size, npoints, 3) - 第二个视图的法向量
            batch_y: (batch_size,) - 标签
            [batch_segments]: (batch_size, npoints) - 分组标签（如果启用分组）
        """
        group_dir = self.group_dirs[idx]
        obj_files = self.group_obj_files[group_dir]

        # 处理obj文件数量到batch_size
        num_files = len(obj_files)
        if num_files >= self.batch_size:
            # 随机采样batch_size个文件
            selected_files = np.random.choice(obj_files, self.batch_size, replace=False)
        else:
            # 重复采样补充到batch_size个文件
            selected_files = []
            selected_files.extend(obj_files)
            # 重复采样补充剩余的
            while len(selected_files) < self.batch_size:
                needed = self.batch_size - len(selected_files)
                additional = np.random.choice(obj_files, min(needed, num_files), replace=False)
                selected_files.extend(additional.tolist())

        # 加载所有选中的文件
        batch_data = []
        for obj_file in selected_files:
            data = self._load_single_obj(obj_file)
            batch_data.append(data)

        # 将数据堆叠成批量的张量格式，便于网络训练并行处理
        batch_v1 = torch.stack([item[0] for item in batch_data])  # (B, npoints, 3)
        batch_v2 = torch.stack([item[1] for item in batch_data])  # (B, npoints, 3)
        batch_normal1 = torch.stack([item[2] for item in batch_data])  # (B, npoints, 3)
        batch_normal2 = torch.stack([item[3] for item in batch_data])  # (B, npoints, 3)
        batch_y = torch.stack([item[4] for item in batch_data])  # (B,)

        # 如果启用了图割分割，还需要堆叠segments
        if self.use_graphcut_segmentation:
            batch_segments = torch.stack([item[5] for item in batch_data])  # (B, npoints)
            return batch_v1, batch_v2, batch_normal1, batch_normal2, batch_y, batch_segments

        return batch_v1, batch_v2, batch_normal1, batch_normal2, batch_y

    def _prep(self, pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        预处理点云，返回处理后的点云和对应的分组标签
        """
        if self.center_unit:
            pts = _center_and_scale_unit_sphere(pts)

        N = pts.shape[0]
        segments = None

        # 如果需要分组，先对原始点云进行分组
        if self.use_graphcut_segmentation:
            segments = segment_pointcloud(pts, k=self.graphcut_k,
                                       c=self.graphcut_c,
                                       min_size=self.graphcut_min_size)

        # 处理点数
        if N > self.npoints:
            idx = np.random.choice(N, self.npoints, replace=False)
            pts = pts[idx, :]
            if segments is not None:
                segments = segments[idx]
        elif N < self.npoints:
            pad_idx = np.random.choice(N, self.npoints - N, replace=True)
            pts = np.concatenate([pts, pts[pad_idx]], axis=0)
            if segments is not None:
                segments = np.concatenate([segments, segments[pad_idx]], axis=0)

        return pts, segments

    def _augment_once(self, pts: np.ndarray) -> np.ndarray:
        """对点云应用一次数据增强"""
        # 旋转增强
        if self.rotation_mode == "xyz":
            pts = random_rotate_xyz(pts)
        elif self.rotation_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")

        # 添加噪声
        if self.jitter:
            pts = jitter_points(pts, sigma=0.01, clip=0.02)

        return pts

    def _augment_once_with_normals(self, pts: np.ndarray, normals: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """对点云和法向量同时应用一次数据增强"""
        # 旋转增强
        if self.rotation_mode == "xyz":
            pts = random_rotate_xyz(pts)
            # 对法向量应用相同的旋转
            angles = np.random.uniform(0, 2*np.pi, 3)
            R = _rotation_matrix_xyz(*angles)
            normals = normals @ R.T
        elif self.rotation_mode == "none":
            pass
        else:
            raise ValueError(f"Unknown rotation_mode: {self.rotation_mode}")

        # 添加噪声（只对点云添加噪声，法向量保持不变）
        if self.jitter:
            pts = jitter_points(pts, sigma=0.01, clip=0.02)

        return pts, normals


# 测试代码
if __name__ == "__main__":
    dataset_path = "/data/cjj/dataset/Fusion360GalleryDataset/a1.0.0_10"

    # 测试原始Fusion360数据集
    print("=== 测试原始Fusion360数据集 ===")
    dataset = Fusion360Dataset(
        root=dataset_path,
        npoints=1024,
        augment=True,
        center_unit=True
    )

    print(f"数据集大小: {len(dataset)}")

    # 测试加载第一个样本
    if len(dataset) > 0:
        pts, normals, label = dataset[0]
        print(f"点云形状: {pts.shape}")
        print(f"法向量形状: {normals.shape}")
        print(f"标签: {label}")

    # 测试对比学习数据集
    print("\n=== 测试对比学习Fusion360数据集 ===")
    contrast_dataset = ContrastiveFusion360Dataset(
        root=dataset_path,
        npoints=1024,
        rotation_mode="xyz",
        jitter=True,
        center_unit=True,
        use_graphcut_segmentation=False
    )

    print(f"对比学习数据集大小: {len(contrast_dataset)}")

    # 测试加载第一个对比学习样本
    if len(contrast_dataset) > 0:
        result = contrast_dataset[0]
        v1, v2, normal1, normal2, y = result[:5]
        print(f"视图1点云形状: {v1.shape}")
        print(f"视图2点云形状: {v2.shape}")
        print(f"视图1法向量形状: {normal1.shape}")
        print(f"视图2法向量形状: {normal2.shape}")
        print(f"标签: {y}")
        if len(result) > 5:
            segments = result[5]
            print(f"分组标签形状: {segments.shape}")


