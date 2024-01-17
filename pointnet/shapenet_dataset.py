from torch_geometric.data import InMemoryDataset
# from torch_geometric.io import read_ply
from torch_geometric.data import Data
import torch
import os
import os.path as osp
import glob
import urllib

import open3d as o3d
import numpy as np


def read_pcd(path):
    pcd = o3d.io.read_point_cloud(path)
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=10.0, max_nn=30))
    pos = torch.from_numpy(np.array(pcd.points)).to(torch.float)
    normal = torch.from_numpy(np.array(pcd.normals)).to(torch.float)
    color = torch.from_numpy(np.array(pcd.colors)).to(torch.float)
    return Data(pos=pos, normal=normal, x=color)


class ShapenetDataset(InMemoryDataset):
    url = "https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip"

    def __init__(self, root, train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        (path, _) = urllib.urlretrieve(self.url, self.root / "raw.zip")

    # def process(self):
    #
    #     torch.save(self.process_set('train'), self.processed_paths[0])
    #     torch.save(self.process_set('test'), self.processed_paths[1])

#    def process_set(self, dataset):
#        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
#        categories = sorted([x.split(os.sep)[-2] for x in categories])

#        data_list = []
#        for target, category in enumerate(categories):
#            folder = osp.join(self.raw_dir, category, dataset)
#            paths = glob.glob(f'{folder}/{category}_*.pcd')
#            for path in paths:
#                data = read_pcd(path)
#                data.y = torch.tensor([target])
#                data_list.append(data)
#
#        if self.pre_filter is not None:
#            data_list = [d for d in data_list if self.pre_filter(d)]
#
#        if self.pre_transform is not None:
#            data_list = [self.pre_transform(d) for d in data_list]

    #     return self.collate(data_list)
