import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from projectaria_tools.projects import ase
from scipy.spatial.transform import Rotation

from .base_dataset import BaseDataset, collate
from .layout_sampling import R_from_cams, sample_layout
from .line_segments import read_line_segments
from .view import read_view
from ..geometry import Camera, Polygon, Pose
from ...settings import DATA_PATH


def flatten_image_tuples(image_tuples: List[Dict]) -> List[Dict]:
    flat_tuples = []
    for image_tuple in image_tuples:
        scene = image_tuple['scene']
        for room, images in image_tuple['images'].items():
            flat_tuples.append({'scene': scene, 'room': room, 'images': images})
    return flat_tuples


def mesh_to_polygon(verts: np.ndarray) -> Polygon:
    # Hacky way to convert a mesh to a Polygon
    # Assumes that the first half of the vertices are the floor
    z = np.unique(verts[:, 2])
    assert len(z) == 2, 'Mesh should have two distinct z-coordinates'

    R = np.eye(3)
    d = [z[0], z[1]]

    verts = verts[:verts.shape[0] // 2, :2]  # Take only the first half (floor)

    # Make sure we start with a x plane
    if not np.isclose(verts[0, 0], verts[1, 0]):
        verts = np.roll(verts, 1, axis=0)

    for i in range(verts.shape[0]):
        vi = verts[i]
        vj = verts[(i + 1) % verts.shape[0]]
        vij = vj - vi
        if i % 2 == 0:  # x plane
            assert np.isclose(vij[0], 0.0)
            d.append(vi[0])
        elif i % 2 == 1:  # y plane
            assert np.isclose(vij[1], 0.0)
            d.append(vi[1])
        else:
            raise ValueError('Mesh should be axis-aligned')

    d = np.array(d)
    return Polygon.from_Rd(R, d)


class ASE(BaseDataset):
    default_conf = {
        'dataset_dir': 'ase/',
        'pose_dir': None,
        'pose_file': '{:04d}.json',

        'num_views': 5,
        'init_layout': None,

        'rotate': True,
        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': False,
        'seed': 0,

        'flatten': False,

        'read_line_segments': False,
        'max_num_line_segments': 100,

        'read_pointcloud': False,
    }

    def _init(self, conf):
        pass

    def get_dataset(self, split):
        return _Dataset(self.conf, split)


class _Dataset(torch.utils.data.Dataset):
    def __init__(self, conf, split):
        if conf.init_layout is None:
            raise ValueError('The initial layout sampling strategy is required.')

        self.root = Path(DATA_PATH, conf.dataset_dir)
        self.conf = conf

        data_dir = Path(__file__).parent / 'ase'

        with open(data_dir / f'scenes_{split}.txt') as f:
            self.scenes = f.readlines()

        with open(data_dir / f'layouts_{split}.json') as f:
            self.layouts = json.load(f)

        with open(data_dir / f'images_{split}.json') as f:
            self.image_tuples = json.load(f)

        camera_path = self.root / 'camera_undistorted.json'
        with open(camera_path) as f:
            camera_data = json.load(f)
        if conf.rotate:
            fx, fy, cx, cy = camera_data['params']
            h = camera_data['height']
            camera_data['params'] = [fy, fx, h - cy, cx]
        self.camera = Camera.from_colmap(camera_data)

        device = ase.get_ase_rgb_calibration()
        self.T_d2c = Pose.from_4x4mat(device.get_transform_device_camera().inverse().to_matrix())
        if conf.rotate:
            R = Rotation.from_euler('z', 90, degrees=True).as_matrix()
            T_delta = Pose.from_Rt(R, np.zeros(3))
            self.T_d2c = T_delta @ self.T_d2c

        if conf.flatten:
            self.image_tuples = flatten_image_tuples(self.image_tuples)

    def _read_view(self, scene_dir: Path, image_name: str, trajectory: np.ndarray, seed: int, T_w2c=None):
        if T_w2c is None:
            traj_idx = int(image_name[8:15])  # "vignette0000043.jpg"
            T_w2d = Pose.from_4x4mat(trajectory[traj_idx].inverse().to_matrix())
            T_w2c = self.T_d2c @ T_w2d

        image_dir = scene_dir / 'rgb_undistorted'
        image_path = image_dir / image_name

        p3D = np.empty((0, 3))
        p3D_idxs = np.empty(0)
        rotation = -1 if self.conf.rotate else 0
        data = read_view(self.conf, image_path, self.camera, T_w2c, p3D, p3D_idxs, rotation=rotation)

        if self.conf.read_line_segments:
            lseg_path = image_dir / 'line_segments.npz'
            l2D, l2D_mask = read_line_segments(
                lseg_path, image_name, self.conf.max_num_line_segments, seed)
            if self.conf.rotate:
                l2D = np.stack([self.camera.size[1] - l2D[..., 1], l2D[..., 0]], axis=-1)
            data['lines2D'] = self.camera.normalize(l2D - 0.5).float()
            data['l2D_mask'] = torch.from_numpy(l2D_mask)

        return data

    def _read_room(self, scene: str, room: str, image_list: List[str], trajectory: np.ndarray, seed: int, room_idx: int | None = None):
        scene_dir = self.root / scene

        if self.conf.pose_dir is not None:
            pose_path = Path(self.conf.pose_dir) / self.conf.pose_file.format(room_idx)
            with open(pose_path) as f:
                poses = json.load(f)

        data = []
        for idx, image in enumerate(image_list[:self.conf.num_views]):
            if self.conf.pose_dir is not None:
                R, t = np.array(poses[idx]["R"]), np.array(poses[idx]["t"])
                if self.conf.rotate:
                    Rz = Rotation.from_euler("z", 90, degrees=True).as_matrix()
                    R, t = Rz @ R, Rz @ t
                T_w2c = Pose.from_Rt(R, t)
            else:
                T_w2c = None
            data.append(self._read_view(scene_dir, image, trajectory, seed, T_w2c=T_w2c))
        data = collate(data)

        layout_gt = self.layouts[scene][room]
        poly_gt = mesh_to_polygon(np.array(layout_gt['verts']))
        data['layout_gt'] = poly_gt.float()

        data['scene'] = scene
        data['room'] = room
        return data

    def __getitem__(self, idx):
        image_tuple = self.image_tuples[idx]
        scene = image_tuple['scene']
        seed = self.conf.seed + idx

        trajectory_path = self.root / scene / 'trajectory.csv'
        trajectory = ase.readers.read_trajectory_file(trajectory_path)['Ts_world_from_device']

        if self.conf.flatten:
            room = image_tuple['room']
            data = self._read_room(scene, room, image_tuple['images'], trajectory, seed, room_idx=idx)
            data['layout_init'] = sample_layout(self.conf, data['T_w2cam'], seed).float()
        else:
            flat_idx = 0  # Flat index for the room
            for i in range(idx):
                flat_idx += len(self.image_tuples[i]['images'].keys())

            data = []
            for room, image_list in image_tuple['images'].items():
                data.append(self._read_room(scene, room, image_list, trajectory, seed, room_idx=flat_idx))
                flat_idx += 1

            R = R_from_cams(torch.cat([d['T_w2cam'] for d in data]), seed)
            for d in data:
                d['layout_init'] = sample_layout(self.conf, d['T_w2cam'], seed, R=R).float()

            data = collate(data)

        if self.conf.read_pointcloud:
            points_path = self.root / scene / 'semidense_points.csv.gz'
            points = ase.readers.read_points_file(str(points_path))
            data['pointcloud'] = torch.from_numpy(points).float()

        return data

    def __len__(self):
        return len(self.image_tuples)
