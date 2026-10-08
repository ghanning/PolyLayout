import json
import logging
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import tqdm
from plyfile import PlyData

from .ase import flatten_image_tuples
from .base_dataset import BaseDataset, collate
from .edge_rendering import EdgeRenderer
from .layout_sampling import R_from_cams, sample_layout
from .line_segments import read_line_segments
from .view import numpy_image_to_torch, read_view
from ..geometry import Camera, Cuboid, Polygon, Pose
from ...settings import DATA_PATH

logger = logging.getLogger(__name__)


def nerfstudio_to_colmap(c2w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''Convert a camera-to-world matrix from Nerfstudio format to
    COLMAP world-to-camera (R, t).

    See https://github.com/nerfstudio-project/nerfstudio/blob/da57d3f5ba5362391b961cb4ce8b5eda4e97268f/nerfstudio/data/dataparsers/colmap_dataparser.py#L155-L160
    '''
    c2w = c2w.copy()
    c2w[2] *= -1
    c2w = c2w[[1, 0, 2, 3]]
    c2w[0:3, 1:3] *= -1
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    return R, t


class ScanNet(BaseDataset):
    default_conf = {
        'dataset_dir': 'scannetpp/',
        'image_subpath': 'data/{}/dslr/undistorted_images/',
        'transform_subpath': 'data/{}/dslr/nerfstudio/transforms_undistorted.json',
        'pointcloud_subpath': 'data/{}/scans/mesh_aligned_0.05.ply',
        'info_dir': 'scannetpp_polylayout_training/',
        'read_info_files': False,
        'pose_dir': None,
        'pose_file': '{:04d}.json',

        'train_num_per_scene': None,
        'val_num_per_scene': None,
        'test_num_per_scene': None,
        'multi_room_num_per_scene': None,
        'multi_room_v2_num_per_scene': None,

        'num_views': 5,
        'init_layout': None,
        'init_layout_max_rot': float(np.deg2rad(15.0)),
        'init_layout_cam_margin': 0.5,
        'init_cuboid_max_trans': 0.5,
        'init_cuboid_max_grow': [-0.5, 0.5],
        'init_polygon_max_shift': [-0.5, 0.5],

        'grayscale': False,
        'resize': None,
        'resize_by': 'max',
        'crop': None,
        'pad': None,
        'optimal_crop': False,
        'seed': 0,

        'max_num_points3D': 500,
        'force_num_points3D': False,

        'flatten': False,

        'read_line_segments': False,
        'max_num_line_segments': 100,

        'render_edge_image': False,
        'edge_image_line_width': 3,

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
        self.conf, self.split = conf, split

        data_dir = Path(__file__).parent / 'scannetpp'

        with open(data_dir / f'scenes_{split}.txt') as f:
            self.scenes = [line.strip() for line in f]

        with open(data_dir / f'layouts_{split}.json') as f:
            self.layouts = json.load(f)

        with open(data_dir / f'images_{split}.json') as f:
            self.image_tuples = json.load(f)

        if conf.flatten:
            assert 'multi_room' in split
            self.image_tuples = flatten_image_tuples(self.image_tuples)

        scenes_with_files = set(s.split(':')[0] for s in self.scenes)

        self.read_transform_files(scenes_with_files)

        if self.conf.read_info_files:
            self.read_info_files(scenes_with_files)

        if self.conf[self.split + '_num_per_scene']:
            self.sample_new_items(conf.seed)
        else:
            self.items = self.image_tuples

        self.renderer_cache = {}

    def read_transform_files(self, scenes):
        self.transforms = {}
        self.frames = {}
        for scene in scenes:
            path = self.root / self.conf.transform_subpath.format(scene)
            with open(path) as f:
                transforms = json.load(f)
            self.transforms[scene] = transforms

            self.frames[scene] = {}
            for f in transforms['frames'] + transforms['test_frames']:
                self.frames[scene][f['file_path']] = f

    def read_info_files(self, scenes):
        logger.info(f'Reading info files')
        self.images, self.points3D, self.p3D_idx = {}, {}, {}
        self.name2idx = {}
        for scene in tqdm.tqdm(scenes):
            path = Path(DATA_PATH, self.conf.info_dir, scene + '.pkl')
            with open(path, 'rb') as f:
                info = pickle.load(f)
            self.images[scene] = info['image_names']
            self.points3D[scene] = info['points3D']
            self.p3D_idx[scene] = info['p3D_idx']
            self.name2idx[scene] = {name: idx for idx, name in enumerate(self.images[scene])}

    def sample_new_items(self, seed):
        logger.info(f'Sampling new images or pairs with seed {seed}')

        image_tuples_per_scene = {}
        for image_tuple in self.image_tuples:
            scene = image_tuple['scene']
            if scene not in image_tuples_per_scene:
                image_tuples_per_scene[scene] = []
            image_tuples_per_scene[scene].append(image_tuple)

        self.items = []
        num_per_scene = self.conf[self.split + '_num_per_scene']
        for scene in tqdm.tqdm(self.scenes):
            image_tuples = np.random.RandomState(seed).choice(
                image_tuples_per_scene[scene], num_per_scene, replace=False)
            self.items.extend(image_tuples)

        np.random.RandomState(seed).shuffle(self.items)

    def _get_renderer(self, width: int, height: int) -> EdgeRenderer:
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        if worker_id not in self.renderer_cache:
            self.renderer_cache[worker_id] = EdgeRenderer(width, height)
        return self.renderer_cache[worker_id]

    def _read_view(self, scene, image_name, layout_gt, seed, T=None):
        image_dir = self.root / self.conf.image_subpath.format(scene)
        image_path = image_dir / image_name

        transforms = self.transforms[scene]
        width, height = transforms['w'], transforms['h']
        params = [transforms['fl_x'], transforms['fl_y'], transforms['cx'], transforms['cy']]
        camera = Camera.from_colmap(
            dict(model='PINHOLE', width=width, height=height, params=params)
        )

        if T is None:
            frame = self.frames[scene][image_name]
            R, t = nerfstudio_to_colmap(np.array(frame['transform_matrix']))
            T = Pose.from_Rt(R, t)

        if self.conf.read_info_files:
            idx = self.name2idx[scene][image_name]
            p3D = self.points3D[scene]
            p3D_idx = self.p3D_idx[scene][idx]
        else:
            p3D, p3D_idx = np.empty((0, 3)), np.empty((0,), dtype=int)

        data = read_view(self.conf, image_path, camera, T, p3D, p3D_idx, random=(self.split == 'train'))
        assert tuple(data['camera'].size.numpy()) == data['image'].shape[1:][::-1]

        obs = p3D_idx
        if self.conf.crop:
            _, valid = data['camera'].world2image(data['T_w2cam']*p3D[obs])
            obs = obs[valid.numpy()]
        num_diff = self.conf.max_num_points3D - len(obs)
        if num_diff < 0:
            obs = np.random.choice(obs, self.conf.max_num_points3D, replace=False)
        num_valid_p3D = len(obs)
        if num_diff > 0 and self.conf.force_num_points3D:
            add = np.random.choice(np.delete(np.arange(len(p3D)), obs), num_diff)
            obs = np.r_[obs, add]
        data['points3D'] = torch.from_numpy(p3D[obs])
        data['p3D_mask'] = torch.zeros(data['points3D'].shape[0], dtype=bool)
        data['p3D_mask'][:num_valid_p3D] = True

        if self.conf.read_line_segments:
            lseg_path = image_dir / 'line_segments.npz'
            l2D, l2D_mask = read_line_segments(
                lseg_path, image_name, self.conf.max_num_line_segments, seed)
            data['lines2D'] = camera.normalize(l2D - 0.5).float()
            data['l2D_mask'] = torch.from_numpy(l2D_mask)

        if self.conf.render_edge_image:
            width, height = data['camera'].size.int().numpy()
            renderer = self._get_renderer(width, height)
            edge_image = renderer.render(layout_gt, data['camera'], T,
                                         self.conf.edge_image_line_width)
            edge_image = 255.0 - edge_image[:, :, 0].astype(np.float32)
            data['edge_image'] = numpy_image_to_torch(edge_image)

        return data

    def _read_room(self, scene: str, room: Optional[str], image_list: List[str], seed: int, room_idx: int | None = None):
        layout = self.layouts[scene]
        if room is not None:
            layout = layout[room]

        if 'R' in layout and 't' in layout and 's' in layout:  # Ground truth cuboid
            R, t, s = (np.array(layout[key]) for key in ('R', 't', 's'))
            layout_gt = Cuboid.from_Rts(R, t, s)
        elif 'R' in layout and 'd' in layout:  # Ground truth polygon
            R, d = (np.array(layout[key]) for key in ('R', 'd'))
            layout_gt = Polygon.from_Rd(R, d)
        else:  # Ground truth mesh
            layout_gt = None

        if self.conf.pose_dir is not None:
            pose_path = Path(self.conf.pose_dir) / self.conf.pose_file.format(room_idx)
            with open(pose_path) as f:
                poses = json.load(f)

        data = []
        for idx, name in enumerate(image_list[:self.conf.num_views]):
            if self.conf.pose_dir is not None:
                R, t = np.array(poses[idx]["R"]), np.array(poses[idx]["t"])
                T = Pose.from_Rt(R, t)
            else:
                T = None
            data.append(self._read_view(scene.split(':')[0], name, layout_gt, seed, T=T))
        data = collate(data)

        if 'multi_room' in self.split:  # Ground truth is either cuboid or mesh
            if layout_gt is not None:
                verts, faces = layout_gt.corners, layout_gt.faces
            else:
                verts, faces = (np.array(layout[key]) for key in ('verts', 'faces'))
            def pad_array(arr: np.ndarray, target_len: int) -> np.ndarray:
                num_pad = target_len - arr.shape[0]
                assert num_pad >= 0
                return np.pad(arr, ((0, num_pad), (0, 0)))
            # Pad to fixed size for batching
            verts = pad_array(verts, 60)
            faces = pad_array(faces, 120)
            data['verts_gt'] = torch.from_numpy(verts).float()
            data['faces_gt'] = torch.from_numpy(faces)
        else:
            data['layout_gt'] = layout_gt.float()

        data['scene'] = scene
        if room is not None:
            data['room'] = room
        return data

    def __getitem__(self, idx):
        image_tuple = self.items[idx]
        scene = image_tuple['scene']
        seed = self.conf.seed + idx

        if 'multi_room' in self.split and not self.conf.flatten:
            flat_idx = 0  # Flat index for the room
            for i in range(idx):
                flat_idx += len(self.image_tuples[i]['images'].keys())

            data = []
            for room, image_list in image_tuple['images'].items():
                data.append(self._read_room(scene, room, image_list, seed, room_idx=flat_idx))
                flat_idx += 1

            R = R_from_cams(torch.cat([d['T_w2cam'] for d in data]), seed)
            for d in data:
                d['layout_init'] = sample_layout(
                    self.conf, d['T_w2cam'], seed, R=R,
                    layout_gt=d.get('layout_gt')).float()

            data = collate(data)
        else:
            room = image_tuple.get('room', None)
            data = self._read_room(scene, room, image_tuple['images'], seed)
            data['layout_init'] = sample_layout(
                self.conf, data['T_w2cam'], seed, layout_gt=data.get('layout_gt')).float()
            if self.split == 'train':  # Train on polygons, even if ground truth is cuboid
                for key in ('layout_gt', 'layout_init'):
                    if key in data and isinstance(data[key], Cuboid):
                        data[key] = Polygon.from_cuboid(data[key]).float()

        if self.conf.read_pointcloud:
            points_path = self.root / self.conf.pointcloud_subpath.format(scene)
            mesh = PlyData.read(points_path, known_list_len={'face': {'vertex_indices': 3}})
            points = np.stack([mesh['vertex']['x'], mesh['vertex']['y'], mesh['vertex']['z']]).T
            data['pointcloud'] = torch.from_numpy(points).float()

        return data

    def __len__(self):
        return len(self.items)
