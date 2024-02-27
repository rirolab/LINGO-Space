"""
Dataset for demonstration
"""
import os
import pickle
import warnings

import numpy as np
from torch.utils.data import Dataset

from dataset_generation.utils.global_vars import BOUNDS, PIXEL_SIZE
import dataset_generation.utils.general_utils as utils



MAX_BOXES = 10


class DemoDataset(Dataset):

    def __init__(self, path, task_list=[], split='train', n_demos=0):
        self._path = path
        self.task_list = task_list
        self.split = split
        self.seeds_per_task = {}
        self.load_annos(n_demos, split)

    def load_annos(self, n_demos, split):
        filenames = []
        seeds = []
        task_names = []
        for task in self.task_list:
            _path = os.path.join(self._path, task + '-' + split, 'action')
            _fnames = sorted(os.listdir(_path))
            _keep = min(n_demos, len(_fnames))
            print(f'Found {len(_fnames)} demos for {task}, keeping {_keep}')
            _fnames = _fnames[:n_demos]
            filenames += _fnames
            seeds += [int(name[(name.find('-') + 1):-4]) for name in _fnames]
            task_names += [task for _ in range(len(_fnames))]

        self.cache = {}
        self.seeds_per_task = {task: [] for task in self.task_list}
        self.seeds_per_task['all'] = []
        _annos = []
        print(f'Loading {split} annotations...')
        for fname, seed, task in zip(filenames, seeds, task_names):
            _path = os.path.join(self._path, task + '-' + split)
            self.cache[task + '/' + fname] = {'seed': seed, 'task': task}
            self.seeds_per_task[task].append((task, seed))
            self.seeds_per_task['all'].append((task, seed))
            with open(os.path.join(_path, 'reward', fname), 'rb') as fid:
                rewards = pickle.load(fid)  # len(actions)
            if not (np.array(rewards)[1:] > 0).all():
                warnings.warn(f'WARNING: imperfect demo {fname} for {task}')
            pairs = []
            for k in range(len(rewards) - 1):
                if rewards[k + 1] == 0:  # unsuccessful action
                    continue
                pairs.append(k)
            _annos += [(fname, task, p) for p in range(len(pairs))]
            self.cache[task + '/' + fname]['obs_act'] = None
        self.annos = _annos

    @staticmethod
    def _action2point(action):
        p0_xyz, p0_xyzw = action['pose0']
        p1_xyz, p1_xyzw = action['pose1']
        p0 = utils.xyz_to_pix(p0_xyz, BOUNDS, PIXEL_SIZE)
        p0_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p0_xyzw)[2])
        p1 = utils.xyz_to_pix(p1_xyz, BOUNDS, PIXEL_SIZE)
        p1_theta = -np.float32(utils.quatXYZW_to_eulerXYZ(p1_xyzw)[2])
        p1_theta = p1_theta - p0_theta
        p0_theta = 0
        return p0, p1, p0_theta, p1_theta


    def add_to_cache(self, fname, task):
        _path = os.path.join(self._path, task + '-' + self.split)
        with open(os.path.join(_path, 'reward', fname), 'rb') as fid:
            rewards = pickle.load(fid)  # len(actions)
        with open(os.path.join(_path, 'color', fname), 'rb') as fid:
            color = pickle.load(fid)  # len(actions), 3, 480, 640, 3
        with open(os.path.join(_path, 'depth', fname), 'rb') as fid:
            depth = pickle.load(fid)  # len(actions), 3, 480, 640
        with open(os.path.join(_path, 'action', fname), 'rb') as fid:
            action = pickle.load(fid)  # len(actions) list of dicts
        with open(os.path.join(_path, 'info', fname), 'rb') as fid:
            info = pickle.load(fid)  # len(actions) list of dicts
        pairs = []
        for k in range(len(rewards) - 1):
            if rewards[k + 1] == 0:  # unsuccessful action
                continue
            p0, p1, p0_theta, p1_theta = self._action2point(action[k])

            pairs.append({
                'lang_goal': info[k]['lang_goal'],
                'color': color[k],
                'depth': depth[k],
                'action': action[k],
                'image': utils.get_image(
                    {'color': color[k], 'depth': depth[k]}
                ),
                'goal_image': utils.get_image(
                    {'color': color[k + 1], 'depth': depth[k + 1]}
                ),
                'p0': (
                    np.round(p0)
                    ),
                'p1': (
                    np.round(p1)
                ),
                'p0_theta': p0_theta,
                'p1_theta': p1_theta
            })
        self.cache[task + '/' + fname]['obs_act'] = pairs

    def get_seed(self, idx):
        name, task, _ = self.annos[idx]
        return self.cache[task + '/' + name]['seed']

    def get_seed_by_task_and_idx(self, task, idx):
        return self.seeds_per_task[task][idx]

    def retrieve_by_task_and_name(self, task, name, obs_act_id=None, theta_sigma=False):
        if self.cache[task + '/' + name]['obs_act'] is None:
            self.add_to_cache(name, task)
        if obs_act_id is not None:
            anno = self.cache[task + '/' + name]['obs_act'][obs_act_id]
            img, p0, p1 = anno['image'], anno['p0'], anno['p1']

        else:  # fetch all steps
            anno, img, p0, p1 = [], [], [], []
            for anno_ in self.cache[task + '/' + name]['obs_act']:
                anno.append(anno_)
                img_, p0_, p1_ = anno_['image'], anno_['p0'], anno_['p1']
                img.append(img_)
                p0.append(p0_)
                p1.append(p1_)
            img = np.stack(img)
            p0 = np.stack(p0)
            p1 = np.stack(p1)
        return anno, img, p0, p1
    
    def _fetch_idx_by_task(self, task):
        _annos = [
            a
            for a, anno in enumerate(self.annos)
            if anno[1] == task
        ]
        idx = np.random.randint(0, len(_annos))
        return _annos[idx]
    def __len__(self):
        return len(self.annos)