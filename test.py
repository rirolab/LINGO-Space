import argparse, os
from PIL import Image
import random
import pickle

import numpy as np
import torch
from torch_geometric.loader import DataLoader
from tqdm import tqdm
# import cv2

from dataset import LINGOSpaceDataset
from estimator.model import EstimatorModel
from estimator.utils import get_max_point
from semantic_parser import gpt_parser
from dataset_generation.environments.environment import Environment
from dataset_generation import tasks
import dataset_generation.utils.general_utils as utils
from dataset_generation.dataset import DemoDataset
from sgg.scene_graph import SceneGraphWithGDino


class LINGOSpaceInference:
    def __init__(self, ckpt_path, device='cuda'):
        # Model hyperparameters
        if 'composite' in ckpt_path:
            dim_in = 768
            dim_h = 256
            self.is_composite = True
        else:
            dim_in = 512
            dim_h = 128
            self.is_composite = False

        self.model = EstimatorModel(
            dim_in=dim_in,
            dim_h=dim_h,
            K=4,
            dim_edge=dim_in,
            num_layers=2,
            num_heads=2,
            act='leaky_relu',
        )
        self.device = device
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device)
        self.model.eval()

    @torch.no_grad()
    def run(self, dino_info, image=None, eval_fname=None, image_fname=None):
        if eval_fname is None or not os.path.exists(eval_fname):
            eval_fname = None
            parsing = gpt_parser.parse(dino_info['lang_goal'])
        else:
            parsing = None
            # check eval_fname and if the parsing is empty, just directly return here
            with open(eval_fname, 'rb') as f:
                eval_info = pickle.load(f)
            if eval_info[0]['parsing'] is None:
                return None
        dataset = LINGOSpaceDataset(split='test', device=self.device, encoder='ViT-L/14' if self.is_composite else 'ViT-B/32')

        data = dataset.process_demo_info(
            dino_info=dino_info,
            image=image,
            eval_fname=eval_fname,
            parsing=parsing,
            image_fname=image_fname
            )
        loader = DataLoader([data], batch_size=1, shuffle=False)
        for batch in loader:
            batch = batch.to(self.device)
            parameters = []
            M = len(batch.pred_features)
            for i in range(M):
                batch_clone = batch.clone()
                batch_clone.pred_features = batch.pred_features[i:i+1]
                batch_clone.ref_features = batch.ref_features[i:i+1]
                
                batch_clone = self.model(batch_clone)
                output = batch_clone.output
            
                parameters.append(output)
                batch.prev_state = batch_clone.prev_state
            
            max_point = get_max_point(parameters, batch.pos) # pixel coordinate
            break
        return max_point
    

class Tester:
    def __init__(self, model, data_loader, args):
        self.model = model
        self.data_loader = data_loader
        self.args = args

        self.env = _set_environment(args)
        self.dino = SceneGraphWithGDino()

    def run(self):
        for i, task_ in enumerate(self.data_loader.dataset.task_list):
            self.evaluate_task(mode='test', eval_task=task_)

    def evaluate_task(self, mode='test', eval_task=None):
        
        dset = self.data_loader.dataset
        success_rates = []
        rewards = []
        parsing_failure_counts = 0
        eval_list = range(len(dset.seeds_per_task[eval_task]))
        pbar = tqdm(eval_list)

        for i in pbar:
            # Set env with seed
            name, seed = dset.get_seed_by_task_and_idx(eval_task, i)
            if 'composite' in name:
                splited = eval_task.split("_")
                depth = int(splited[-1])
                task = tasks.names[splited[0]](depth=[depth, depth])
                task.language_augment = True
                task.overlap = True
            else:
                task = tasks.names[name]()
            eval_info_base_path = f"{self.args.data_root}/{eval_task}-test/eval_info"   

            # print(eval_info_base_path)         
            task.mode = mode
            task.name = name
            np.random.seed(seed)
            random.seed(seed)


            self.env.set_task(task)
            obs, _ = self.env.reset() # get rgb image and depth in 3 views
            info = self.env.info
            if self.args.record:
                self.env.start_rec(f'{i+1:06d}')
            for j in range(1):
                episode = str(i).zfill(6)
                
                eval_info_path = f"{eval_info_base_path}/{episode}-{seed}.pkl"

                img = utils.get_image(obs)
                color = img[..., :3]
                hmap = img[..., 3]

                color = color.transpose(1, 0, 2)
                color = np.array(color[..., :3]).astype(np.uint8) # color channel should be r, g, b order
                predicted_bboxes, _ = self.dino.detect_boxes(color)
                processed_img = Image.fromarray(color).convert('RGB')

                dino_info = {}
                dino_info['lang_goal'] = info['lang_goal']
                dino_info['bbox'] = predicted_bboxes
                dino_info['edges'] = self.dino.make_sg()

                dino_info['pick_goal'] = None
                dino_info['place_goal'] = None
                
                result = self.model.run(dino_info=dino_info, image=processed_img, eval_fname=eval_info_path)
                if result is None:
                    # parsing failed
                    parsing_failure_counts += 1
                    rewards.append(0)
                    success_rates.append(0)
                    continue
                
                if self.args.record:
                    color = utils.get_image(obs)[..., :3]
                    color = color.transpose(1, 0, 2)
                    caption = info['lang_goal']
                    self.env.add_video_frame_text(caption=caption)

                place_res = result
                place_pix = [int(place_res[0]), int(place_res[1])]
                
                
                gt_action = dset.retrieve_by_task_and_name(eval_task, f'{i:06d}-{seed}.pkl', obs_act_id=j)
                anno = gt_action[0]
                pick_pixel = gt_action[2]

                pick_pos = utils.pix_to_xyz(pick_pixel, hmap,
                                    self.env.task.bounds, self.env.task.pix_size)
                p0_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -anno['p0_theta']))
                
                place_pix = [min(place_pix[0], 639), min(place_pix[1], 319)]
                place_pos = utils.pix_to_xyz(place_pix, hmap,
                                    self.env.task.bounds, self.env.task.pix_size)
                p1_xyzw = utils.eulerXYZ_to_quatXYZW((0, 0, -anno['p1_theta']))
                pick_pose = (np.asarray(pick_pos), np.asarray(p0_xyzw))
                place_pose = (np.asarray(place_pos), np.asarray(p1_xyzw))
                act =  {'pose0': pick_pose, 'pose1': place_pose}
                
                obs, _, _, done, _, _ = self.env.step(act)     
                            
            
                _, _, obj_mask = task.get_true_image(self.env)
                cur_reward = self.env.task.reward(
                        done=done, obj_mask=obj_mask
                    )
                rewards.append(cur_reward)
                
                if cur_reward > 0.99:
                    success_rate = 1
                else:
                    success_rate = 0
                success_rates.append(success_rate)

                    # total_reward += cur_reward
                info = self.env.info
                done = self.env.task.done()   
                if done:
                    break

            pbar.set_postfix({"r": np.mean(rewards), "sc": np.mean(success_rates)})
            
            if self.args.record:
                self.env.end_rec()
        mean_reward = np.mean(rewards)
        mean_success_rate = np.mean(success_rates)
        print(f'Mean reward {eval_task}: {mean_reward*100:.1f}')
        print(f"success rate {eval_task}: {mean_success_rate*100:.1f}")
        print(f"parsing failure rate {eval_task}: {parsing_failure_counts/len(rewards)*100:.1f}")


def _set_environment(args):
    record_cfg = {
        'save_video': args.record,
        'save_video_path': "./videos",
        'add_text': False,
        'fps': 25,
        'video_height': 320,
        'video_width': 640,
        }
    env = Environment(
        'dataset_generation/environments/assets/',
        disp=False,
        shared_memory=False,
        hz=480,
        record_cfg=record_cfg
    )
    return env


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--dataset_root", default='data', dest='data_root', type=str)
    argparser.add_argument("--dataset_type", default='composite', dest='task', type=str)
    argparser.add_argument("--ckpt_root", default='checkpoints', type=str)
    argparser.add_argument("--ndemos_test", default=50, type=int)
    argparser.add_argument("--record", default=False, action='store_true')
    argparser.add_argument("--device", default='cuda', type=str)
    
    args = argparser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    task_name = args.task
    task = task_name.split("_")[-1]

    if task == 'composite':
        total_task = []
        for i in range(1, 7):
            total_task.append(f"{task}_{i}")
    else:
        total_task = [task]
            
    dataset = DemoDataset(
        args.data_root,
        task_list=total_task,
        n_demos=args.ndemos_test,
        split='test'
        )
    data_loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )

    
    ckpt_path = os.path.join(args.ckpt_root, args.task, 'best.pt')
    
    model = LINGOSpaceInference(ckpt_path, device)
    tester = Tester(model, data_loader, args)
    tester.run()


if __name__ == "__main__":
    main()
