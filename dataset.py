import os, pickle, tqdm, copy
from PIL import Image
import numpy as np
import torch
import clip
from torch_geometric.data import InMemoryDataset, Data
from transformers import AutoModel, AutoImageProcessor, AutoTokenizer
import omegaconf


MAX_LENGTH = 77 # same as CLIP tokenizer

class LINGOSpaceDataset(InMemoryDataset):
    """
    Dataset class for LINGO-Space dataset

    Args:
        root (str): root directory of the dataset
        type (str): type of the dataset (e.g., compositional)
        split (str): split of the dataset
        n (int): the number of data to load (0 for all data)
        device (str): device to use
        encoder (str): encoder to use
        max_x (int): the maximum width of the image
        max_y (int): the maximum height of the image
        transform (callable): transform to apply to the data
        pre_transform (callable): transform to apply to the data before saving
        pre_filter (callable): filter to apply to the data before saving
    """
    def __init__(
        self,
        root='data',
        type='composite',
        split='train',
        n=0,
        device='cuda',
        encoder='ViT-L/14',
        transform=None, pre_transform=None, pre_filter=None, **kwargs):
        self.root = root
        self.type = type
        self.split = split
        self.n = n
        self.device = device

        # Load encoder
        self.encoder_name = encoder
        encoder_model, self.image_processor = clip.load(self.encoder_name, device=device)
        encoder_model.requires_grad_(False)
        encoder_model.eval()
        self.encode_image = lambda x: encoder_model.encode_image(
            x.to(self.device)).float().cpu()
        self.encode_text = lambda x: encoder_model.encode_text(
            clip.tokenize(x).to(self.device)).float().cpu()
        
        if split == 'test': return
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)
        
    @property
    def raw_dir(self):
        return os.path.join(self.root)
    
    @property
    def raw_file_names(self):
        fnames = []
        counts = {'train': 0, 'val': 0}
        for split in ['train', 'val']:
            subdir = os.path.join(self.raw_dir, f'{self.type}-{split}')
            for fname in sorted(os.listdir(os.path.join(subdir, 'info'))):
                fnames.append(os.path.join(subdir, 'info', fname))
                counts[split] += 1
                if self.n > 0 and split == 'train' and counts[split] >= self.n:
                    break
        print(counts)
        return fnames

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed', self.type)
    
    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt']
    
    def normalize(self, x, y, max_x, max_y):
        """ Normalize x, y coordinates to [-1, 1] """
        if x < 0: x = 0
        if y < 0: y = 0
        if x > max_x: x = max_x
        if y > max_y: y = max_y

        x = x / max_x * 2 - 1
        y = y / max_y * 2 - 1

        # Keep the aspect ratio
        if max_x > max_y:
            ratio = max_x / max_y
            y = y / ratio
        return x, y

    def get_bboxes(self, info):
        """ Get the ground-truth bboxes and attach to info """
        id_to_bbox = {}
        for id, (bbox, quat, name) in info['fixed'].items():
            (x1, y1), (x2, y2) = bbox
            id_to_bbox[id] = (x1, y1, x2, y2)
        for id, (bbox, quat, name) in info['rigid'].items():
            (x1, y1), (x2, y2) = bbox
            id_to_bbox[id] = (x1, y1, x2, y2)
        for id, (bbox, quat, name) in info['deformable'].items():
            (x1, y1), (x2, y2) = bbox
            id_to_bbox[id] = (x1, y1, x2, y2)
        info['bbox'] = id_to_bbox

    def get_gt_parsing(self, info):
        """ Get the ground-truth parsing and attach to info """
        assert 'rel_ids' in info and 'ref_ids' in info

        # Get source, i.e., the object to pick
        assert (np.array(info['rel_ids']) == info['rel_ids'][0]).all(), \
            f'{info["rel_ids"]} != {info["rel_ids"][0]}'
        
        rel_id = info['rel_ids'][0]
        source = info['names'][rel_id]

        # Get target, i.e., the space to place (ref obj, relation)
        target = []
        assert len(info['ref_ids']) == len(info['relations']), \
            f'len({info["ref_ids"]}) != len({info["relations"]})'
        
        for ref_id, relation in zip(info['ref_ids'], info['relations']):
            ref_name = info['names'][ref_id]
            target.append((ref_name, relation))
        
        # Attach parsing to info
        info['parsing'] = {
            'action': 'move',
            'source': source,
            'target': target,
        }

    def process_one_info(self, info, image=None):        
        has_dino = True
        # Get bboxes (from dino info)
        if self.split == 'train':
            assert 'bbox' not in info
        if 'bbox' not in info:
            has_dino = False
            self.get_bboxes(info)

        # Get id_to_idx
        id_to_idx = {}
        for idx, id in enumerate(info['bbox'].keys()):
            id_to_idx[id] = idx
        
        # Create a dummy image
        if image is None:
            raise ValueError('Image is None')
            image = Image.new('RGB', (50, 50), color='white')

        # Get cropped image / Normalize bbox (idx_to_bbox) / Get idx_to_pos
        idx_to_cropped_img = {}
        idx_to_pos = {}
        idx_to_bbox = {}
        for id, bbox in info['bbox'].items():
            idx = id_to_idx[id]
            x1, y1, x2, y2 = bbox
            
            # Crop image
            cropped_image = image.crop((x1, y1, x2, y2))
            # Preprocess image
            cropped_image = self.image_processor(cropped_image)
            idx_to_cropped_img[idx] = cropped_image

            # Normalize the bbox and update idx_to_bbox
            x1, y1 = self.normalize(x1, y1, *image.size)
            x2, y2 = self.normalize(x2, y2, *image.size)
            idx_to_bbox[idx] = ((x1, y1, x2, y2))
            idx_to_pos[idx] = (((x1 + x2) / 2, (y1 + y2) / 2))
            
        # Sort by key (idx) and convert to list
        cropped_images = []
        pos = []
        bboxes = []
        _idx = 0
        for idx in sorted(idx_to_cropped_img.keys()):
            assert idx == _idx
            _idx += 1
            cropped_images.append(idx_to_cropped_img[idx])
            pos.append(idx_to_pos[idx])
            bboxes.append(idx_to_bbox[idx])
        
        # Stack cropped images
        cropped_images = torch.stack(cropped_images, dim=0) # [num_nodes, 3, 224, 224]
        assert cropped_images.size(0) == len(pos) == len(bboxes), \
            f'{cropped_images.size(0)} != len({pos}) != len({bboxes})'
        
        # Get viz feature
        viz_features = self.encode_image(cropped_images)


        # Get edges (from dino info)
        if self.split == 'train':
            assert 'edges' not in info
        if 'edges' not in info:
            assert has_dino == False
            info['edges'] = info['graph']
            
        # Create a graph by connecting edges
        edge_index = []
        edge_predicate = []
        for subj_id, predicate, obj_id in info['edges']:
            subj_idx = id_to_idx[subj_id]
            obj_idx = id_to_idx[obj_id]
            edge_index.append([subj_idx, obj_idx])
            edge_predicate.append(predicate)
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        if edge_index.numel() == 0: # no edges
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((edge_index.size(1), 768)) # 768: CLIP's output size
        else:
            edge_attr = self.encode_text(edge_predicate)
        assert edge_index.size(1) == edge_attr.size(0), \
            f'{edge_index.size(1)} != {edge_attr.size(0)}'
        

        # Get parsing (from eval info)
        if self.split == 'train':
            assert 'parsing' not in info
        if 'parsing' not in info:
            self.get_gt_parsing(info)
        
        # Encode each predicate and reference object name
        predicates = []
        ref_obj_names = []
        pred_features = []
        ref_obj_features = []
        for ref_obj_name, predicate in info['parsing']['target']:
            predicates.append(predicate)
            ref_obj_names.append(ref_obj_name)

            # Encode the relation
            pred_feature = self.encode_text(predicate)
            pred_features.append(pred_feature)

            # Encode the reference object name
            ref_obj_feature = self.encode_text(ref_obj_name)
            ref_obj_features.append(ref_obj_feature)
        
        pred_features = torch.cat(pred_features, dim=0)
        ref_obj_features = torch.cat(ref_obj_features, dim=0)


        # Optional: Get objects' names
        idx_to_name = {}
        if 'names' in info:
            for id, name in info['names'].items():
                idx = id_to_idx[id]
                idx_to_name[idx] = name
            assert len(idx_to_name) == len(pos) == len(bboxes), \
                f'{len(idx_to_name)} != len({pos}) != len({bboxes})'
        names = []
        for idx in sorted(idx_to_name.keys()):
            names.append(idx_to_name[idx])
        

        # Data 
        x = torch.arange(len(pos), dtype=torch.long)
        data = Data(
            x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(pos),

            viz_features=viz_features,
            pred_features=pred_features,
            ref_features=ref_obj_features,
            bboxes=torch.tensor(bboxes),
            
            text=info['lang_goal'] if 'lang_goal' in info else None,
            parsing=info['parsing'],
            predicates=predicates,
            ref_obj_names=ref_obj_names,

            cropped_images=cropped_images,

            info_fname=info['info_fname'] if 'info_fname' in info else None,
            image_fname=info['image_fname'] if 'image_fname' in info else None,
            eval_fname=info['eval_fname'] if 'eval_fname' in info else None,
            dino_fname=info['dino_fname'] if 'dino_fname' in info else None,

            names=names,
            id_to_idx=list(id_to_idx.items()),
        )

        # Get ref_idxs
        if self.split != 'test' and has_dino == False:
            ref_idxs = []
            ref_obj_bboxes = []
            for ref_id in info['ref_ids']:
                ref_idxs.append(id_to_idx[ref_id])
                ref_obj_bboxes.append(info['bbox'][ref_id]) # ref_obj_bboxes is not normalized
            data.ref_idxs = torch.tensor(ref_idxs)
            data.ref_bboxes = torch.tensor(ref_obj_bboxes)
  
        return data
    

    def process_demo_info(self, dino_info, image=None, parsing=None, eval_fname=None, image_fname=None):
        assert image is not None and image_fname is None
        if parsing is not None:
            dino_info['parsing'] = parsing

        if eval_fname is not None:
            with open(eval_fname, 'rb') as f:
                eval_info = pickle.load(f)
            dino_info['parsing'] = eval_info[0]['parsing']
            dino_info['eval_fname'] = eval_fname
        
        assert 'parsing' in dino_info
        
        if image_fname is not None:
            image = Image.open(image_fname)
            dino_info['image_fname'] = image_fname

        data = self.process_one_info(dino_info, image)
        return data    
    
    def process_one_file(self, info_fname, image_dir, eval_fname, dino_fname):
        # Load info
        with open(info_fname, 'rb') as f:
            info = pickle.load(f)
        
        # Load eval info
        if os.path.exists(eval_fname) and self.split != 'train':
            with open(eval_fname, 'rb') as f:
                eval_info = pickle.load(f)
        else:
            eval_info = None
        
        # Load dino info
        if os.path.exists(dino_fname) and self.split != 'train':
            with open(dino_fname, 'rb') as f:
                dino_info = pickle.load(f)
        else:
            dino_info = None

        # Process each frame
        seq_len = len(info)
        data_list = []
        for s in range(seq_len-1):
            info[s]['info_fname'] = info_fname

            # Load image
            if os.path.exists(image_dir):
                image_fname = os.path.join(image_dir, f'{s}.png')
                info[s]['image_fname'] = image_fname
                image = Image.open(image_fname)
                assert image.size == (640, 320)
            else:
                raise ValueError(f'Image not found: {image_dir}')

            if eval_info is not None:
                info[s]['parsing'] = eval_info[s]['parsing']
                info[s]['eval_fname'] = eval_fname

            if dino_info is not None:
                info[s]['bbox'] = dino_info[s]['bbox']
                info[s]['edges'] = dino_info[s]['edges']
                info[s]['dino_fname'] = dino_fname
        
            data = self.process_one_info(info[s], image)

            # Get y
            if 'place_goal' in info[s] and info[s]['place_goal']:
                x1, y1, x2, y2 = list(info[s]['place_goal'].values())[0]
                x1, y1 = self.normalize(x1, y1, *image.size)
                x2, y2 = self.normalize(x2, y2, *image.size)
                y = torch.tensor([(x1 + x2) / 2, (y1 + y2) / 2])
            else:
                y = self.get_y(info[s], info[s+1], *image.size)
            data.y = y

            data_list.append(data)

        return data_list

    
    def get_y(self, before, after, W, H):
        possible_target_ids = list(before['move_goal'].keys())
        target_id = None
        for id, (bbox, quat, name) in after['fixed'].items():
            if target_id is None and id in possible_target_ids:
                before_bbox = before['bbox'][id] # before bbox
                before_bbox = (
                    *self.normalize(before_bbox[0], before_bbox[1], W, H),
                    *self.normalize(before_bbox[2], before_bbox[3], W, H),
                )

                (x1, y1), (x2, y2) = bbox # after bbox
                after_bbox = (
                    *self.normalize(x1, y1, W, H),
                    *self.normalize(x2, y2, W, H),
                    )
                # Check if the bbox is moved
                if abs(after_bbox[0] - before_bbox[0]) > 0.01 \
                    or abs(after_bbox[1] - before_bbox[1]) > 0.01:
                    target_id = id
                    target_bbox = after_bbox
                    break

        for id, (bbox, quat, name) in after['rigid'].items():
            if target_id is None and id in possible_target_ids:
                before_bbox = before['bbox'][id] # before bbox
                before_bbox = (
                    *self.normalize(before_bbox[0], before_bbox[1], W, H),
                    *self.normalize(before_bbox[2], before_bbox[3], W, H),
                )

                (x1, y1), (x2, y2) = bbox # after bbox
                after_bbox = (
                    *self.normalize(x1, y1, W, H),
                    *self.normalize(x2, y2, W, H),
                    )
                # Check if the bbox is moved
                if abs(after_bbox[0] - before_bbox[0]) > 0.01 \
                    or abs(after_bbox[1] - before_bbox[1]) > 0.01:
                    target_id = id
                    target_bbox = after_bbox
                    break

        for id, (bbox, quat, name) in after['deformable'].items():
            if target_id is None and id in possible_target_ids:
                before_bbox = before['bbox'][id] # before bbox
                before_bbox = (
                    *self.normalize(before_bbox[0], before_bbox[1], W, H),
                    *self.normalize(before_bbox[2], before_bbox[3], W, H),
                )

                (x1, y1), (x2, y2) = bbox # after bbox
                after_bbox = (
                    *self.normalize(x1, y1, W, H),
                    *self.normalize(x2, y2, W, H),
                    )
                # Check if the bbox is moved
                if abs(after_bbox[0] - before_bbox[0]) > 0.01 \
                    or abs(after_bbox[1] - before_bbox[1]) > 0.01:
                    target_id = id
                    target_bbox = after_bbox
                    break

        assert target_id is not None

        y = torch.tensor([
            (target_bbox[0] + target_bbox[2]) / 2,
            (target_bbox[1] + target_bbox[3]) / 2
            ])
        return y


    def process(self):
        data_list_by_split = {
            'train': [],
            'val': [],
        }
        for info_fname in tqdm.tqdm(self.raw_file_names):
            dirname = os.path.dirname(info_fname)
            basename = os.path.basename(info_fname)
            image_dir = os.path.join(
                dirname.replace('info', 'images'), basename.replace('.pkl', ''))
            eval_fname = info_fname.replace('info', 'eval_info')
            dino_fname = info_fname.replace('info', 'dino_info')
            data_list = self.process_one_file(
                info_fname, image_dir, eval_fname, dino_fname)
            
            if 'train' in info_fname:
                split = 'train'
            elif 'val' in info_fname:
                split = 'val'
            data_list_by_split[split].extend(data_list)
        
        # Save
        for split in ['train', 'val']:
            data_list = data_list_by_split[split]
            torch.save(
                self.collate(data_list),
                os.path.join(self.processed_dir, f'{split}.pt')
                )


if __name__ == '__main__':
    # Test
    LINGOSpaceDataset('data')