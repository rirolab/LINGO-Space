import os
from copy import copy

import cv2
import random
import numpy as np
from dataset_generation.utils.global_vars import IN_SHAPE
from dataset_generation.tasks.task import Task
import dataset_generation.utils.general_utils as utils
import dataset_generation.tasks.spatial_relations as spatial_relations
import pybullet as p


class LingoSpaceComposite(Task):

    def __init__(self, depth=[2, 4]):
        super().__init__()
        self.total_objects_num = 7
        # if you want to use more objects, you can just simply increase depth
        # self.total_objects_num = max(self.total_objects_num, depth[1]+1)
        # to make more natural language
        self.replacelist = {"to the above of": "above", "to the below of": "below", "racoon": "raccoon", "to the far of": "far away from", "to the close of": "close to"}
        self.depth = depth
        self.language_augment = False
        self.save = True
        self.max_steps = 1
        self.overlap = False
        
        self.objects = {
            "ring": "hanoi/disk3_scaled.urdf",
            "cube": "box/box-template.urdf",
            "cylinder": "box/cylinder-template.urdf",
            "bowl": "bowl/bowl_scaled.urdf"
        }

        
        self.object_sizes = {
            "ring": (0.04, 0.04, 0.02),
            "cube": (0.04, 0.04, 0.02),
            "cylinder": (0.04, 0.04, 0.02),
            "bowl": (0.04, 0.04, 0.02),
        }
        
        self.object_scales = {
            "ring": (0.0008, 0.0008, 0.0008),
            "cube": (1.0, 1.0, 0.5),
            "cylinder": (0.04, 0.04, 0.02),
            "bowl": (0.8, 0.8, 0.02)
        }
        
        self.synonym_list = {
            'red': ['scarlet', 'vermilion', 'ruby'],
            'blue': ['azure', 'cobalt', 'sapphire'],
            'green': ['greenish', 'viridescent'],
            'yellow': ['lemon', 'amber', 'golden'],
            'brown': ['hazel', 'auburn', 'coopery'],
            'gray': ['grey', 'silvery', 'pearly'],
            'pink': ['rosy', 'pale red', 'salmon'],
            'white': ['colourless', 'unpigmented', 'undyed'],
            'ring': ['band', 'loop', 'hoop'],
            'cube': ['hexahedron', 'cuboid', 'block'],
            'bowl': ['container', 'vessel', 'receptacle']
        }
        
        self.relation_to_class = {
            "left": spatial_relations.LeftSeenColors(),
            "right": spatial_relations.RightSeenColors(),
            "above": spatial_relations.AboveSeenColors(),
            "below": spatial_relations.BelowSeenColors(),
            "close": spatial_relations.CloseSeenColors(),
            "far": spatial_relations.FarSeenColors(),
            "left above": spatial_relations.LeftAboveSeenColors(),
            "right above": spatial_relations.RightAboveSeenColors(),
            "left below": spatial_relations.LeftBelowSeenColors(),
            "right below": spatial_relations.RightBelowSeenColors()
        }
        for rel in self.relation_to_class.values():
            rel.buffer = int(rel.buffer * 0.4)

        self.conjugate = {
            "left": "right",
            "right": "left",
            "above": "below",
            "below": "above",
            "far": "far",
            "close": "close",
            "left above": "right below",
            "right below": "left above",
            "right above": "left below",
            "left below": "right above"
        }
    
    def get_colors(self):
        return utils.TRAIN_COLORS

    def reset(self, env):
        super().reset(env)
        self.labels = {}
        all_objects = [f"{color} {object_}" for color in self.get_colors() for object_ in list(self.objects.keys())]
        
        depth = random.choice(np.arange(self.depth[0], self.depth[1]+1))
        rel_object = random.choice(all_objects)
        all_objects.remove(rel_object)
        utterance = f"put the {rel_object} "
        video_utterance = f"put the {rel_object} "
        
        ref_objects = []
        ref_relations = []
        # generate relationships first
        for i in range(depth):
            ref_object = random.choice(all_objects)
            # if the setting does not allow overlapping object, remove it from object candidates
            if not self.overlap:
                all_objects.remove(ref_object)
            ref_objects.append(ref_object)
            
            ref_rel = random.choice(list(self.relation_to_class.keys()))
            # using more than one directional relationship does not narrow down space
            # Therefore, disable using overlapped relatioships except far and close
            while ref_rel in ref_relations and ref_rel not in ["far", "close"]:
                ref_rel = random.choice(list(self.relation_to_class.keys()))
            ref_relations.append(ref_rel)

            utterance += f'to the {ref_rel} of the {ref_object} and '
        
        utterance = utterance[:-5]

        # load a dummy object to random place, this will be the goal position        
        dummy_obj_ids, rel_gt_pose = self.load_objects(
            env, [rel_object],
            dummy=True
        )
        
        if dummy_obj_ids == None:
            env.failed_datagen = True
            return None

        if self.mode == 'test':
            # in the case of test, strictly disable getting partial point without execution
            rel_constraint_image = np.zeros((IN_SHAPE[0], IN_SHAPE[1]))
        else:
            rel_constraint_image = np.ones((IN_SHAPE[0], IN_SHAPE[1]))

        ref_obj_ids = []
        for ref_object, ref_relation in zip(ref_objects, ref_relations):
            
            # load reference object, based on the relationship between the dummy object
            ref_obj_id, ref_obj_pose = self.load_objects(
                env, [ref_object], motion="fixed", 
                constraint=True, ref_obj_id=dummy_obj_ids[0][0],
                ref_relations=ref_relation)
                
            # can fail if satisfying the constraints are impossible
            if env.failed_datagen:
                return None

            # get constraint
            _, _, obj_mask = self.get_true_image(env)
            try:
                # save constraints to decide where to put the pick object at the initial scene
                constraint_image = self.relation_to_class[ref_relation].get_constraint(obj_mask, ref_obj_id[0][0])
                if self.mode == 'test':
                    rel_constraint_image = np.logical_or(rel_constraint_image, constraint_image)
                else:
                    rel_constraint_image *= constraint_image # for pick object placement

                relations = self.get_binary_relations(dummy_obj_ids[0][0], ref_obj_id[0][0], obj_mask)
    
                if ref_relation not in relations:
                    env.failed_datagen = True
                    return None
            except:
                env.failed_datagen = True
                return None

            ref_obj_ids.append(ref_obj_id)

        # place the pick object at the place violating at least one relations in train, whole in test
        self.constraint_image = rel_constraint_image.astype(np.float64)
        rel_obj_id, rel_obj_pose = self.load_objects(
            env, [rel_object], constraint=True, motion="move", 
            conjugate=True)

        if env.failed_datagen:
            return None
        
        # add a few distractyors, randomly
        num_distractors = np.random.randint(0, self.total_objects_num-len(ref_objects))
        if num_distractors != 0:
            if not self.overlap:
                # random sample does not allow overlapping
                distractor_objects = random.sample(all_objects, k=num_distractors)
            else:
                # but choice allows overlapping
                distractor_objects = random.choices(all_objects, k=num_distractors)
            distractor_objects_id = self.load_objects(
                env, distractor_objects, distractor=True
            )
            if env.failed_datagen:
                return None

        self.delete_objects(dummy_obj_ids)
        all_obj_ids = [rel_obj_id[0][0]] + [ref_obj_id[0][0] for ref_obj_id in ref_obj_ids]

        goal_check_info = {
            "rel_idxs": [0] * len(ref_relations),
            "ref_idxs": np.arange(1, len(ref_obj_ids) + 1),
            "relations": ref_relations,
            "obj_ids": all_obj_ids,
            "original_lang_goal": utterance
        }
        
        # set goal locations
        goal_poses = self.set_goals(
            rel_obj_id,
            [[rel_gt_pose[0][0]], [rel_gt_pose[0][1]]],
            goal_check_info)

        # set task goals now really
        self.goals.append((
            rel_obj_id, np.eye(len(rel_obj_id)), goal_poses,
            False, False, 'relations', None, 1
        ))
        
        if self.language_augment:
            utterance = self.augment_language(rel_object, ref_objects, ref_relations)
        for k, v in self.replacelist.items():
            utterance = utterance.replace(k, v)
        self.lang_goals.append(utterance)
        self.lang_video_goals.append(video_utterance)

        if self.save:
            return self.labels
    
    def load_objects(
        self, env, target_objects, locs=None, distractor=False, 
            constraint=None, ref_obj_id=None, motion=None, dummy=False, 
            conjugate=False, ref_relations=None):
        
        obj_ids = []
        obj_poses = []
        constraint_fn = self.get_constraint if constraint else None

        for obj in target_objects:
            # this is the color and shape object, need to extract information
            color, target_object = obj.split(" ")
            object_urdf = self.objects[target_object]
            size = self.object_sizes[target_object]
            obj_pose = self.get_random_pose(
                env, size, constraint_fn=constraint_fn,
                ref_obj_id=ref_obj_id, conjugate=conjugate,
                ref_relations=ref_relations)
            if obj_pose[0] == None:
                print(f"Not Enough Space for {obj}: Need to Resample")
                print(ref_relations)
                env.set_failed_dategen(True)
                return None, None

            if target_object not in  ['bowl','ring'] :
                # to be more conservative for bowl and ring
                if dummy:
                    size = [size[0]+0.02, size[1]+0.02, size[2]]

                object_urdf = self.fill_template(object_urdf, {'DIM': (size[0],size[0],size[2])})
            else:
                scale = self.object_scales[target_object]
                object_urdf = self.fill_template(object_urdf, {'SCALE': (scale[0],scale[0],scale[2])})
            
            object_id = env.add_object(object_urdf, obj_pose, dummy=dummy)
            if object_id == None:
                env.set_failed_dategen(True)
                return None, None
            if os.path.exists(object_urdf):
                os.remove(object_urdf)
            obj_ids.append((object_id, (0, None)))
            obj_poses.append(obj_pose)

            p.changeVisualShape(object_id, -1, rgbaColor=utils.COLORS[color] + [1])
            
            # saving ground truths
            if not dummy:
                if not distractor:
                    env.obj_ids[motion].append(object_id)
                self.labels[object_id] = obj
        return obj_ids, obj_poses[0]
    

    def augment_language(self, rel_objcet, ref_objects, ref_relations):
        # here, you should avoid any kinds of ambiguity
        utterance = "put the "
        # need to separate color and shape
        sep = rel_objcet.split(" ")
        for obj in sep:
            if obj in self.synonym_list:
                obj = random.choice(self.synonym_list[obj]+[obj])
            utterance += f"{obj} "
        
        
        for obj, rel in zip(ref_objects, ref_relations):
            sep = obj.split(" ")
            word = ""
            for o in sep:
                if o in self.synonym_list:
                    o = random.choice(self.synonym_list[o]+[o])
                word += f"{o} "
            word = word[:-1]
            utterance += f"to the {rel} of the {word} and "
                
        final_sentence = utterance[:-5] 
                
        return final_sentence
    
    def is_match(self, pose0, pose1, symmetry):
        # To disalbe checking rotational angle, for sure
        diff_pos = np.float32(pose0[0][:2]) - np.float32(pose1[0][:2])
        dist_pos = np.linalg.norm(diff_pos)

        return (dist_pos < self.pos_eps)
    
    def get_random_pose(
        self, env, obj_size,
        constraint_fn=None, ref_obj_id=None, 
        conjugate=False, ref_relations=None):
        
        # get random position, following constraints
        # Get erosion size of object in pixels.
        max_size = max(obj_size)
        erode_size = int(np.round(max_size / self.pix_size)) + 5
        _, hmap, obj_mask = self.get_true_image(env)

        # Randomly sample an object pose within free-space pixels.
        
        free = np.ones(obj_mask.shape, dtype=np.uint8)
        free[0, :], free[:, 0], free[-1, :], free[:, -1] = 0, 0, 0, 0
        for obj_ids in env.obj_ids.values():
            for obj_id in obj_ids:
                free[obj_mask == obj_id] = 0

        # constraint is a binary image with value 1 where it satisfies contraint
        # and 0 otheriwse
        if constraint_fn != None:
            constraint_image = constraint_fn(
                obj_mask, ref_obj_id, conjugate=conjugate, 
                ref_relations=ref_relations
            )
            free = free * constraint_image.astype(np.uint8)

        free = cv2.erode(free, np.ones((erode_size, erode_size), np.uint8))
        if np.sum(free) == 0:  # no free space for this object
            return None, None

        # There is free space, place it at a random location/pose.
        pix = utils.sample_distribution(np.float32(free))
        pos = utils.pix_to_xyz(pix, hmap, self.bounds, self.pix_size)
        pos = (pos[0], pos[1], obj_size[2] / 2)
        theta = 0 # disable rotationm always fized number
        rot = utils.eulerXYZ_to_quatXYZW((0, 0, theta))
        return pos, rot
    
    def delete_objects(self, dummy_obj_ids):
        for obj_id in dummy_obj_ids:
            p.removeBody(obj_id[0])
        return
    
    def set_goals(self, obj_ids, locs, goal_check_info):
        assert len(obj_ids) == len(locs[0])

        goal_poses = []
        for obj_id, x, y in zip(obj_ids, locs[0], locs[1]):
            pose = p.getBasePositionAndOrientation(obj_id[0])
            goal_pose = (x, y, pose[0][2])
            goal_poses.append((goal_pose, pose[1], goal_check_info))
        return goal_poses

    def get_box_from_obj_id(self, obj_id, obj_mask):
        obj_loc = np.where(obj_mask == obj_id)
        x1, x2 = np.min(obj_loc[0]), np.max(obj_loc[0])
        y1, y2 = np.min(obj_loc[1]), np.max(obj_loc[1])
        return [x1, y1, x2, y2]

    def get_constraint(self, obj_mask, ref_obj_ids, ref_relations=None, conjugate=False):
        if conjugate:
            return (-self.constraint_image + 1)

        assert ref_relations is not None

        constraint_image_ = self.relation_to_class[self.conjugate[ref_relations]].get_constraint(obj_mask, ref_obj_ids)
        return constraint_image_