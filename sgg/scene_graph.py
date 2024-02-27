import os
from torchvision.ops import box_convert, nms
from PIL import Image
import torch
from groundingdino.util.inference import load_model, predict
import groundingdino.datasets.transforms as T


class SceneGraphWithGDino():
    def __init__(
            self,
            groundingDino_rootpath='./sgg/GroundingDINO',
            TEXT_PROMPT="objects",
            BOX_TRESHOLD=0.2,
            TEXT_TRESHOLD=0.25,
            IOU_THRESHOLD=0.4,
            disable_load=False
            ):
        CONFIG_PATH = os.path.join(groundingDino_rootpath, 'groundingdino/config/GroundingDINO_SwinT_OGC.py')
        WEIGHTS_PATH = os.path.join(groundingDino_rootpath, "weights/groundingdino_swint_ogc.pth")
        assert os.path.isfile(CONFIG_PATH)
        assert os.path.isfile(WEIGHTS_PATH)
        if disable_load:
            self.mode = None
        else:
            self.model = load_model(CONFIG_PATH, WEIGHTS_PATH)
        
        self.TEXT_PROMPT=TEXT_PROMPT
        self.BOX_TRESHOLD=BOX_TRESHOLD
        self.TEXT_TRESHOLD=TEXT_TRESHOLD
        self.IOU_THRESHOLD=IOU_THRESHOLD
        
        self.boxes = None
        self.scene_graph = None

    def detect_boxes(self, image):
        # ************* convert image to PIL Image ('RGB')
        """
        return : bounding_boxes, confidence_scores
        :: bounding_boxes    = dict( object_idx : [x1,y1,x2,y2])
        :: confidence_scores = dict( object_idx : score)
        """
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_source = Image.fromarray(image).convert("RGB")
        image_transformed, _ = transform(image_source, None)

        boxes, logits, phrases = predict(
            model=self.model, 
            image=image_transformed, 
            caption=self.TEXT_PROMPT, 
            box_threshold=self.BOX_TRESHOLD, 
            text_threshold=self.TEXT_TRESHOLD
        )
        h,w,_ = image.shape
        xyxy = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=xyxy, in_fmt="cxcywh", out_fmt="xyxy")
        keep_indices = nms(xyxy, logits, self.IOU_THRESHOLD)
        phrases = [phrases[i] for i in keep_indices]
        
        xyxy, logits = xyxy[keep_indices], logits[keep_indices]       
        final_boxes = dict()
        final_conf = dict()
        box_new_id = 0
        for id in range(xyxy.shape[0]):
            x1,y1,x2,y2 = xyxy[id].to(dtype=torch.int32).tolist()
            if (x2-x1)*(y2-y1) > 500*300:
                continue
            final_boxes[box_new_id] = xyxy[id].to(dtype=torch.int32).tolist()
            final_conf[box_new_id] = float(logits[id])
            box_new_id += 1
        
        self.boxes = final_boxes
        return final_boxes, final_conf

    def make_sg(self, boxes=None):
        """
        return scene graph based on self.boxes
        :: scene_graph = list( (id_1, 'in', id_2), ...)
        """

        def is_in(box1, box2):
            # box1 is in box2 >> return True
            return box1[0] > box2[0] and box1[1] > box2[1] and box1[2] < box2[2] and box1[3] < box2[3]
        
        def is_near(box1, box2):
            cx1,cy1 = (box1[0]+box1[2])/2, (box1[1]+box1[3])/2
            cx2,cy2 = (box2[0]+box2[2])/2, (box2[1]+box2[3])/2       
            return ((cx1-cx2)**2+(cy1-cy2)**2) < (150**2)
        
        if boxes == None:
            boxes = self.boxes
        sg = []
        for i1, id1 in enumerate(boxes.keys()):
            for i2, id2 in enumerate(boxes.keys()):
                if i1 >= i2:
                    continue
                box1, box2 = boxes[id1], boxes[id2]
                if is_in(box1, box2):
                    sg.append((id1, 'in', id2))
                elif is_in(box2, box1):
                    sg.append((id2, 'in', id1))
                elif is_near(box1,box2):
                    sg.append((id1, 'near', id2))
                    sg.append((id2, 'near', id1))
        self.scene_graph = sg
        return sg
