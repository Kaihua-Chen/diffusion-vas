from torch.utils.data import Dataset
import json
import pycocotools.mask as maskUtils
import cv2
import time
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import time


class SailVos_diffusion_vas(Dataset):
    def __init__(self,
                 path,
                 rgb_base_path,
                 total_num=-1,
                 channel_num=3,
                 width=256,
                 height=128,
                 read_rgb=False,):

        self.path = path
        self.total_num = total_num
        self.rgb_base_path = rgb_base_path
        self.channel_num = channel_num
        self.width = width
        self.height = height
        self.read_rgb = read_rgb

        self.samples = self._load_samples()


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        amodal_segs = sample['segmentation']
        modal_segs = sample['visible_mask']
        amodal_bboxes = sample['bbox']
        image_ids = sample['image_ids']
        obj_id = sample['obj_id']
        cat_id = sample['category_id']
        image_file_names = sample['image_file_names']

        videos, videos_modal, videos_rgb = [], [], []
        videos_rgb_paths = []
        modal_bboxes = []

        for i, seg in enumerate(modal_segs):
            tmp_frame, tmp_modal_bbox = self._process_segment(seg)
            videos_modal.append(tmp_frame)
            modal_bboxes.append(tmp_modal_bbox)

        for i, seg in enumerate(amodal_segs):
            tmp_frame, _ = self._process_segment(seg)
            videos.append(tmp_frame)

        if self.read_rgb == True:
            for i in range(len(image_file_names)):
                rgb_path = self.rgb_base_path + image_file_names[i]
                # rgb_path = self.rgb_base_path + "rgb/" + image_file_names[i]
                # rgb_path = rgb_path.replace('.bmp', '.png')

                img = cv2.imread(rgb_path)
                videos_rgb_paths.append(image_file_names[i])
                img = cv2.resize(img, (self.width, self.height))
                videos_rgb.append(img)

            rgb_res = torch.tensor(np.array(videos_rgb), dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0


        modal_res = torch.tensor(np.array(videos_modal), dtype=torch.float32).permute(0, 3, 1, 2) * 2.0 - 1.0
        amodal_res = torch.tensor(np.array(videos), dtype=torch.float32).permute(0, 3, 1, 2) * 2.0 - 1.0

        amodal_bboxes = torch.tensor(amodal_bboxes, dtype=torch.float32)
        modal_bboxes = torch.tensor(modal_bboxes, dtype=torch.float32)

        image_ids = torch.tensor(image_ids, dtype=torch.int32)
        obj_id = torch.tensor(obj_id, dtype=torch.int32)
        cat_id = torch.tensor(cat_id, dtype=torch.int32)

        res_dict = {}
        res_dict['amodal_res'] = amodal_res
        res_dict['modal_res'] = modal_res
        res_dict['amodal_bboxes'] = amodal_bboxes
        res_dict['modal_bboxes'] = modal_bboxes
        res_dict['image_ids'] = image_ids
        res_dict['obj_id'] = obj_id
        res_dict['cat_id'] = cat_id

        if self.read_rgb == True:
            res_dict['rgb_res'] = videos_rgb
            res_dict['rgb_res_paths'] = videos_rgb_paths

        return res_dict

    def _process_segment(self, seg):
        mask = self._decode_coco_rle(seg, seg['size'][0], seg['size'][1])
        modal_x, modal_y, modal_w, modal_h = self._get_bbox_from_mask(mask)
        mask = cv2.resize(mask, (self.width, self.height))
        final_image = np.stack((mask,) * self.channel_num, axis=-1)
        return final_image, [modal_x, modal_y, modal_w, modal_h]

    def _decode_coco_rle(self, rle, height, width):
        mask = maskUtils.decode(rle)
        if len(mask.shape) < 3:
            mask = mask.reshape((height, width))
        return mask

    def _get_bbox_from_mask(self, mask):
        # Find the coordinates of the non-zero values in the mask
        y_coords, x_coords = np.nonzero(mask)

        # If there are no non-zero values, return an empty bbox
        if len(y_coords) == 0 or len(x_coords) == 0:
            return None

        # Get the bounding box coordinates
        x_min = np.min(x_coords)
        x_max = np.max(x_coords)
        y_min = np.min(y_coords)
        y_max = np.max(y_coords)

        # Calculate width and height
        width = x_max - x_min + 1
        height = y_max - y_min + 1

        # Return the bounding box as [x_min, y_min, width, height]
        return [x_min, y_min, width, height]


    def _load_samples(self):
        with open(self.path, 'r') as file:
            samples = json.load(file)

        if self.total_num < 0:
            return samples
        else:
            return samples[:self.total_num]

