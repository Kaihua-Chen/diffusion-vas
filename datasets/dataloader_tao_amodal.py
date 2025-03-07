from torch.utils.data import Dataset
import json
import pycocotools.mask as maskUtils
import cv2
import numpy as np
import torch

class TAO_amodal_diffusion_vas(Dataset):
    def __init__(self,
                 path,
                 rgb_base_path,
                 total_num=-1,
                 channel_num=3,
                 read_rgb=False,):

        self.path = path
        self.total_num = total_num
        self.rgb_base_path = rgb_base_path
        self.channel_num = channel_num
        self.read_rgb = read_rgb

        self.samples = self._load_samples()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        track_id = sample['track_id']
        cat_id = sample['category_id']
        vid_id = sample['video_id']

        image_file_names2 = sample['file_names']
        image_file_names = [name.split('/')[-1] for name in image_file_names2]

        modal_segs = sample['rles']
        height = sample['height']
        width = sample['width']
        amodal_bboxes = sample['amodal_bboxes']

        videos_rgb = []
        if self.read_rgb == True:
            for i in range(len(image_file_names)):
                rgb_path = self.rgb_base_path + image_file_names2[i]
                img = cv2.imread(rgb_path)
                videos_rgb.append(img)

        videos_modal = []
        for i, seg in enumerate(modal_segs):
            rle_dict = {
                "counts": seg,
                "size": [height, width]
            }
            tmp_frame = self._process_segment(rle_dict)
            videos_modal.append(tmp_frame)

        modal_res = torch.tensor(np.array(videos_modal), dtype=torch.float32).permute(0, 3, 1, 2) * 2.0 - 1.0

        amodal_bboxes = torch.tensor(amodal_bboxes, dtype=torch.float32)
        track_id = torch.tensor(track_id, dtype=torch.int32)
        cat_id = torch.tensor(cat_id, dtype=torch.int32)
        vid_id = torch.tensor(vid_id, dtype=torch.int32)

        res_dict = {}
        res_dict['modal_res'] = modal_res
        res_dict['amodal_bboxes'] = amodal_bboxes
        res_dict['track_id'] = track_id
        res_dict['cat_id'] = cat_id
        res_dict['vid_id'] = vid_id
        res_dict['image_file_names'] = image_file_names
        res_dict['height'] = height
        res_dict['width'] = width

        if self.read_rgb == True:
            res_dict['rgb_res'] = videos_rgb

        return res_dict

    def _process_segment(self, seg):
        mask = self._decode_coco_rle(seg, seg['size'][0], seg['size'][1])
        final_image = np.stack((mask,) * self.channel_num, axis=-1)
        return final_image

    def _decode_coco_rle(self, rle, height, width):
        mask = maskUtils.decode(rle)
        if len(mask.shape) < 3:
            mask = mask.reshape((height, width))
        return mask

    def _load_samples(self):
        with open(self.path, 'r') as file:
            samples = json.load(file)

        if self.total_num < 0:
            return samples
        else:
            return samples[:self.total_num]