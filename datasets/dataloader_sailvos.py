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


class SailVos_seqs_light_svd(Dataset):
    def __init__(self,
                 path,
                 rgb_base_path,
                 total_num=-1,
                 channel_num=1,
                 pair_type="ma",
                 is_crop=False,
                 is_modal_crop=False,
                 width=256,
                 height=128,
                 read_rgb=False,
                 is_raw_rgb=True,
                 read_depth=False,
                 is_raw_depth=False):

        self.path = path
        self.total_num = total_num
        self.rgb_base_path = rgb_base_path

        self.samples = self._load_samples()

        self.channel_num = channel_num
        self.pair_type = pair_type
        self.is_crop = is_crop
        self.is_modal_crop = is_modal_crop

        self.width = width
        self.height = height
        self.read_rgb = read_rgb
        self.is_raw_rgb = is_raw_rgb
        self.read_depth = read_depth
        self.is_raw_depth = is_raw_depth

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
        videos_depth = []
        modal_bboxes = []

        if "m" in self.pair_type:
            for i, seg in enumerate(modal_segs):
                tmp_frame, tmp_modal_bbox = self._process_segment(seg, amodal_bboxes[i])
                videos_modal.append(tmp_frame)
                modal_bboxes.append(tmp_modal_bbox)

        if "a" in self.pair_type:
            for i, seg in enumerate(amodal_segs):
                tmp_frame = self._process_segment2(seg, amodal_bboxes[i], modal_bboxes[i])
                videos.append(tmp_frame)

        if self.read_rgb == True:
            for i in range(len(image_file_names)):
                # rgb_path = self.rgb_base_path + image_file_names[i]
                rgb_path = self.rgb_base_path + "rgb/" + image_file_names[i]
                rgb_path = rgb_path.replace('.bmp', '.png')
                img = cv2.imread(rgb_path)
                videos_rgb_paths.append(image_file_names[i])
                if self.is_raw_rgb == False:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.width, self.height))
                videos_rgb.append(img)

            rgb_res = torch.tensor(np.array(videos_rgb), dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0

        if self.read_depth == True:
            for i in range(len(image_file_names)):
                depth_path = self.rgb_base_path + "depth/" + image_file_names[i]
                depth_path = depth_path.replace('.bmp', '.png')
                img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)


                if self.is_crop == True:
                    x, y, w, h = amodal_bboxes[i]
                    scale_w = 512 / 1280
                    scale_h = 256 / 800
                    x_new = int(x * scale_w)
                    y_new = int(y * scale_h)
                    w_new = max(int(w * scale_w), 1)
                    h_new = max(int(h * scale_h), 1)



                    img = img[int(y_new):int(y_new + h_new), int(x_new):int(x_new + w_new)]




                img = cv2.resize(img, (self.width, self.height))
                if self.is_raw_depth == False:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                videos_depth.append(img)
            if self.is_raw_depth == False:
                depth_res = torch.tensor(np.array(videos_depth), dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0
            else:
                depth_res = torch.tensor(np.array(videos_depth), dtype=torch.float32) / 127.5 - 1.0

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
            if self.is_raw_rgb == True:
                res_dict['rgb_res'] = videos_rgb
            else:
                res_dict['rgb_res'] = rgb_res

            res_dict['rgb_res_paths'] = videos_rgb_paths

        if self.read_depth == True:
            res_dict['depth_res'] = depth_res

        return res_dict

    def _process_segment(self, seg, bbox):
        mask = self._decode_coco_rle(seg, seg['size'][0], seg['size'][1])
        x, y, w, h = bbox
        modal_x, modal_y, modal_w, modal_h = self._get_bbox_from_mask(mask)

        if self.is_modal_crop:
            # cropped_object = mask[int(modal_y):int(modal_y + modal_h), int(modal_x):int(modal_x + modal_w)]
            cropped_object = self._extend_crop(mask, modal_x, modal_y, modal_w, modal_h, 3)
            mask = cropped_object

        elif self.is_crop:
            cropped_object = mask[int(y):int(y + h), int(x):int(x + w)]
            mask = cropped_object
            # mask = cv2.resize(cropped_object, (224, 224))

        mask = cv2.resize(mask, (self.width, self.height))
        final_image = np.stack((mask,) * self.channel_num, axis=-1)
        return final_image, [modal_x, modal_y, modal_w, modal_h]

    def _process_segment2(self, seg, bbox, modal_bbox):
        mask = self._decode_coco_rle(seg, seg['size'][0], seg['size'][1])
        x, y, w, h = bbox
        modal_x, modal_y, modal_w, modal_h = modal_bbox

        if self.is_modal_crop:
            # cropped_object = mask[int(modal_y):int(modal_y + modal_h), int(modal_x):int(modal_x + modal_w)]
            cropped_object = self._extend_crop(mask, modal_x, modal_y, modal_w, modal_h, 3)
            mask = cropped_object

        elif self.is_crop:
            cropped_object = mask[int(y):int(y + h), int(x):int(x + w)]
            mask = cropped_object
            # mask = cv2.resize(cropped_object, (224, 224))

        mask = cv2.resize(mask, (self.width, self.height))
        final_image = np.stack((mask,) * self.channel_num, axis=-1)
        return final_image

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

    def _extend_crop(self, mask, modal_x, modal_y, modal_w, modal_h, extension_factor=3):
        # Calculate the new width and height based on the extension factor
        extended_w = extension_factor * modal_w
        extended_h = extension_factor * modal_h

        # Calculate the new crop coordinates
        start_x = int(modal_x - (extension_factor - 1) / 2 * modal_w)
        start_y = int(modal_y - (extension_factor - 1) / 2 * modal_h)
        end_x = int(modal_x + (extension_factor + 1) / 2 * modal_w)
        end_y = int(modal_y + (extension_factor + 1) / 2 * modal_h)

        # Ensure the coordinates are within the original mask boundaries
        original_h, original_w = mask.shape
        pad_left = max(0, -start_x)
        pad_top = max(0, -start_y)
        pad_right = max(0, end_x - original_w)
        pad_bottom = max(0, end_y - original_h)

        # Create a new mask with the desired size, initialized to 0
        new_mask = np.zeros((extended_h, extended_w), dtype=mask.dtype)

        # Determine the coordinates to copy the original mask content
        crop_start_x = max(0, start_x)
        crop_start_y = max(0, start_y)
        crop_end_x = min(original_w, end_x)
        crop_end_y = min(original_h, end_y)

        # Copy the original mask content to the new mask
        new_mask[pad_top:pad_top + (crop_end_y - crop_start_y), pad_left:pad_left + (crop_end_x - crop_start_x)] = mask[
                                                                                                                   crop_start_y:crop_end_y,
                                                                                                                   crop_start_x:crop_end_x]

        return new_mask

    def _load_samples(self):
        with open(self.path, 'r') as file:
            samples = json.load(file)

        if self.total_num < 0:
            return samples
        else:
            return samples[:self.total_num]




class TAO_seqs_light_svd(Dataset):
    def __init__(self,
                 path,
                 rgb_base_path,
                 total_num=-1,
                 channel_num=1,
                 is_crop=False,
                 is_modal_crop=False,
                 read_rgb=False,
                 is_raw_rgb=True,
                 read_depth=False,
                 is_raw_depth=False):

        self.path = path
        self.total_num = total_num
        self.rgb_base_path = rgb_base_path

        self.samples = self._load_samples()

        self.channel_num = channel_num
        self.is_crop = is_crop
        self.is_modal_crop = is_modal_crop

        self.read_rgb = read_rgb
        self.is_raw_rgb = is_raw_rgb
        self.read_depth = read_depth
        self.is_raw_depth = is_raw_depth

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
                # rgb_path = self.rgb_base_path + image_file_names[i]

                rgb_path = self.rgb_base_path + image_file_names2[i]

                img = cv2.imread(rgb_path)
                if self.is_raw_rgb == False:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # img = cv2.resize(img, (self.width, self.height))
                videos_rgb.append(img)

            rgb_res = torch.tensor(np.array(videos_rgb), dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0

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
            if self.is_raw_rgb == True:
                res_dict['rgb_res'] = videos_rgb
            else:
                res_dict['rgb_res'] = rgb_res

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



class SailVos_seqs_light_svd2(Dataset):
    def __init__(self,
                 path,
                 rgb_base_path,
                 total_num=-1,
                 channel_num=1,
                 pair_type="ma",
                 is_crop=False,
                 is_modal_crop=False,
                 width=256,
                 height=128,
                 read_rgb=False,
                 is_raw_rgb=True,
                 read_depth=False,
                 is_raw_depth=False):

        self.path = path
        self.total_num = total_num
        self.rgb_base_path = rgb_base_path

        self.samples = self._load_samples()

        self.channel_num = channel_num
        self.pair_type = pair_type
        self.is_crop = is_crop
        self.is_modal_crop = is_modal_crop

        self.width = width
        self.height = height
        self.read_rgb = read_rgb
        self.is_raw_rgb = is_raw_rgb
        self.read_depth = read_depth
        self.is_raw_depth = is_raw_depth

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
        videos_depth = []
        modal_bboxes = []

        if "m" in self.pair_type:
            for i, seg in enumerate(modal_segs):
                tmp_frame, tmp_modal_bbox = self._process_segment(seg, amodal_bboxes[i])
                videos_modal.append(tmp_frame)
                modal_bboxes.append(tmp_modal_bbox)

        if "a" in self.pair_type:
            for i, seg in enumerate(amodal_segs):
                tmp_frame = self._process_segment2(seg, amodal_bboxes[i], modal_bboxes[i])
                videos.append(tmp_frame)

        if self.read_rgb == True:
            for i in range(len(image_file_names)):
                rgb_path = self.rgb_base_path + image_file_names[i]
                # rgb_path = self.rgb_base_path + "rgb/" + image_file_names[i]
                # rgb_path = rgb_path.replace('.bmp', '.png')
                img = cv2.imread(rgb_path)
                videos_rgb_paths.append(image_file_names[i])
                if self.is_raw_rgb == False:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (self.width, self.height))
                videos_rgb.append(img)

            rgb_res = torch.tensor(np.array(videos_rgb), dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0

        if self.read_depth == True:
            for i in range(len(image_file_names)):
                depth_path = self.rgb_base_path + "depth/" + image_file_names[i]
                depth_path = depth_path.replace('.bmp', '.png')
                img = cv2.imread(depth_path, cv2.IMREAD_GRAYSCALE)


                if self.is_crop == True:
                    x, y, w, h = amodal_bboxes[i]
                    scale_w = 512 / 1280
                    scale_h = 256 / 800
                    x_new = int(x * scale_w)
                    y_new = int(y * scale_h)
                    w_new = max(int(w * scale_w), 1)
                    h_new = max(int(h * scale_h), 1)



                    img = img[int(y_new):int(y_new + h_new), int(x_new):int(x_new + w_new)]




                img = cv2.resize(img, (self.width, self.height))
                if self.is_raw_depth == False:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                videos_depth.append(img)
            if self.is_raw_depth == False:
                depth_res = torch.tensor(np.array(videos_depth), dtype=torch.float32).permute(0, 3, 1, 2) / 127.5 - 1.0
            else:
                depth_res = torch.tensor(np.array(videos_depth), dtype=torch.float32) / 127.5 - 1.0

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
            if self.is_raw_rgb == True:
                res_dict['rgb_res'] = videos_rgb
            else:
                res_dict['rgb_res'] = rgb_res

            res_dict['rgb_res_paths'] = videos_rgb_paths

        if self.read_depth == True:
            res_dict['depth_res'] = depth_res

        return res_dict

    def _process_segment(self, seg, bbox):
        mask = self._decode_coco_rle(seg, seg['size'][0], seg['size'][1])
        x, y, w, h = bbox
        modal_x, modal_y, modal_w, modal_h = self._get_bbox_from_mask(mask)

        if self.is_modal_crop:
            # cropped_object = mask[int(modal_y):int(modal_y + modal_h), int(modal_x):int(modal_x + modal_w)]
            cropped_object = self._extend_crop(mask, modal_x, modal_y, modal_w, modal_h, 3)
            mask = cropped_object

        elif self.is_crop:
            cropped_object = mask[int(y):int(y + h), int(x):int(x + w)]
            mask = cropped_object
            # mask = cv2.resize(cropped_object, (224, 224))

        mask = cv2.resize(mask, (self.width, self.height))
        final_image = np.stack((mask,) * self.channel_num, axis=-1)
        return final_image, [modal_x, modal_y, modal_w, modal_h]

    def _process_segment2(self, seg, bbox, modal_bbox):
        mask = self._decode_coco_rle(seg, seg['size'][0], seg['size'][1])
        x, y, w, h = bbox
        modal_x, modal_y, modal_w, modal_h = modal_bbox

        if self.is_modal_crop:
            # cropped_object = mask[int(modal_y):int(modal_y + modal_h), int(modal_x):int(modal_x + modal_w)]
            cropped_object = self._extend_crop(mask, modal_x, modal_y, modal_w, modal_h, 3)
            mask = cropped_object

        elif self.is_crop:
            cropped_object = mask[int(y):int(y + h), int(x):int(x + w)]
            mask = cropped_object
            # mask = cv2.resize(cropped_object, (224, 224))

        mask = cv2.resize(mask, (self.width, self.height))
        final_image = np.stack((mask,) * self.channel_num, axis=-1)
        return final_image

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

    def _extend_crop(self, mask, modal_x, modal_y, modal_w, modal_h, extension_factor=3):
        # Calculate the new width and height based on the extension factor
        extended_w = extension_factor * modal_w
        extended_h = extension_factor * modal_h

        # Calculate the new crop coordinates
        start_x = int(modal_x - (extension_factor - 1) / 2 * modal_w)
        start_y = int(modal_y - (extension_factor - 1) / 2 * modal_h)
        end_x = int(modal_x + (extension_factor + 1) / 2 * modal_w)
        end_y = int(modal_y + (extension_factor + 1) / 2 * modal_h)

        # Ensure the coordinates are within the original mask boundaries
        original_h, original_w = mask.shape
        pad_left = max(0, -start_x)
        pad_top = max(0, -start_y)
        pad_right = max(0, end_x - original_w)
        pad_bottom = max(0, end_y - original_h)

        # Create a new mask with the desired size, initialized to 0
        new_mask = np.zeros((extended_h, extended_w), dtype=mask.dtype)

        # Determine the coordinates to copy the original mask content
        crop_start_x = max(0, start_x)
        crop_start_y = max(0, start_y)
        crop_end_x = min(original_w, end_x)
        crop_end_y = min(original_h, end_y)

        # Copy the original mask content to the new mask
        new_mask[pad_top:pad_top + (crop_end_y - crop_start_y), pad_left:pad_left + (crop_end_x - crop_start_x)] = mask[
                                                                                                                   crop_start_y:crop_end_y,
                                                                                                                   crop_start_x:crop_end_x]

        return new_mask

    def _load_samples(self):
        with open(self.path, 'r') as file:
            samples = json.load(file)

        if self.total_num < 0:
            return samples
        else:
            return samples[:self.total_num]


