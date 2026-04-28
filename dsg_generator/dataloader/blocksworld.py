import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import random
import numpy as np
import pickle
import os
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from fasterRCNN.lib.model.utils.config import cfg

class Blocksworld(Dataset):

    def __init__(self, mode, datasize, data_path=None):

        self.mode = mode
        root_path = data_path
        self.frames_path = os.path.join(root_path, 'frames/')

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_path, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_path, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.spatial_relationships = self.relationship_classes



        print('-------loading annotations---------slowly-----------')
        with open(os.path.join(root_path, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
            object_bbox = pickle.load(f)
        f.close()
        print('--------------------finish!-------------------------')

        if datasize == 'mini': 
            small_object = {}
            for i in list(object_bbox.keys())[:80]:
                small_object[i] = object_bbox[i]
            object_bbox = small_object


        # collect valid frames
        video_dict = {}
        for i in object_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == mode: #train or testing?
                frame_valid = False
                for j in object_bbox[i]: # the frame is valid if there is visible bbox
                    if j['visible']:
                        frame_valid = True
                if frame_valid:
                    video_name, frame_num = i.split('_')
                    if video_name in video_dict.keys():
                        video_dict[video_name].append(i)
                    else:
                        video_dict[video_name] = [i]

        self.video_list = []
        self.video_size = [] # (w,h)
        self.gt_annotations = []
        self.non_heatmap_nums = 0
        self.one_frame_video = 0
        self.valid_nums = 0

        # Loading the DSG annotation from the ground-truth data
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                gt_annotation_frame = [{"frame": j}]
                video.append(j)
                self.valid_nums += 1
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        k['identifier'] = k['identifier']
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][2], k['bbox'][3]]) # xyxy

                        spat_rel_names = torch.tensor(
                            [self.spatial_relationships.index(r) for r in k['unary_relationships']] + 
                            [self.spatial_relationships.index(r) for r in k['binary_relationships'] for _ in range(len(k['binary_relationships'][r]))],
                            dtype=torch.long
                        )
                        spat_rel_args = list(
                            [k['identifier'] for r in k['unary_relationships']] + 
                            [ele for r in k['binary_relationships'] for ele in k['binary_relationships'][r]]
                        )
                        k['spatial_relationship'] = spat_rel_names, spat_rel_args # pass on combination of relationships + other object ids
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            self.video_list.append(video)
            self.gt_annotations.append(gt_annotation_video)

        print('x'*60)
        print('There are {} videos and {} valid frames'.format(len(self.video_list), self.valid_nums))
        print('x' * 60)

    def __getitem__(self, index):

        # Get the pixels and gt information for a single datapoint
        frame_names = self.video_list[index]
        processed_ims = []
        im_scales = []

        for idx, name in enumerate(frame_names):
            import imageio
            im = imageio.imread(os.path.join(self.frames_path, name))

            if len(im.shape) == 2:
                im = im[:,:,np.newaxis]
                im = np.concatenate((im,im,im), axis=2)
            elif im.shape[2] == 4:
                im = im[:,:,:3]

            # flip the channel, since the original one using cv2
            # rgb -> bgr
            im = im[:,:,::-1]



            im, im_scale = prep_im_for_blob(im, cfg.PIXEL_MEANS, cfg.TRAIN.SCALES[0], cfg.TRAIN.MAX_SIZE) 
            im_scales.append(im_scale)
            processed_ims.append(im)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)
        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index

    def __len__(self):
        return len(self.video_list)

def cuda_collate_fn(batch):
    # Dummy function, skip zipping the batch
    return batch[0]
