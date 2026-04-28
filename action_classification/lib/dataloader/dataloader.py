import torch
import numpy as np
import os
import pickle
from config.config_loader import CONFIG

from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from fasterRCNN.lib.model.utils.config import cfg

class BlocksworldSequences(torch.utils.data.Dataset):
    """Blocksworld image dsg_sequences dataset."""

    def __init__(self, mode, root_dir):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.mode = mode
        self.root_dir = root_dir
        self.frames_path = os.path.join(self.root_dir, 'frames/')

        # collect the object classes
        self.object_classes = ['__background__']
        with open(os.path.join(root_dir, 'annotations/object_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.object_classes.append(line)
        f.close()

        # collect relationship classes
        self.relationship_classes = []
        with open(os.path.join(root_dir, 'annotations/relationship_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.relationship_classes.append(line)
        f.close()
        self.spatial_relationships = self.relationship_classes

        # collect possible actions
        self.action_name_classes = [""]
        with open(os.path.join(root_dir, 'annotations/action_name_classes.txt'), 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                self.action_name_classes.append(line)
        f.close()        

        self.num_classes = len(self.action_name_classes)
        self.dsg_sequences = []
        self.adj_matrices = []
        self.labels = []

        # load the data
        self.load_data()

    def load_data(self):
        print('-------loading annotations---------'+self.mode+'-----------')
        # object bbox
        with open(os.path.join(self.root_dir, 'annotations/object_bbox_and_relationship.pkl'), 'rb') as f:
            object_bbox = pickle.load(f)
        f.close()

        # actions
        with open(os.path.join(self.root_dir, 'annotations/action_descriptions.pkl'), 'rb') as f:
            actions = pickle.load(f)
        f.close()
        print('--------------------finish!-------------------------')

        # Load debug conf
        conf = CONFIG.ac
        datasize = conf['datasize']
        if datasize == 'small':
            if self.mode == 'train':
                object_bbox = dict(list(object_bbox.items())[:80])
                actions = dict(list(actions.items())[:80])
            else:
                object_bbox = dict(list(object_bbox.items())[-20:])
                actions = dict(list(actions.items())[-20:])
        elif datasize == 'overfit':
            if self.mode == 'train':
                object_bbox = dict(list(object_bbox.items())[:10])
                actions = dict(list(actions.items())[:10])
            else:
                object_bbox = dict(list(object_bbox.items())[-10:])
                actions = dict(list(actions.items())[-10:])
        elif datasize == 'full':
            pass


        # collect valid frames ##############################
        video_dict = {}
        for i in object_bbox.keys():
            if object_bbox[i][0]['metadata']['set'] == self.mode: #train or testing?
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
        self.dsg_gt_annotations = []
        self.action_annotations = []
        self.valid_nums = 0

        # extract the object bounding boxes + relationships
        for i in video_dict.keys():
            video = []
            gt_annotation_video = []
            for j in video_dict[i]:
                gt_annotation_frame = [{"frame": j}]
                video.append(j)
                self.valid_nums += 1
                # each frames's objects and human
                for k in object_bbox[j]:
                    if k['visible']:
                        assert k['bbox'] != None, 'warning! The object is visible without bbox'
                        k['class'] = self.object_classes.index(k['class'])
                        k['identifier'] = k['identifier']
                        k['bbox'] = np.array([k['bbox'][0], k['bbox'][1], k['bbox'][2], k['bbox'][3]]) # xyxy

                        spat_rel_names = torch.tensor(
                            [self.spatial_relationships.index(r) for r in k['unary_relationships']] + 
                            [self.spatial_relationships.index(r) for r in k['binary_relationships']],
                            dtype=torch.long
                        )
                        spat_rel_args = list(
                            [k['identifier'] for r in k['unary_relationships']] + 
                            [k['binary_relationships'][r][0] for r in k['binary_relationships']]
                        )
                        k['spatial_relationship'] = spat_rel_names, spat_rel_args # pass on combination of relationships + other object ids
                        gt_annotation_frame.append(k)
                gt_annotation_video.append(gt_annotation_frame)

            self.video_list.append(video)
            self.dsg_gt_annotations.append(gt_annotation_video)

        # extract the action sequences
        video_dict = {}
        for i in actions.keys():
            if actions[i]['metadata']['set'] == self.mode: #train or testing?
                video_name, frame_num = i.split('_')
                if video_name in video_dict.keys():
                    video_dict[video_name].append(i)
                else:
                    video_dict[video_name] = [i]
        for i in video_dict.keys():
            
            gt_annotation_video = []
            for j in video_dict[i]:
                gt_annotation_frame = {"frame": j}
                gt_annotation_frame['predicate'] = actions[j]['predicate']
                gt_annotation_frame['args'] = actions[j]['args']
                gt_annotation_video.append(gt_annotation_frame)

            self.action_annotations.append(gt_annotation_video)

        print('x'*60)
        print('There are {} image sequences and {} valid frames'.format(len(self.video_list), self.valid_nums))
        print('x' * 60)
    

    def __len__(self):
        """Return the length of the dataset."""
        return len(self.video_list)

    def __getitem__(self, index):
        """Get a sample from the dataset."""        
        frame_names = self.video_list[index]
        processed_ims = []
        gt_boxes = []
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

            # get gtbox for each object
            objs = self.dsg_gt_annotations[index][idx][1:]
            gt_box_item = []
            for obj in objs:
                bbox = obj['bbox']# * im_scale
                cls = obj['class']
                gt_box_item.append([bbox[0], bbox[1], bbox[2], bbox[3], cls]) # xyxy
            gt_boxes.append(gt_box_item)

        blob = im_list_to_blob(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        # check for sequence length missmatch
        seq_lens = [len(f) for f in gt_boxes]
        if min(seq_lens) != max(seq_lens):
            # padd all other frames to the max length
            for s_i in range(len(gt_boxes)):
                while len(gt_boxes[s_i]) < max(seq_lens):
                    gt_boxes[s_i].append([5,5,10,10,0]) # pad with dummy box


        try:
            gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)
        except:
            import pdb; pdb.set_trace()

        #gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

        return img_tensor, im_info, gt_boxes, num_boxes, index

def cuda_collate_fn(batch):
    """
    don't need to zip the tensor

    """
    return batch[0]