import os, sys
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import scipy
from skimage.io import imread, imsave
from skimage.transform import estimate_transform, warp, resize, rescale
from glob import glob
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm
import pickle

class CelebV_hq(Dataset):
    def __init__(self, K, image_size, scale, trans_scale = 0, isTemporal=False, isEval=False, isSingle=False, sep_num=(0,-100)):
        '''
        K must be less than 6
        '''
        self.K = K
        self.image_size = image_size
        self.imagefolder = '/data/vision_team/CelebV-HQ/35666'
        self.kptfolder = '/data/vision_team/CelebV-HQ/kpt'
        self.segfolder ='/data/vision_team/CelebV-HQ/segmentation'
        self.pklfolder = '/data/vision_team/CelebV-HQ_FLAME'

        self.pkllist = glob(self.pklfolder + '/*/*0.pkl')[sep_num[0]:sep_num[1]]
        self.datalist = []
        for fname in tqdm(self.pkllist) :
            path,n = os.path.split(fname)
            _, vname = os.path.split(path)
            n = vname + '_frame' + n[:-4]
            kpt_name = path.replace('CelebV-HQ_FLAME','CelebV-HQ/kpt') +'/'+n+'.npy'
            image_name = path.replace('CelebV-HQ_FLAME','CelebV-HQ/35666')+'/'+n+'.jpg'
            seg_name = path.replace('CelebV-HQ_FLAME','CelebV-HQ/segmentation')+'/'+n+'.png'

            if os.path.isfile(kpt_name) and os.path.isfile(image_name) and os.path.isfile(seg_name):
                self.datalist.append(image_name)

        
        self.isTemporal = isTemporal
        self.scale = scale #[scale_min, scale_max]
        self.trans_scale = trans_scale #[dx, dy]
        self.isSingle = isSingle
        if isSingle:
            self.K = 1

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        images_list = []; kpt_list = []; mask_list = []; pkl_list = []
        for i in range(self.K):
            image_path = self.datalist[idx]

            
            kpt_path = os.path.join(image_path.replace('35666','kpt')[:-3] + 'npy')  
            seg_path = os.path.join(image_path.replace('35666','segmentation')[:-3] + 'png')

            p,n = os.path.split(image_path)
            frame_id = n[-8:-4]
            pkl_path = os.path.join(p.replace('CelebV-HQ/35666','CelebV-HQ_FLAME'), frame_id + '.pkl')
                                            
            image = imread(image_path)/255.
            kpt = np.load(kpt_path)[:,:2]
            mask = np.array(imread(seg_path, cv2.IMREAD_UNCHANGED))/255.
            with open(pkl_path, 'rb') as f:
	            pkl = pickle.load(f)

            ### crop information
            tform = self.crop(image, kpt)
            ## crop 
            cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_mask = warp(mask, tform.inverse, output_shape=(self.image_size, self.image_size))
            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)

            # normalized kpt
            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            images_list.append(cropped_image.transpose(2,0,1))
            kpt_list.append(cropped_kpt)
            mask_list.append(cropped_mask)
            pkl_list.append(pkl)

        ###
        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3
        mask_array = torch.from_numpy(np.array(mask_list)).type(dtype = torch.float32) #K,224,224,3

        if self.isSingle:
            images_array = images_array.squeeze()
            kpt_array = kpt_array.squeeze()
            mask_array = mask_array.squeeze()
                    
        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'mask': mask_array,
            'pkl' : pkl_list
        }
        
        return data_dict

    def crop(self, image, kpt):
        left = np.min(kpt[:,0]); right = np.max(kpt[:,0]); 
        top = np.min(kpt[:,1]); bottom = np.max(kpt[:,1])

        h, w, _ = image.shape
        old_size = (right - left + bottom - top)/2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 ])#+ old_size*0.1])
        # translate center
        trans_scale = (np.random.rand(2)*2 -1) * self.trans_scale
        center = center + trans_scale*old_size # 0.5

        scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        size = int(old_size*scale)

        # crop image
        src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        DST_PTS = np.array([[0,0], [0,self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        
        # cropped_image = warp(image, tform.inverse, output_shape=(self.image_size, self.image_size))
        # # change kpt accordingly
        # cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T # np.linalg.inv(tform.params)
        return tform
    