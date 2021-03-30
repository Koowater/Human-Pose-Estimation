import cv2
import sys
import os
import numpy as np
import utils.img

class GenerateHeatmap():
    def __init__(self, output_res, num_parts):
        self.output_res = output_res
        self.num_parts = num_parts
        sigma = self.output_res/64
        self.sigma = sigma
        size = 6 * sigma + 3
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0, y0 = 3 * sigma + 1, 3 * sigma + 1
        self.g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    def __call__(self, keypoints):
        # from keypoints, return heatmaps 
        hms = np.zeros(shape = (self.output_res, self.output_res, self.num_parts), dtype = np.float32)
        sigma = self.sigma
        for p in keypoints:
            for idx, pt in enumerate(p):
                if pt[0] > 0: 
                    x, y = int(pt[0]), int(pt[1])
                    if x < 0 or y < 0 or x >= self.output_res or y >= self.output_res:
                        continue
                    ul = int(x - 3 * sigma - 1), int(y - 3 * sigma - 1)
                    br = int(x + 3 * sigma + 2), int(y + 3 * sigma + 2)

                    c,d = max(0, -ul[0]), min(br[0], self.output_res) - ul[0]
                    a,b = max(0, -ul[1]), min(br[1], self.output_res) - ul[1]

                    cc,dd = max(0, ul[0]), min(br[0], self.output_res)
                    aa,bb = max(0, ul[1]), min(br[1], self.output_res)
                    hms[aa:bb, cc:dd, idx] = np.maximum(hms[aa:bb, cc:dd, idx], self.g[a:b,c:d])
        return hms


class Dataset():
    def __init__(self, config, annot):
        self.input_res = config['train']['input_res']
        self.output_res = config['train']['output_res']
        self.generateHeatmap = GenerateHeatmap(self.output_res, config['inference']['num_parts'])
        self.annot = annot

        self.current_path = os.path.dirname(os.path.abspath(__file__))
        sys.path.append(self.current_path)
  
        self.load_mpii(self.annot)
    
    # def __len__(self):
    #     return len(self.index)

    # def __getitem__(self, idx):
    #     return self.loadImage(self.index[idx % len(self.index)])

    def load_mpii(self, annot):
        from ref import MPII
        self.ds = MPII(annot)
        self.idx_list = np.arange(self.ds.getLength())

    def gen(self):
        np.random.shuffle(self.idx_list)
        for idx in self.idx_list:
            img, heatmaps = self.loadImage(idx)
            yield img, heatmaps

    def gen_eval(self):
        for idx in self.idx_list:
            img, keypoints, normalizing = self.loadImage(idx, True)
            yield img, keypoints, normalizing

    def loadImage(self, idx, eval=False):
        ds = self.ds
        
        ## load + crop
        orig_img = ds.get_img(idx)
        path = ds.get_path(idx)
        orig_keypoints = ds.get_kps(idx)
        kptmp = orig_keypoints.copy()
        c = ds.get_center(idx)
        s = ds.get_scale(idx)
        normalize = ds.get_normalized(idx)
        
        cropped = utils.img.crop(orig_img, c, s, (self.input_res, self.input_res))
        for i in range(np.shape(orig_keypoints)[1]):
            if orig_keypoints[0,i,0] > 0:
                orig_keypoints[0,i,:2] = utils.img.transform(orig_keypoints[0,i,:2], c, s, (self.input_res, self.input_res))
        keypoints = np.copy(orig_keypoints)
        
        ## augmentation -- to be done to cropped image
        height, width = cropped.shape[0:2]
        center = np.array((width/2, height/2))
        scale = max(height, width)/200

        aug_rot=0
        
        aug_rot = (np.random.random() * 2 - 1) * 30.
        aug_scale = np.random.random() * (1.25 - 0.75) + 0.75
        scale *= aug_scale
            
        mat_mask = utils.img.get_transform(center, scale, (self.output_res, self.output_res), aug_rot)[:2]

        mat = utils.img.get_transform(center, scale, (self.input_res, self.input_res), aug_rot)[:2]
        inp = cv2.warpAffine(cropped, mat, (self.input_res, self.input_res)).astype(np.float32)/255
        keypoints[:,:,0:2] = utils.img.kpt_affine(keypoints[:,:,0:2], mat_mask)
        if np.random.randint(2) == 0:
            inp = self.preprocess(inp)
            inp = inp[:,::-1]
            keypoints = keypoints[:, ds.flipped_parts['mpii']]
            keypoints[:,:,0] = self.output_res - keypoints[:, :, 0]
            orig_keypoints = orig_keypoints[:, ds.flipped_parts['mpii']]
            orig_keypoints[:,:,0] = self.input_res - orig_keypoints[:, :, 0]
        
        ## set keypoints to 0 when were not visible initially (so heatmap all 0s)
        for i in range(np.shape(orig_keypoints)[1]):
            if kptmp[0,i,0] == 0 and kptmp[0,i,1] == 0:
                keypoints[0,i,0] = 0
                keypoints[0,i,1] = 0
                orig_keypoints[0,i,0] = 0
                orig_keypoints[0,i,1] = 0
        
        if eval == True:
            return np.reshape(inp.astype(np.float32), (1, 256, 256, 3)), keypoints, ds.get_normalized(idx)

        ## generate heatmaps on outres
        heatmaps = self.generateHeatmap(keypoints)
        
        return np.reshape(inp.astype(np.float32), (1, 256, 256, 3)), np.reshape(heatmaps.astype(np.float32), (1, 64, 64, 16))

    def preprocess(self, data):
        # random hue and saturation
        data = cv2.cvtColor(data, cv2.COLOR_RGB2HSV);
        delta = (np.random.random() * 2 - 1) * 0.2
        data[:,:,0] = np.mod(data[:,:,0] + (delta * 360 + 360.), 360.)

        delta_sature = np.random.random() + 0.5
        data[:,:,1] *= delta_sature
        data[:,:,1] = np.maximum( np.minimum(data[:,:,1], 1), 0 )
        data = cv2.cvtColor(data, cv2.COLOR_HSV2RGB)

        # adjust brightness
        delta = (np.random.random() * 2 - 1) * 0.3
        data += delta

        # adjust contrast
        mean = data.mean(axis=2, keepdims=True)
        data = (data - mean) * (np.random.random() + 0.5) + mean
        data = np.minimum(np.maximum(data, 0), 1)
        return data
