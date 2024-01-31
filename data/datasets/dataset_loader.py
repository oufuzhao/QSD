# 


# encoding: utf-8

import numpy as np
import os.path as osp
from PIL import Image
from torch.utils.data import Dataset

import json
import random


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset_name, dataset, quality_dict, transform=None):
        self.dataset = dataset
        self.transform = transform
        self.quality_dict = quality_dict

        import pdb
        print('='*20)
        #print(dataset)
        #pdb.set_trace()
        
        outfile = f'/root/autodl-tmp/Re-ID_datasets/piddict-{dataset_name}.json'
        #import numpy as np
        id_num = set()
        id_set = []
        for _, tmp_id, _ in self.dataset:
            id_num.add(int(tmp_id))
            id_set.append(int(tmp_id))
        id_num = sorted(list(id_num), reverse=False)
        id_set = np.asarray(id_set)
        # Build dictionary and feed into data
        dict_data = {}
        for i in id_num:
            tmp_idx = list(np.where(id_set == int(i))[0])
            dict_data[str(i)] = [ii for ii in tmp_idx]
        print("Done! The Output Json File is {}".format(dict_to_json(dict_data, outfile)))
         
        with open(outfile, 'r') as inf: self.piddict=json.load(inf) 
        #with open('/data/home/fuzhaoou/Projects/Re-ID-mine/toDataset/cuhk03/piddict.json','r') as inf: self.piddict=json.load(inf)
        #with open('/data/home/fuzhaoou/Projects/Re-ID-mine/toDataset/msmt17/piddict.json','r') as inf: self.piddict=json.load(inf)
        #with open('/data/home/fuzhaoou/Projects/Re-ID-mine/toDataset/dukemtmc-reid/piddict.json','r') as inf: self.piddict=json.load(inf)
        #with open('/data/home/fuzhaoou/Projects/Re-ID-mine/toDataset/market1501/piddict.json','r') as inf: self.piddict=json.load(inf)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)
        if self.transform is not None:
            img = self.transform(img)
            
        index2=index
        pid_allindex=self.piddict[str(pid)]
        while index2==index:
            sub_index=random.randint(0,len(pid_allindex)-1)
            index2=pid_allindex[sub_index]
          
        img_path2, pid2, camid2 = self.dataset[index2]
        img2 = read_image(img_path2)
        if self.transform is not None:
            img2 = self.transform(img2)
        try: 
            q_label = self.quality_dict[img_path]
            q_label2 = self.quality_dict[img_path2]
        except: 
            # If the sample does not have quality annotations, set the same quality annotations.
            q_label = 50.0
            q_label2 = 50.0
        return img, pid, camid, img_path, q_label, img2, pid2, camid2, img_path2, q_label2

class ImageDataset_ori(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def dict_to_json(dict_data, save_path):
    # cls=NpEncoder, 解决 xxx is not JSON serializable
    json_data = json.dumps(dict_data, cls=NpEncoder)
    # with open('save_path', 'w') as f:
    #     f.write()
    outfile = open(save_path, 'w')
    outfile.write(json_data)
    outfile.close()
    return save_path






















# encoding: utf-8


# import os.path as osp
# from PIL import Image
# from torch.utils.data import Dataset

# import json
# import random


# def read_image(img_path):
#     """Keep reading image until succeed.
#     This can avoid IOError incurred by heavy IO process."""
#     got_img = False
#     if not osp.exists(img_path):
#         raise IOError("{} does not exist".format(img_path))
#     while not got_img:
#         try:
#             img = Image.open(img_path).convert('RGB')
#             got_img = True
#         except IOError:
#             print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
#             pass
#     return img


# class ImageDataset(Dataset):
#     """Image Person ReID Dataset"""

#     def __init__(self, dataset, transform=None):
#         self.dataset = dataset
#         self.transform = transform
#         with open('/root/autodl-tmp/Re-ID_datasets/cuhk03/piddict.json','r') as inf:
#             self.piddict=json.load(inf)

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, index):
#         img_path, pid, camid = self.dataset[index]
#         img = read_image(img_path)
#         if self.transform is not None:
#             img = self.transform(img)
            
#         index2=index
#         pid_allindex=self.piddict[str(pid)]
# #        print('===')
# #        print('pid_allindex',pid_allindex)
# #        print('len(pid_allindex)',len(pid_allindex))
# #        print('pid',pid)
#         while index2==index:
#             sub_index=random.randint(0,len(pid_allindex)-1)
# #            print('sub_index',sub_index)
#             index2=pid_allindex[sub_index]
# #            print('index2',index2)
            
#         img_path2, pid2, camid2 = self.dataset[index2]
#         img2 = read_image(img_path2)
#         if self.transform is not None:
#             img2 = self.transform(img2)
            

#         return img, pid, camid, img_path, img2, pid2, camid2, img_path2



class ImageDataset_Test(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path