import numpy as np
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
#from functools import partial

def preprocess3(image1,image2,image3):
     #final result
    transform1 = transforms.Compose([
         transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
     ]
    )
    x1 = cv2.imread(image1)
    #x1 = cv2.resize(x1, (400, 400))
    x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
    #x1 = x1.swapaxes(1,2).swapaxes(0,1)
    x1 = np.array(x1)
    x1 = x1.reshape((1,400,600))
    x2 = cv2.imread(image2)
    #x2 = cv2.resize(x2, (400, 400))
    x2 = cv2.cvtColor(x2, cv2.COLOR_BGR2GRAY)
    #x2 = x2.swapaxes(1,2).swapaxes(0,1)
    x2 = np.array(x2) 
    x2 = x2.reshape((1,400,600))
    x3 = cv2.imread(image3)
    #x3 = cv2.resize(x3, (400, 400))
    x3 = cv2.cvtColor(x3, cv2.COLOR_BGR2GRAY)
    #x3 = x3.swapaxes(1,2).swapaxes(0,1)
    x3 = np.array(x3)
    x3 = x3.reshape((1,400,600))
    X = np.concatenate((x1,x2,x3),axis=0)
    return torch.FloatTensor(X)


def preprocess1(image1,image3):
     #final result
    x1 = cv2.imread(image1)
    #x1 = cv2.resize(x1, (400, 400))
    #x1 = cv2.cvtColor(x1, cv2.COLOR_BGR2GRAY)
    #x1 = x1.swapaxes(1,2).swapaxes(0,1)
    x1 = np.array(x1)
    x1 = x1.reshape((-1,400,600))

    x3 = cv2.imread(image3)
    #x3 = cv2.resize(x3, (400, 400))
    #x3 = cv2.cvtColor(x3, cv2.COLOR_BGR2GRAY)
    #x3 = x3.swapaxes(1,2).swapaxes(0,1)
    x3 = np.array(x3)
    x3 = x3.reshape((-1,400,600))

    X = np.concatenate((x1,x3),axis=0)
    return torch.FloatTensor(X)

def preprocess2(image):
    #final result
    transform1 = transforms.Compose([
         transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
     ]
    )
    X1 = cv2.imread(image)
    X1 = cv2.resize(X1, (128, 224))
    X1 = cv2.cvtColor(X1, cv2.COLOR_BGR2GRAY)
    #X1 = X1.swapaxes(1,2).swapaxes(0,1)
    X = np.array(X1)
    X = X.reshape((1,128,224))
    return torch.FloatTensor(X)

class MyDataset(data.Dataset):
    def __init__(self, poses_xy, images, depths, loaderIMG=preprocess1, loaderDepth=preprocess2):
        # TODO
        # 1. Initialize file path or list of file names.
        self.images = images
        self.depths = depths
        self.poses_xy = poses_xy
        #self.poses_yaw = poses_yaw
        self.loaderIMG = loaderIMG
        self.loaderDepth = loaderDepth

    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        pose1_xy = self.poses_xy[index]
        pose2_xy = self.poses_xy[index+2]
        #pose1_yaw = self.poses_yaw[index]
        #pose2_yaw = self.poses_yaw[index+2]
        #pose = pose2 - pose1
        image1 = self.images[index]
        image3 = self.images[index+2]
        depth = self.depths[index+1]

        img_rgb = self.loaderIMG(image1,image3)
        img_depth = self.loaderDepth(depth)
        pose1_xy = np.array(pose1_xy)
        pose2_xy = np.array(pose2_xy)
        #pose1_yaw = np.array(pose1_yaw)
       # pose2_yaw = np.array(pose2_yaw)
        #print(pose1_xy,pose2_xy)
        pose_xy = torch.FloatTensor(pose2_xy - pose1_xy)
       # pose_yaw = torch.FloatTensor(pose2_yaw - pose1_yaw)
        return pose_xy, img_rgb, img_depth 
        

    def __len__(self):
        # You should change 0 to the total size of your dataset.
        #print(len(self.depths))
        #return int(len(self.depths)*2/3)
        return len(self.depths)-2