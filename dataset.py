import torch
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import GroupShuffleSplit

data_folder= "FullIJCNN2013/"
gt_file    = data_folder+"gt.txt"
data_traffic = pd.read_csv(gt_file,delimiter=";",names=["img_name","x1","y1","x2","y2","cls_label"])


def visualizeLabels():
    '''
        Once You visualize this you understand that data
        is imbalanced as well you will get the idea of how data
        is distributed.
        There are many techniques to deal with this situation
        1> remove any thing that is less frequent but in that case you are loosing information.
        2> add more data so that u have balanced dataset.
        3> Use something called weighted NN to train ur model which will take care about the imbalanced dataset.
    '''
    freq_label = data_traffic["cls_label"].value_counts()
    plt.bar(freq_label.keys(),freq_label.values)
    plt.xlabel("classes")
    plt.ylabel("frequency")
    plt.title("class vs frequency")
    plt.show()

def visualizeImages():
    '''
        In this function we basically crop the image, so that we focus on the sign part.
        Than we resize the image.
        Feel free to play around with function.
    '''
    imgName = data_traffic["img_name"][0]
    img     = cv2.imread(os.path.join(data_folder,imgName))
    img     = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img     = cv2.GaussianBlur(img,(3,3),1)
    # let's draw the region of interest around it
    x1,y1,x2,y2 = data_traffic["x1"][0],data_traffic["y1"][0],data_traffic["x2"][0],data_traffic["y2"][0]
    img  = img[y1:y2,x1:x2]
    img  = cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)

    #cv2.rectangle(img,(x1,y1),(x2,y2),[0,255,0],2)
    # plt.imshow(img,cmap='gray')
    #plt.show()
    # normalize
    #img = (img - img.min())/(img.max() - img.min())
    #img  = (img - img.mean())/(img.std())
    plt.hist(img)
    plt.show()
class dataset(torch.utils.data.Dataset):
    def __init__(self):
        super(dataset,self).__init__()
        self.traffic_data = data_traffic
        self.img_folder  = data_folder

    def __preprocessing__(self,img_path,boundingBox):
        img  = cv2.imread(img_path)
        img  = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        x1,y1,x2,y2 = boundingBox
        img  = img[y1:y2,x1:x2]
        # we perform gaussian blur
        img  = cv2.GaussianBlur(img,(3,3),2)
        img  = cv2.resize(img,(32,32),interpolation=cv2.INTER_AREA)
        img  = (img - img.mean())/(img.std())
        return torch.Tensor(img).unsqueeze(0)

    def __getitem__(self,idx):

        row  = self.traffic_data.loc[idx]
        img_path,x1,y1,x2,y2,label = row["img_name"], row["x1"],row["y1"],row["x2"],row["y2"],row["cls_label"]
        img_path = os.path.join(self.img_folder,img_path)
        img  = self.__preprocessing__(img_path,(x1,y1,x2,y2))
        return img,label

    def __len__(self):
        return len(self.traffic_data)



def read_data(batch_size):
    # first we divide the data in train test and val
    gs = GroupShuffleSplit(n_splits=2,test_size=0.2,train_size=0.8,random_state=42)
    train_indices, val_index = next(gs.split(data_traffic["img_name"],data_traffic["cls_label"],groups=data_traffic["cls_label"]))

    train_dataset = data_traffic.loc[train_indices]
    gs1 = GroupShuffleSplit(n_splits=2,test_size=0.1,train_size=0.9,random_state=42)
    train_index, test_index = next(gs1.split(train_dataset["img_name"],train_dataset["cls_label"],groups=train_dataset["cls_label"]))
    # third we apply the Weighted Random Sampler to our train data
    train_dataset_sampler = data_traffic.loc[train_index]
    class_sample_count = np.array([len(np.where(train_dataset_sampler["cls_label"]==t)[0]) for t in np.unique(train_dataset_sampler["cls_label"])])
    weight_classes = 1./class_sample_count
    weights =  [weight_classes[t] for t in train_dataset_sampler["cls_label"]]
    # and for validation we sample using SubsetRandmSampler
    validation = torch.utils.data.SubsetRandomSampler(val_index)
    test       = torch.utils.data.SubsetRandomSampler(test_index)
    train      = torch.utils.data.WeightedRandomSampler(torch.DoubleTensor(weights),num_samples=len(weights),replacement=False)

    data = dataset()

    train_loader = torch.utils.data.DataLoader(data,batch_sampler=torch.utils.data.BatchSampler(train,batch_size=batch_size,drop_last=False))
    test_loader  = torch.utils.data.DataLoader(data,batch_sampler=torch.utils.data.BatchSampler(test,batch_size=batch_size,drop_last=False))
    val_loader   = torch.utils.data.DataLoader(data,batch_sampler=torch.utils.data.BatchSampler(validation,batch_size=batch_size,drop_last=False))

    return train_loader, test_loader, val_loader

