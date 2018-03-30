
import numpy as np
import torch.utils.data as data
import csv
import cv2
from PIL import Image
from utils import progress_bar

class CamelDataset(data.Dataset):
    """ camelyon17 dataset class for pytorch dataloader

    """
    
    def __init__(self, csv_path='train.csv', limit=0, transform=None, target_transform=None):
        super(CamelDataset, self).__init__()
        self.transform = transform
        self.target_transform = target_transform
        self.data = []
        self.labels = []

        csv_file = open(csv_path, 'r', encoding='utf-8')
        csv_reader = csv.reader(csv_file)
        cnt = 0
        
        p_rows = []
        n_rows = []
        for img, label in csv_reader:
            if label == '1':
                p_rows.append([img, label])
            else:
                n_rows.append([img, label])
        rows = zip(p_rows, n_rows)
        min_len = min(len(p_rows), len(n_rows))
        for (p_img, p_label), (n_img, n_label) in rows:
            progress_bar(cnt, min_len, csv_path)
            p_array = cv2.imread(p_img, cv2.IMREAD_COLOR)
            self.data.append(p_array)
            self.labels.append(p_label)
            n_array = cv2.imread(n_img, cv2.IMREAD_COLOR)
            self.data.append(n_array)
            self.labels.append(n_label)
            cnt += 1
            if cnt > limit / 2:
                break

        self.data = np.array(self.data)
        self.labels = np.array(self.labels)


    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        
        img = self.data[index]
        img = Image.fromarray(img)
        target = self.labels[index]
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target 


    def __len__(self):
        return len(self.data)


