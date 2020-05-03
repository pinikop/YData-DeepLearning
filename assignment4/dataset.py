import os
from PIL import Image

from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor


class PascalDataset(Dataset):
    
    def __init__(self, root, indices=None, 
                 transform_x=None, 
                 transform_y_mid=None, 
                 transform_y_large=None):
        
        super().__init__()
        
        self.X_size = (72, 72)
        self.y_mid_size = (144, 144)
        self.y_large_size = (288, 288)
        
        self.root = os.path.join(root, 'JPEGImages')
        self.image_list = os.listdir(self.root)
        
        self.indices = indices
        
        if self.indices:
            self.image_list = [self.image_list[i] for i in self.indices]
            
        self.transform_X = self._transform(self.X_size, transform_x)
        self.transform_y_mid = self._transform(self.y_mid_size, transform_y_mid)
        self.transform_y_large = self._transform(self.y_large_size, transform_y_large)


    @staticmethod
    def _transform(size, transform):
        base_transform = Compose([Resize(size), ToTensor()])
        
        return (Compose([transform, base_transform]) 
                if transform else base_transform)
                                    
                                      
    def __len__(self):
        return len(self.image_list)
    
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_list[idx])
        img = Image.open(img_path)
        
        X = self.transform_X(img)
        y_mid = self.transform_y_mid(img)
        y_large = self.transform_y_large(img)
        
        return X, y_mid, y_large