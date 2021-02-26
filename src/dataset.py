import numpy as np # linear algebra
import torch
import torchvision.transforms as T
import imageio

def get_transform(train):
    transforms = []
    transforms.append(T.ToPILImage()) # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.Resize(224))
    if train:
      pass
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

class LymphDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, df, transforms=None):
        self.images_path = images_path
        self.df = df
        self.transforms = transforms
        
        self.data_dict = {i: {"path": paths,
                              "lymph_count": self.df.LYMPH_COUNT[i],
                              "gender": int(self.df.GENDER[i] == "M"),
                              "age": 2021 - int(self.df.DOB[i][-4:]),
                              "label": self.df.LABEL[i]
                             }
                         for i, paths in enumerate(self.images_path)}
        
    def load_image(self, path):
        images = imageio.imread(path).astype(np.uint8)
        return images
    
    def __getitem__(self, image_id):
        images = [self.load_image(path) for path in self.data_dict[image_id]['path']]
        lymph_counts = torch.as_tensor(self.data_dict[image_id]["lymph_count"], dtype=torch.float32)
        gender = torch.as_tensor(self.data_dict[image_id]["gender"], dtype=torch.int64)
        age = torch.as_tensor(self.data_dict[image_id]["age"], dtype=torch.int64)
        label = torch.as_tensor(self.data_dict[image_id]["label"], dtype=torch.int64)
        
        if self.transforms:
            images = torch.cat([self.transforms(image)[None, :, :, :] for image in images], axis=0)
        
        return images, lymph_counts, gender, age, label
    
    def __len__(self):
        return len(self.data_dict)
        #return 2 if len(self.data_dict) != 42 else 42