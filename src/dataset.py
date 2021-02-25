import numpy as np # linear algebra
import torch
import torchvision.transforms as T
import imageio

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToPILImage())
    transforms.append(T.Resize(224))
    if train:
      # You can add here some data augmentation
      pass # Remove this if you add data augmentation
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

class LymphDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, lymph_counts, genders, dobs, labels = None, transforms=None):
        self.transforms = transforms
        self.img_list = images_path
        self.lymph_counts = lymph_counts
        self.genders = genders
        self.dobs = dobs
        self.labels = labels
        
        self.data_dict = {i: {"path": img_path,
                              "lymph_count": lymph_counts[i],
                              "gender": int(genders[i] == "M"),
                              "age": 2021 - int(dobs[i][-4:]),
                              "label": bool(self.labels is not None) and labels[i]
                             }
                         for i, img_path in enumerate(self.img_list)}
        
    def load_image(self, path):
        images = imageio.imread(path).astype(np.uint8)
        return images
    
    def __getitem__(self, image_id):
        images = [self.load_image(path) for path in self.data_dict[image_id]['path']]
        lymph_counts = torch.as_tensor(self.data_dict[image_id]["lymph_count"], dtype=torch.float32)
        gender = torch.as_tensor(self.data_dict[image_id]["gender"], dtype=torch.int64)
        age = torch.as_tensor(self.data_dict[image_id]["age"], dtype=torch.int64)
        label = torch.as_tensor(self.data_dict[image_id]["label"], dtype=torch.int64) if self.labels is not None else None
        
        if self.transforms:
            images = torch.cat([self.transforms(image)[None, :, :, :] for image in images], axis=0)
        
        return images, lymph_counts, gender, age, label
    
    def __len__(self):
        return len(self.data_dict)