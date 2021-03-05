import imageio
import numpy as np  # linear algebra
import torch
import torchvision.transforms as T


def get_transform(train):
    transforms = []
    transforms.append(T.ToPILImage())  # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.Resize(224))
    transforms.append(T.GaussianBlur(3, sigma=.5))
    if train:
        transforms.append(T.RandomRotation(180))
        transforms.append(T.RandomHorizontalFlip())
        transforms.append(T.RandomVerticalFlip())
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    return T.Compose(transforms)


class LymphDataset(torch.utils.data.Dataset):
    def __init__(self, images_path, df, transforms=None):
        self.images_path = images_path
        self.df = df
        self.transforms = transforms

        self.data_dict = {i: {"path": paths,
                              "lymph_count": self.df.LYMPH_COUNT[i],
                              "gender": self.df.GENDER[i],
                              "age": self.df.AGE[i],
                              "label": self.df.LABEL[i]
                              }
                          for i, paths in enumerate(self.images_path)}

    def load_image(self, path):
        images = imageio.imread(path).astype(np.uint8)
        return images

    def __getitem__(self, patient_id):
        images = [self.load_image(path) for path in self.data_dict[patient_id]['path']]
        medical_data = torch.as_tensor([
            self.data_dict[patient_id]["lymph_count"],
            self.data_dict[patient_id]["gender"],
            self.data_dict[patient_id]["age"]], dtype=torch.float32)
        label = torch.as_tensor(self.data_dict[patient_id]["label"], dtype=torch.int64)
        
        if self.transforms:
            images = torch.stack([self.transforms(image) for image in images])
        
        return images, medical_data, label

    def __len__(self):
        return len(self.data_dict)
        #return 4 if len(self.data_dict) != 42 else 42
