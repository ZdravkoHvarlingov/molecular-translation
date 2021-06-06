import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.transforms import (Compose, Normalize, Resize)


class MoleculesDatasetInference(Dataset):
    def __init__(self, data_df, transform):
        self.df = data_df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        pil_img = Image.open(row["image_url"])
        tensor_image = torchvision.transforms.ToTensor()(pil_img)

        return self.transform(tensor_image),


def retrieve_inference_dataloader(dataframe, batch_size=4):
    transform = Compose([
        Resize((256,256)),
        Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MoleculesDatasetInference(dataframe, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size,
        num_workers=0, 
        pin_memory=True,
        shuffle=False
    )

    return dataloader

def retrieve_image_tensor(image_path):
    transform = Compose([
        Resize((256,256)),
        Normalize(mean=[0.5], std=[0.5])
    ])

    pil_img = Image.open(image_path)
    tensor_image = torchvision.transforms.ToTensor()(pil_img)

    return transform(tensor_image)
