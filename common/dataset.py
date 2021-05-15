from common.vocabulary import Vocabulary
import torch
import torchvision

from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.transforms import Compose, Normalize, Resize, RandomHorizontalFlip, RandomVerticalFlip, RandomRotation


class MoleculesDataset(Dataset):
    def __init__(self, data_df, vocab, transform):
        self.df = data_df
        self.transform = transform
        self.vocab = vocab
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # print(f'Getting item with idx: {idx}')
        row = self.df.iloc[idx]

        pil_img = Image.open(row["image_url"])
        tensor_image = torchvision.transforms.ToTensor()(pil_img)
        
        caption_vec = []
        caption_vec += [self.vocab.stoi["<SOS>"]]
        caption_vec += self.vocab.numericalize(row["InChI"])
        caption_vec += [self.vocab.stoi["<EOS>"]]
        
        return (
            self.transform(tensor_image),
            torch.as_tensor(caption_vec)
        )
    

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self, pad_idx, batch_first=False):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)
        return imgs,targets


def retrieve_train_dataloader(dataframe, vocab: Vocabulary, batch_size=8, shuffle=True):
    pad_idx = vocab.stoi['<PAD>']
    transform = Compose([
        RandomVerticalFlip(),
        RandomHorizontalFlip(),
        RandomRotation(180),
        Resize((256,256)),
        Normalize(mean=[0.5], std=[0.5]),
    ])

    dataset = MoleculesDataset(dataframe, vocab, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=True,
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
    )

    return dataloader

def retrieve_evaluate_dataloader(dataframe, vocab: Vocabulary, batch_size=8, shuffle=True):
    pad_idx = vocab.stoi['<PAD>']
    transform = Compose([
        Resize((256,256)),
        Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = MoleculesDataset(dataframe, vocab, transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=0, 
        pin_memory=True,
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True)
    )

    return dataloader
