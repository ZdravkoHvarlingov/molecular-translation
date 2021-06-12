import torch
import torchvision
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.transforms import (Compose, Normalize,
                                               RandomHorizontalFlip,
                                               RandomRotation,
                                               RandomVerticalFlip, Resize)

from common.vocabulary import Vocabulary


class MoleculesDataset(Dataset):
    def __init__(self, data_df, vocab, transform):
        self.df = data_df
        self.transform = transform
        self.vocab = vocab
        self.sos_id = vocab.stoi['<SOS>']
        self.eos_id = vocab.stoi['<EOS>']
        self.pad_id = vocab.stoi['<PAD>']
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # print(f'Getting item with idx: {idx}')
        row = self.df.iloc[idx]

        pil_img = Image.open(row["image_url"])
        tensor_image = torchvision.transforms.ToTensor()(pil_img)
        
        numericalized_inchi = self.vocab.numericalize(row["InChI"])

        caption_vec = []
        caption_vec.append(self.sos_id)
        caption_vec.extend(numericalized_inchi)
        caption_vec.append(self.eos_id)

        return (
            self.transform(tensor_image),
            torch.as_tensor(caption_vec)
        )
    

class CapsCollate:
    """
    Collate to apply the padding to the captions with dataloader
    """
    def __init__(self, pad_idx, batch_first=False, sequence_length=None):
        self.pad_idx = pad_idx
        self.batch_first = batch_first
        self.sequence_length = sequence_length
    
    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=self.batch_first, padding_value=self.pad_idx)

        if self.sequence_length is not None:
            # In this case we make targets length equal to sequence_length
            targets = self._pad_to_sequence_length(targets)
        
        return imgs,targets

    def _pad_to_sequence_length(self, targets):
        batch_sequence_length = targets.shape[1]
        if batch_sequence_length > self.sequence_length:
            raise ValueError(f'Sequence length is supposed to be more or equal to the maximum length of a sequence inside the batch. Seq length: {self.sequence_length}, batch sequence length: {batch_sequence_length}.')
        
        result = torch.ones(targets.shape[0], self.sequence_length, dtype=torch.long) * self.pad_idx
        result[:, :batch_sequence_length] = targets

        return result


def retrieve_train_dataloader(dataframe, vocab: Vocabulary, batch_size=8, shuffle=True, sequence_length=None):
    pad_idx = vocab.stoi['<PAD>']
    transform = Compose([
        # RandomVerticalFlip(),
        # RandomHorizontalFlip(),
        # RandomRotation(180),
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
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True, sequence_length=sequence_length)
    )

    return dataloader

def retrieve_evaluate_dataloader(dataframe, vocab: Vocabulary, batch_size=8, shuffle=False, sequence_length=None):
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
        collate_fn=CapsCollate(pad_idx=pad_idx,batch_first=True, sequence_length=sequence_length)
    )

    return dataloader
