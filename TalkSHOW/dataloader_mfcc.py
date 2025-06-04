from mfcc_dataset import MFCCDataset
from torch.utils.data import DataLoader

def create_mfcc_dataloader(mfcc_folder, batch_size=32, num_workers=4):
    dataset = MFCCDataset(mfcc_folder)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=False
    )
    return dataloader

