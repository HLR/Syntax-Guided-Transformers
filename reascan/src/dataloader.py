import torch
import json
import numpy as np


class MyData:
    def __init__(self,input_command,target_sequence,target_location,situation,mask) -> None:
        self.input_command = input_command
        self.target_sequence = target_sequence
        self.target_location = target_location
        self.situation = situation
        self.mask = mask
    
    def __str__(self) -> str:
        return str({
        "input_command":self.input_command,
        "target_sequence":self.target_sequence,
        "target_location":self.target_location,
        "mask":self.mask,
        "situation":self.situation
        })
        
def collate_batch(batch):
    
    data = {
        "input_command":[],
        "target_sequence":[],
        "target_location":[],
        "mask":[],
        "situation":[]
        }
    
    for item in batch:
        data["input_command"].append(item["input_command"])
        data["target_sequence"].append(item["target_sequence"])
        data["target_location"].append(item["target_location"])
        data["situation"].append(item["situation"])
        if item.get("mask"):
            data["mask"].append(torch.tensor(np.array(item["mask"])))
        else:
            data["mask"] = None
    data["input_command"] = data["input_command"]
    data["target_sequence"] = data["target_sequence"]
    data["target_location"] = torch.stack(data["target_location"])
    data["mask"] = data["mask"]
    data["situation"] = torch.stack(data["situation"])
    return MyData(data["input_command"],data["target_sequence"],data["target_location"],data["situation"],data["mask"])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, device):
        self.device = device
        self.all_data = []
        for line in open(data_path).readlines():
            line_data = json.loads(line.strip())
            line_data['target_location'] = torch.tensor((line_data['target_location'])).to(device)
            line_data['situation'] = torch.tensor((line_data['situation'])).to(device)
            self.all_data.append(line_data)
        self.data = self.all_data

    def set_limit(self, limit):
        self.data = list(filter(lambda x:len(x["input_command"]) <= limit, self.all_data))
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def dataloader(data_path, device, batch_size=32, random_shuffle=True):
    return torch.utils.data.DataLoader(CustomDataset(data_path, device),batch_size=batch_size,shuffle=random_shuffle,collate_fn=collate_batch)
