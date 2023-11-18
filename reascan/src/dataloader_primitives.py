import torch
import json
from torch.utils.data import DataLoader
import numpy as np
class MyData:
    def __init__(self,input_command,target_location,situation,example_index,mask) -> None:
        self.input_command = input_command
        self.target_location = target_location
        self.target_sequence = input_command
        self.situation = situation
        self.example_index = example_index
        self.mask = mask

    def __str__(self) -> str:
        return str({
            "input_command":self.input_command,
            "target_sequence":self.target_sequence,
            "target_location":self.target_location,
            "example_index":self.example_index,
            "mask":self.mask,
            "situation":self.situation
        })
def collate_batch(batch):
    data = {
        "input_command":[],
        "target_location":[],
        "example_index": [],
        "situation":[],
        "mask":[],
        }
    for item in batch:
        data["input_command"].append(item["input_command"])
        data["target_location"].append(item["target_location"])
        data["situation"].append(item["situation"])
        data["example_index"].append(item["example_index"])
        if item.get("mask"):
            data["mask"].append(torch.tensor(np.array(item["mask"])))
        else:
            data["mask"] = None
    data["input_command"] = data["input_command"]
    data["situation"] = torch.stack(data["situation"])
    data["target_location"] = torch.stack(data["target_location"])
    data["mask"] = data["mask"]

    data["example_index"] = data["example_index"]
    return MyData(data["input_command"],data["target_location"],data["situation"],data["example_index"],data["mask"])

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
        self.data = []
        for x in self.all_data:
            if len(x["input_command"]) <= limit:
                self.data.append(x)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def dataloader_primitives(data_path, device, batch_size=32, random_shuffle=True):
    return DataLoader(CustomDataset(data_path, device),batch_size=batch_size,shuffle=random_shuffle,collate_fn=collate_batch)

# import torch
# import torchtext as tt
# from torchtext.legacy import data
# import re
# def dataloader(data_path, device, batch_size=32, random_shuffle=True):

#     TARGET_LOCATION_FIELD = data.RawField(postprocessing=lambda x: torch.DoubleTensor(x).to(device))
#     SITUATION_FIELD = data.RawField(postprocessing=lambda x: torch.DoubleTensor(x).to(device))
#     # def fn(commands):
#     #     new_commands = []
#     #     for command in commands:            
#     #         # new_command = ",".join(command).replace("),","").replace("(,","").replace("(","").replace(")","")
#     #         # new_command = command.replace("),","").replace("(,","").replace("(","").replace(")","")
#     #         # new_command = re.sub("[A-Z]+","",new_command)
#     #         # new_command = re.sub(",+",",",new_command)
#     #         # new_command = re.sub(",$","",new_command)
#     #         # new_command = new_command.strip(",")
#     #         # new_command = [x for x in command if x not in ["(", ")"] and not re.match("[A-Z]+",x) ]
#     #         new_command = [x for x in command if x not in ["(", ")"]]
#     #         new_commands.append(new_command)            
#     #     return new_commands
#     COMMAND_FIELD = data.RawField()
#     TARGET_ACTION_FIELD = data.RawField()
    
#     dataset = data.TabularDataset(path=data_path, format="json", 
#                                   fields={'target_location': ('target_location', TARGET_LOCATION_FIELD), 
#                                           'situation': ('situation', SITUATION_FIELD), 
#                                           'input_command': ('input_command', COMMAND_FIELD), 
#                                           'target_sequence': ('target_sequence', TARGET_ACTION_FIELD)})
        
#     iterator = data.Iterator(dataset, batch_size=batch_size, 
#                              device=device,shuffle=random_shuffle)
        
#     return iterator