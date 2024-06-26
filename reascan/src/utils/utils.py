import torch
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
import numpy as np

PAD_token = 0
SOS_token = 1
EOS_token = 2
CLS_token = 3

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {"PAD": 0, "SOS": 1, "EOS": 2, "CLS": 3}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "CLS"}
        self.n_words = 4  # Count PAD and SOS and EOS and CLS

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1
            
            
def prepareData(lang1, lang2, dataloader, include_target=False,command_lang=None,action_lang=None,exclude_lang2=False):
    if command_lang is None:
        command_lang = Lang(lang1) 
    if action_lang is None:
        action_lang = Lang(lang2)
        
    for data in tqdm(dataloader):
        if exclude_lang2:
            for command in data.input_command:
                for word_1 in command:
                    command_lang.addWord(word_1)
        else:
            for command, target in zip(data.input_command, data.target_sequence):
                for word_1 in command:
                    command_lang.addWord(word_1)
                for word_2 in target:
                    action_lang.addWord(word_2)
                
    if include_target:
        for index in range(1, 37):
            command_lang.addWord('target_'+str(index))

    print("Counting words...")
    print("Counted words:")
    print(command_lang.name, command_lang.n_words)
    print(action_lang.name, action_lang.n_words)
    
    return command_lang, action_lang


def indexesFromSentence(lang, sentence):
    if lang.name == 'action':
        indexes = [SOS_token] + [lang.word2index[word] for word in sentence] + [EOS_token]
    if lang.name == 'command':        
        if sentence[0] == "ROOT" or sentence[0] == SOS_token:
            indexes = [lang.word2index[word] for word in sentence]            
        else:
            indexes = [CLS_token] + [lang.word2index[word] for word in sentence] + [EOS_token]
    return indexes

def adjust_mask(config,mask):
    if not config.self_att_mask_upper:
        mask = np.tril(mask, k= 0 if config.self_att_mask_diag else -1)
    if not config.self_att_mask_lower:
        mask = np.triu(mask, k= 0 if config.self_att_mask_diag else 1)
    return mask

def tensorFromBatch(lang, batch, device, max_length=40, no_pad=False, target_data=None, include_target=False,input_mask=None,infer_max_length=True,config=None):
    batch_text = []
    batch_text_no_pad = []
    batch_text_mask = []
    if include_target:
        target_data = get_loc_info(target_data)
    if infer_max_length:
        max_length = max(map(len,batch)) + 2
    for i, sentence in enumerate(batch):
        if include_target:
            indexes = indexesFromSentence(lang, sentence+[target_data[i]])
        else:
            indexes = indexesFromSentence(lang, sentence)
            
        batch_text_no_pad.append(torch.tensor(indexes, dtype=torch.long, device=device))
        if input_mask is None:
            mask = [1] * len(indexes)
        else:
            mask = adjust_mask(mask=np.array(input_mask[i]),config=config)
        if len(indexes) < max_length:
            padding = [PAD_token] * (max_length - len(indexes))
            indexes = indexes + padding
            if input_mask is None:
                mask += padding
            else:
                new_mask = np.zeros((len(indexes),len(indexes)))
                new_mask[:mask.shape[0],:mask.shape[1]] = mask
                mask = new_mask
            
        batch_text.append(indexes)
        batch_text_mask.append(mask)
        
    batch_text = torch.tensor(batch_text, dtype=torch.long, device=device)
    batch_text_mask = torch.tensor(batch_text_mask, dtype=torch.long, device=device)
    
    if no_pad:
        return batch_text_no_pad
    else:
        return batch_text, batch_text_mask


def tensorFromBatchMLM(lang, batch, device, max_length=40, no_pad=False, target_data=None, include_target=False):
    
    def prob_mask_like(t, prob):
        return torch.zeros((len(t),)).float().uniform_(0, 1) < prob

    batch_text = []
    batch_text_no_pad = []
    batch_text_mask = []
    batch_masked_text = []
    
    if include_target:
        target_data = get_loc_info(target_data)
    max_length = max(map(len,batch)) + 2

    for i, sentence in enumerate(batch):
        if include_target:
            indexes = indexesFromSentence(lang, sentence+[target_data[i]])
        else:
            indexes = indexesFromSentence(lang, sentence)
            masked_indexes = indexes[:]
            masks = prob_mask_like(masked_indexes,0.15)
            for token_idx, mask_val in enumerate(masks):
                if mask_val:
                    masked_indexes[token_idx] = lang.word2index["MASK"]
        batch_text_no_pad.append(torch.tensor(indexes, dtype=torch.long, device=device))
        mask = [1] * len(indexes)
        if len(indexes) < max_length:
            padding = [PAD_token] * (max_length - len(indexes))
            indexes = indexes + padding
            masked_indexes = masked_indexes + padding
            mask += padding
        batch_text.append(indexes)
        batch_text_mask.append(mask)
        batch_masked_text.append(masked_indexes)
    batch_text = torch.tensor(batch_text, dtype=torch.long, device=device)
    batch_text_mask = torch.tensor(batch_text_mask, dtype=torch.long, device=device)
    batch_masked_text = torch.tensor(batch_masked_text, dtype=torch.long, device=device)
    
    if no_pad:
        return batch_text_no_pad
    else:
        return batch_text, batch_text_mask, batch_masked_text


def tensorFromBatch_v2(lang, batch, device, max_length=40, no_pad=False):
    batch_text = []
    batch_text_no_pad = []
    batch_text_mask = []
    for sentence in batch:
        indexes = torch.tensor(indexesFromSentence(lang, sentence), dtype=torch.long, device=device)
        batch_text_no_pad.append(torch.tensor(indexes, dtype=torch.long, device=device))
        mask = torch.tensor([1] * len(indexes), dtype=torch.long, device=device)
        
        batch_text.append(indexes)
        batch_text_mask.append(mask)
    
    batch_text = pad_sequence(batch_text, batch_first=True, padding_value=0)
    batch_text_mask = pad_sequence(batch_text_mask, batch_first=True, padding_value=0)
    
    if no_pad:
        return batch_text_no_pad
    else:
        return batch_text, batch_text_mask
    
    
def worldFromBatch(batch, v_feature_size, device):
    batch_world = []
    batch_world_mask = []
    batch_world_loc = []
    for world in batch:
        world = [[1] * v_feature_size] + world.reshape((36, v_feature_size)).tolist()    #Ones initialization for CLS token
        world_mask = [1] * 37          #37th for CLS token
        batch_world.append(world)
        batch_world_mask.append(world_mask)
        world_loc = []
        world_loc.append([-1, -1])     #Adding for CLS token
        for row in range(0, 6):
            for col in range(0, 6):
                world_loc.append([row, col])
        batch_world_loc.append(world_loc)
        
    batch_world = torch.tensor(batch_world, dtype=torch.float, device=device)
    batch_world_mask = torch.tensor(batch_world_mask, dtype=torch.long, device=device)
    batch_world_loc = torch.tensor(batch_world_loc, dtype=torch.float, device=device)
    
    return batch_world, batch_world_mask, batch_world_loc


def worldFromBatch_4_conv(batch, v_feature_size, device):
    batch_world = []
    batch_world_mask = []
    for world in batch:
        world = world.reshape((6, 6, v_feature_size)).tolist()
        world_mask = [1] * 36
        batch_world.append(world)
        batch_world_mask.append(world_mask)
        
    batch_world = torch.tensor(batch_world, dtype=torch.float, device=device)
    batch_world_mask = torch.tensor(batch_world_mask, dtype=torch.long, device=device)
    
    return batch_world, batch_world_mask


def locFromBatch(batch, device):
    batch_loc = []
    target_loc = batch.reshape(-1, 36)
    
    for loc in target_loc:
        batch_loc.append(int(torch.argmax(loc, dim=0)))
    
    batch_loc = torch.tensor(batch_loc, dtype=torch.long, device=device)
    
    return batch_loc

def locFromBatchFullShape(batch, device):
    # batch_loc = []
    target_loc = batch.reshape(-1, 36)
    
    # for loc in target_loc:
    #     batch_loc.append(int(torch.argmax(loc, dim=0)))
    
    batch_loc = torch.tensor(target_loc, dtype=torch.float32, device=device)
    
    return batch_loc


def get_loc_info(batch):
    batch_loc = []
    target_loc = batch.reshape(-1, 36)
    
    for loc in target_loc:
        batch_loc.append('target_'+str((int(torch.argmax(loc, dim=0))+1)))
        
    return batch_loc