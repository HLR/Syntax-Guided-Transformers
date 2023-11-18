import json
import argparse
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Union
from itertools import product
import random
from functools import reduce
random.seed(0)
from parse import DKConsituencyParser, DKStanfordDependencyParser

constituency_text_parser = DKConsituencyParser()
dependency_text_parser = DKStanfordDependencyParser()
COLOR_ADJECTIVES = {"green", "red", "blue", "yellow"}

def get_target_loc_vector(target_dict, grid_size):
    
    target_pos = target_dict['position']
    row = int(target_pos['row'])
    col = int(target_pos['column'])
    target_loc_vector = [[0] * 6 for i in range(grid_size)]
    target_loc_vector[row][col] = 1
    
    return np.array(target_loc_vector,dtype=np.bool_)

def get_target_loc_vectors(target_positions, grid_size):
    loc_vector = [[0] * 6 for i in range(grid_size)]
    for row,col in target_positions:
        loc_vector[row][col] = 1
    return np.array(loc_vector,dtype=np.bool_)


def parse_sparse_situation_gscan(situation_representation: dict, grid_size: int) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _   _       _       _    _ _ _ _    _   _ _ _ _]
     1 2 3 4 square cylinder circle y g r b  agent E S W N
     _______  _________________________ _______ ______ _______
       size             shape            color  agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    num_object_attributes = len([int(bit) for bit in situation_representation["target_object"]["vector"]])
    # Object representation + agent bit + agent direction bits (see docstring).
    num_grid_channels = num_object_attributes + 1 + 4

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)

    # Place the agent.
    agent_row = int(situation_representation["agent_position"]["row"])
    agent_column = int(situation_representation["agent_position"]["column"])
    agent_direction = int(situation_representation["agent_direction"])
    agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
    agent_representation[-5] = 1
    agent_representation[-4 + agent_direction] = 1
    grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    placed_position = set([])
    for placed_object in situation_representation["placed_objects"].values():
        object_vector = np.array([int(bit) for bit in placed_object["vector"]], dtype=np.int32)
        object_row = int(placed_object["position"]["row"])
        object_column = int(placed_object["position"]["column"])
        placed_position.add((object_row, object_column))
        if (object_row, object_column) not in placed_position:
            grid[object_row, object_column, :] = np.concatenate([object_vector, np.zeros([5], dtype=np.int32)])
        else:
            overlay = np.concatenate([object_vector, np.zeros([5], dtype=np.int32)])
            grid[object_row, object_column, :] += overlay # simply add it.
    return grid


def parse_sparse_situation_reascan(situation_representation: dict, grid_size: int, place_agent=True) -> np.ndarray:
    """
    Each grid cell in a situation is fully specified by a vector:
    [_ _ _ _   _       _       _     _  _ _ _ _ _ _ _ _ _ _ _ _    _   _ _ _ _]
     1 2 3 4 circle cylinder square box r b g y 1 2 3 4 r b g y  agent E S W N
     _______  _________________________ _______ _______ _______ ______ _______
       size             shape            color  box_size box_color agent agent dir.
    :param situation_representation: data from dataset.txt at key "situation".
    :param grid_size: int determining row/column number.
    :return: grid to be parsed by computational models.
    """
    num_object_attributes = len([int(bit) for bit in situation_representation["target_object"]["vector"]])
    num_box_attributes = 8
    # Object representation + agent bit + agent direction bits (see docstring).
    num_grid_channels = num_object_attributes + num_box_attributes + 1 + 4

    # Initialize the grid.
    grid = np.zeros([grid_size, grid_size, num_grid_channels], dtype=int)
    if place_agent:
        # Place the agent.
        agent_row = int(situation_representation["agent_position"]["row"])
        agent_column = int(situation_representation["agent_position"]["column"])
        agent_direction = int(situation_representation["agent_direction"])
        agent_representation = np.zeros([num_grid_channels], dtype=np.int32)
        agent_representation[-5] = 1
        agent_representation[-4 + agent_direction] = 1
        grid[agent_row, agent_column, :] = agent_representation

    # Loop over the objects in the world and place them.
    placed_position = set([])
    for placed_object in situation_representation["placed_objects"].values():
        object_vector = np.array([int(bit) for bit in placed_object["vector"]], dtype=np.int32)
        if placed_object["object"]["shape"] == "box":
            box_vec_1 = np.array([int(bit) for bit in placed_object["vector"][0:4]], dtype=np.int32)
            box_vec_2 = np.array([int(bit) for bit in placed_object["vector"][8:12]], dtype=np.int32)
            box_vector = np.concatenate([box_vec_1, box_vec_2])
            object_vector[0:4] = 0
            object_vector[8:12] = 0
        else:
            box_vector = np.zeros([8], dtype=np.int32)
        object_row = int(placed_object["position"]["row"])
        object_column = int(placed_object["position"]["column"])
        placed_position.add((object_row, object_column))
        if (object_row, object_column) not in placed_position:
            grid[object_row, object_column, :] = np.concatenate([object_vector, box_vector, np.zeros([5], dtype=np.int32)])
        else:
            overlay = np.concatenate([object_vector, box_vector, np.zeros([5], dtype=np.int32)])
            grid[object_row, object_column, :] += overlay # simply add it.
    return grid

def exclude_boxes_fn(placed_objects):
    return [obj for obj in placed_objects if obj["object"]["shape"] != "box"]

def match_objects(situation_representation, key:str,exclude_boxes=False):
    all_match_objects = []
    
    if key.isnumeric(): # size related
        for placed_object in situation_representation['placed_objects'].values():
            if placed_object["object"]["size"] == key:
                all_match_objects.append(placed_object)
    else:
        for placed_object in situation_representation['placed_objects'].values():
            object_desc = " ".join([ x[1] for x in list(sorted(placed_object["object"].items()))[:2]])
            if key in object_desc:
                all_match_objects.append(placed_object)
    if exclude_boxes:
        all_match_objects = exclude_boxes_fn(all_match_objects)
    return all_match_objects


def get_object_description(placed_object, situation_representation,split:str):
    while True:
        placed_object_info = placed_object["object"]
        items = list(placed_object_info.items())
        choices = sorted(random.sample(items[:2],k=random.randint(1,2)))
        values = [x[1] for x in choices]
            
        description = " ".join(values)
        matched_objects = match_objects(situation_representation, description)
        non_same_matched_objects = [x for x in matched_objects if x !=placed_object]
        all_are_smaller = len(non_same_matched_objects) > 0 and all([int(placed_object_info["size"]) > int(x["object"]["size"]) for x in non_same_matched_objects])
        all_are_bigger = len(non_same_matched_objects) > 0 and all([int(placed_object_info["size"]) < int(x["object"]["size"]) for x in non_same_matched_objects])
        add_size_info = random.choice([0,1]) and (all_are_smaller or all_are_bigger)
        if len(values) == 1 and values[0] in COLOR_ADJECTIVES:
            description = description + " object"
        if add_size_info:
            if all_are_smaller:
                description = "big " + description 
                matched_objects = [placed_object]
            elif all_are_bigger: 
                description = "small " + description 
                matched_objects = [placed_object]
        if split == "train" and "yellow" in description and "square" in description: # A1 -> No Yellow and Square
            continue
        if split == "train" and "small" in description and "cylinder" in description: # A3 -> No Small and Cylinder
            continue                    
        break
    return description, matched_objects
    
def generate_samples(situation_representation: dict, grid_size: int, split:str):
    samples = set()
    for placed_object in situation_representation['placed_objects'].values():
        ## object match 
        # print(placed_object)
        if split == "train" and placed_object["object"]["shape"] == "square" and placed_object["object"]["color"] == "red":# A2 -> No Red and Square
            continue
        description, matched_objects = get_object_description(placed_object, situation_representation, split)
        # print(description, len(matched_objects))

        same_row_target_loc_vectors = []
        same_col_target_loc_vectors = []
        same_color_target_loc_vectors = []
        same_shape_target_loc_vectors = []
        same_size_target_loc_vectors = []
        inside_of_target_loc_vectors = [] 
        for matched_obj in matched_objects:
            row, col = int(matched_obj["position"]["row"]), int(matched_obj["position"]["column"])
            if matched_obj["object"]["shape"] != "box":
                relation = "in the same row as "
                target_locations = product(range(row,row+1),range(0,grid_size))
                target_loc_vectors = get_target_loc_vectors(target_locations,grid_size)   
                same_row_target_loc_vectors.append(target_loc_vectors)

                relation = "in the same column as "
                target_locations = product(range(0,grid_size),range(col,col+1))
                target_loc_vectors = get_target_loc_vectors(target_locations,grid_size)   
                same_col_target_loc_vectors.append(target_loc_vectors)
            else:
                relation = "inside of "
                target_locations = list(product(range(row,row+int(matched_obj["object"]["size"])),range(col,col+int(matched_obj["object"]["size"]))))
                target_loc_vectors = get_target_loc_vectors(target_locations,grid_size)   
                inside_of_target_loc_vectors.append(target_loc_vectors)
                # print(description, len(matched_objects), "same size", target_loc_vectors)

            relation = "in the same color as "
            target_locations = [(int(x["position"]["row"]),int(x["position"]["column"])) for x in match_objects(situation_representation,matched_obj["object"]["color"])]
            target_loc_vectors = get_target_loc_vectors(target_locations,grid_size)   
            same_color_target_loc_vectors.append(target_loc_vectors)

            relation = "in the same shape as "
            target_locations = [(int(x["position"]["row"]),int(x["position"]["column"])) for x in match_objects(situation_representation,matched_obj["object"]["shape"])]
            target_loc_vectors = get_target_loc_vectors(target_locations,grid_size)   
            same_shape_target_loc_vectors.append(target_loc_vectors)

            relation = "in the same size as "
            target_locations = [(int(x["position"]["row"]),int(x["position"]["column"])) for x in match_objects(situation_representation,matched_obj["object"]["size"])]
            target_loc_vectors = get_target_loc_vectors(target_locations,grid_size)   
            same_size_target_loc_vectors.append(target_loc_vectors)
        if same_row_target_loc_vectors:
            samples.add({
                "pompt" :"in the same row as a " + description,
                "output": np.array(reduce(np.logical_or,same_row_target_loc_vectors))
            }.values())
        if same_col_target_loc_vectors:
            samples.add({
                "pompt" :"in the same column as a " + description,
                "output": np.array(reduce(np.logical_or,same_col_target_loc_vectors))
            }.values())
        if same_color_target_loc_vectors:
            samples.add({
                "pompt" :"in the same color as a " + description,
                "output": np.array(reduce(np.logical_or,same_color_target_loc_vectors))
            }.values())
        if same_shape_target_loc_vectors:
            samples.add({
                "pompt" :"in the same shape as a " + description,
                "output": np.array(reduce(np.logical_or,same_shape_target_loc_vectors))
            }.values())
        if same_size_target_loc_vectors:
            samples.add({
                "pompt" :"in the same size as a " + description,
                "output": np.array(reduce(np.logical_or,same_size_target_loc_vectors))
            }.values())
        if inside_of_target_loc_vectors:
            # print(inside_of_target_loc_vectors)
            samples.add({
                "pompt" : "inside of a " + description,
                "output": np.array(reduce(np.logical_or,inside_of_target_loc_vectors))
            }.values())
    return samples


def data_loader(file_path: str, args) -> Dict[str, Union[List[str], np.ndarray]]:
    with open(file_path, 'r') as infile:
        all_data = json.load(infile)
        grid_size = int(all_data["grid_size"])
        splits = list(all_data["examples"].keys())
        loaded_data = {}
        loaded_data_set = {}
        for split in splits:
            loaded_data[split] = list()
            loaded_data_set[split] = set()
            print(split + ':')
            situations = [x["situation"] for x in all_data["examples"][split]]
            unique_situations_set = set()
            for example_index, situation_dict in tqdm(enumerate(situations),total=len(situations),dynamic_ncols=True):
                if args.dataset in ['gscan', 'google']:
                    situation = parse_sparse_situation_gscan(situation_representation=situation_dict,
                                                   grid_size=grid_size)
                if args.dataset == 'reascan':
                    if args.embedding == 'modified':
                        situation = parse_sparse_situation_reascan(situation_representation=situation_dict,
                                                       grid_size=grid_size,place_agent=False)
                    elif args.embedding == 'default':
                        situation = parse_sparse_situation_gscan(situation_representation=situation_dict,
                                                       grid_size=grid_size)
                hashed_situation = hash(tuple(tuple(tuple(y) for y in x ) for x in situation.tolist()))
                if hashed_situation in unique_situations_set:
                    continue
                unique_situations_set.add(hashed_situation)
                
                samples = generate_samples(situation_dict, grid_size,split=split)
                samples = random.choices(list(samples),k=min(5,len(samples)))
                for prompt, labels in samples:
                    labels = np.array(labels,dtype=np.int32)
                    hashed_data = tuple({
                        "example_index":example_index,
                        "input_command": tuple(prompt.split()),
                        "target_location": tuple( tuple(x) for x in labels.tolist()),
                        "situation": tuple( tuple( tuple(y) for y in x ) for x in situation.tolist())}.items())
                    if hashed_data not in loaded_data_set[split]:
                        loaded_data_set[split].add(hashed_data) 
                        input_command = prompt.split()
                        if args.mode == "paranthesis":
                            input_command = constituency_text_parser.parse(" ".join(input_command))
                            input_command_tree_masking = None
                        elif args.mode == "dependency_mask":
                            input_command, input_command_tree_masking = dependency_text_parser.get_parse_tree_masking(input_command)
                        elif args.mode == "constituency_mask":
                            input_command, input_command_tree_masking = constituency_text_parser.get_parse_tree_masking(input_command)
                        elif args.mode == "full_parsed":
                            input_command = constituency_text_parser.get_full_parse(" ".join(input_command))
                            input_command_tree_masking = None
                        else:
                            input_command, input_command_tree_masking = input_command, None
                        x_data = {
                                "input_command": input_command,
                                "example_index":example_index,
                                "target_location": labels.tolist(),
                                "situation": situation.tolist()
                                }
                        if input_command_tree_masking is not None:
                            x_data["mask"] = np.array(input_command_tree_masking, dtype=np.int32).tolist()
                        loaded_data[split].append(x_data)
            print(len(loaded_data_set[split]))
            print(len(loaded_data[split]))
    return loaded_data

def check_intersection_among_envs():
    train_envs = set()
    with open(file_paths_reascan[0], 'r') as infile:
        all_data = json.load(infile)
        grid_size = int(all_data["grid_size"])
        split_data = set()
        for item in all_data["examples"]["train"]:
            situation = parse_sparse_situation_reascan(situation_representation=item["situation"],
                                                    grid_size=grid_size,place_agent=False)
            situation = tuple(tuple([ tuple(y) for y in x]) for x in situation)
            train_envs.add(situation)
    splits = {}
    for file_path in file_paths_reascan[1:]:
        print('Processing {} ...'.format(file_path.split('/')[3]))
        with open(file_path, 'r') as infile:
            split = file_path.split("/")[-2][-2:]
            all_data = json.load(infile)
            grid_size = int(all_data["grid_size"])
            split_data = set()
            for item in all_data["examples"]["test"]:
                situation = parse_sparse_situation_reascan(situation_representation=item["situation"],
                                                       grid_size=grid_size,place_agent=False)
                situation = tuple(tuple([ tuple(y) for y in x]) for x in situation)
                split_data.add(situation)
            print(split,len(split_data), len(train_envs) ,len(split_data & train_envs))
            # splits[split] = split_data
            # for split, envs in splits.items():
            #     print(split,len(envs), len(train_envs) ,len(envs & train_envs))
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='reascan', choices=['reascan', 'gscan', 'google'], help='Choose dataset for preprocessing')
    parser.add_argument('--mode', type=str, default='normal', choices=['dependency_mask', 'constituency_mask', 'normal', 'paranthesis', 'full_parsed'], help='Choose dataset for preprocessing')
    parser.add_argument('--embedding', type=str, default='modified', choices=['modified', 'default'], help='Which embedding to use')
        
    args = parser.parse_args()
    if args.mode == "dependency_mask":
        SRC_ADD = '../../data-with-dep-mask'
    elif args.mode == "paranthesis":
        SRC_ADD = '../../data-with-paranthesis'
    elif args.mode == "constituency_mask":
        SRC_ADD = '../../data-with-mask'
    elif args.mode == "full_parsed":
        SRC_ADD = '../../data-full-parse'
    elif args.mode == "normal":
        SRC_ADD = '../../data'
    else:
        raise Exception("Invalid data mode: {args.mode}")
    file_paths_reascan = [
                          f'{SRC_ADD}/ReaSCAN-v1.1/ReaSCAN-compositional/data-compositional-splits.txt', 
                        #   f'{SRC_ADD}/ReaSCAN-v1.1/ReaSCAN-compositional-a1/data-compositional-splits.txt',
                        #   f'{SRC_ADD}/ReaSCAN-v1.1/ReaSCAN-compositional-a2/data-compositional-splits.txt',
                        #   f'{SRC_ADD}/ReaSCAN-v1.1/ReaSCAN-compositional-a3/data-compositional-splits.txt',
                        #   f'{SRC_ADD}/ReaSCAN-v1.1/ReaSCAN-compositional-b1/data-compositional-splits.txt',
                        #   f'{SRC_ADD}/ReaSCAN-v1.1/ReaSCAN-compositional-b2/data-compositional-splits.txt',
                        #   f'{SRC_ADD}/ReaSCAN-v1.1/ReaSCAN-compositional-c1/data-compositional-splits.txt',
                        #   f'{SRC_ADD}/ReaSCAN-v1.1/ReaSCAN-compositional-c2/data-compositional-splits.txt',
                        ]

    file_paths_gscan = [f'{SRC_ADD}/ReaSCAN-v1.1/gSCAN-compositional_splits/dataset.txt']

    file_paths_google = [f'{SRC_ADD}/spatial_relation_splits/dataset.txt']

    if args.dataset == 'reascan':
        file_paths = file_paths_reascan
    elif args.dataset == 'gscan':
        file_paths = file_paths_gscan
    elif args.dataset == 'google':
        file_paths = file_paths_google


    for file_path in file_paths:
        print('Processing {} ...'.format(file_path.split('/')[3]))
        data = data_loader(file_path, args)
        for split, dt in tqdm(data.items(),dynamic_ncols=True):
            print('Dumping {} json ...'.format(split))
            print(len(dt))
            if args.dataset == 'reascan':
                if args.embedding == 'modified':
                    with open(file_path.split('data-compositional')[0] + split + '_spatial.json', 'w') as f:
                        for line in tqdm(dt,dynamic_ncols=True):
                            f.write(json.dumps(line) + '\n')
                    print("Saved " + file_path.split('data-compositional')[0] + split + '_spatial.json')
                elif args.embedding == 'default':
                    with open(file_path.split('data-compositional')[0] + split + '_spatial_default_embedding.json', 'w') as f:
                        for line in tqdm(dt,dynamic_ncols=True):
                            f.write(json.dumps(line) + '\n')
            elif args.dataset == 'gscan':
                with open(file_path.split('dataset')[0] + split + '_spatial.json', 'w') as f:
                    for line in tqdm(dt,dynamic_ncols=True):
                        f.write(json.dumps(line) + '\n')
            elif args.dataset == 'google':
                with open(file_path.split('dataset')[0] + split + '_spatial.json', 'w') as f:
                    for line in tqdm(dt,dynamic_ncols=True):
                        f.write(json.dumps(line) + '\n')