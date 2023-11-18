import json
from tqdm import tqdm
import random

#Set ReaSCAN data path

def make_train_data(data_path):
    random.seed(1249)
    print("loading training data...")
    data = [json.loads(line) for line in tqdm(open(data_path + 'ReaSCAN-compositional/train.json', 'r'))]
    random.shuffle(data)
    print("training data loaded")
    # [markdown]
    # ### Custom Train Split

    #
    data_112000 = random.sample(data, min(len(data), 112000))

    #
    with open(f'{data_path}/ReaSCAN-compositional/train_112000.json', 'w') as f:
        for line in data_112000:
            f.write(json.dumps(line) + '\n')

    print("training data loaded 112000")

    #
    data_56000 = random.sample(data,  min(len(data), 56000))

    #
    with open(f'{data_path}/ReaSCAN-compositional/train_56000.json', 'w') as f:
        for line in data_56000:
            f.write(json.dumps(line) + '\n')

    print("training data loaded 56000")

def make_dev_data(data_path):

    print("val data loading")

    data_a1 = [json.loads(line) for line in tqdm(open(data_path + 'ReaSCAN-compositional-a1/test.json', 'r'))]
    data_a2 = [json.loads(line) for line in tqdm(open(data_path + 'ReaSCAN-compositional-a2/test.json', 'r'))]
    data_a3 = [json.loads(line) for line in tqdm(open(data_path + 'ReaSCAN-compositional-a3/test.json', 'r'))]
    data_b1 = [json.loads(line) for line in tqdm(open(data_path + 'ReaSCAN-compositional-b1/test.json', 'r'))]
    data_b2 = [json.loads(line) for line in tqdm(open(data_path + 'ReaSCAN-compositional-b2/test.json', 'r'))]
    data_c1 = [json.loads(line) for line in tqdm(open(data_path + 'ReaSCAN-compositional-c1/test.json', 'r'))]
    data_c2 = [json.loads(line) for line in tqdm(open(data_path + 'ReaSCAN-compositional-c2/test.json', 'r'))]

    #
    data_a1_1000 = random.sample(data_a1, min(len(data_a1),1000))
    data_a2_1000 = random.sample(data_a2, min(len(data_a2),1000))
    data_a3_1000 = random.sample(data_a3, min(len(data_a3),1000))
    data_b1_1000 = random.sample(data_b1, min(len(data_b1),1000))
    data_b2_1000 = random.sample(data_b2, min(len(data_b2),1000))
    data_c1_1000 = random.sample(data_c1, min(len(data_c1),1000))
    data_c2_1000 = random.sample(data_c2, min(len(data_c2),1000))

    dev_comp_7000 = data_a1_1000 + data_a2_1000 + data_a3_1000 + data_b1_1000 + data_b2_1000 + data_c1_1000 + data_c2_1000
    random.shuffle(dev_comp_7000)

    print("val data loaded 7000")

    with open(f'{data_path}/ReaSCAN-compositional/dev_comp.json', 'w') as f:
        for line in tqdm(dev_comp_7000):
            f.write(json.dumps(line) + '\n')

    #
    data_a1_500 = random.sample(data_a1, min(len(data_a1),500))
    data_a2_500 = random.sample(data_a2, min(len(data_a2),500))
    data_a3_500 = random.sample(data_a3, min(len(data_a3),500))
    data_b1_500 = random.sample(data_b1, min(len(data_b1),500))
    data_b2_500 = random.sample(data_b2, min(len(data_b2),500))
    data_c1_500 = random.sample(data_c1, min(len(data_c1),500))
    data_c2_500 = random.sample(data_c2, min(len(data_c2),500))

    dev_comp_3500 = data_a1_500 + data_a2_500 + data_a3_500 + data_b1_500 + data_b2_500 + data_c1_500 + data_c2_500
    random.shuffle(dev_comp_3500)
    print("dev_comp_3500 _ loaded")

    #
    with open(f'{data_path}/ReaSCAN-compositional/dev_comp_3500.json', 'w') as f:
        for line in tqdm(dev_comp_3500):
            f.write(json.dumps(line) + '\n')


def make_aux_data(data_path='./data/ReaSCAN-v1.1'):
    make_train_data(data_path)
    make_dev_data(data_path)

def make_google_aux_data(data_path='./data/spatial_relation_splits/'):
    print("################")
    random.seed(1249)
    data_visual = [json.loads(line) for line in open(data_path + 'visual.json', 'r')]
    data_relation = [json.loads(line) for line in open(data_path + 'relation.json', 'r')]
    data_relative_position_1 = [json.loads(line) for line in open(data_path + 'relative_position_1.json', 'r')]
    data_relative_position_2 = [json.loads(line) for line in open(data_path + 'relative_position_2.json', 'r')]
    data_referent = [json.loads(line) for line in open(data_path + 'referent.json', 'r')]
    data_visual_1000 = random.sample(data_visual, 1000)
    data_relation_1000 = random.sample(data_relation, 1000)
    data_relative_position_1_1000 = random.sample(data_relative_position_1, 1000)
    data_relative_position_2_1000 = random.sample(data_relative_position_2, 1000)
    data_referent_1000 = random.sample(data_referent, 1000)

    dev_comp_5000 = data_visual_1000 + data_relation_1000 + data_relative_position_1_1000 + data_relative_position_2_1000 + data_referent_1000
    random.shuffle(dev_comp_5000)
    with open(f'{data_path}/dev_comp.json', 'w') as f:
        for line in dev_comp_5000:
            f.write(json.dumps(line) + '\n')
    data_visual_500 = random.sample(data_visual, 500)
    data_relation_500 = random.sample(data_relation, 500)
    data_relative_position_1_500 = random.sample(data_relative_position_1, 500)
    data_relative_position_2_500 = random.sample(data_relative_position_2, 500)
    data_referent_500 = random.sample(data_referent, 500)

    dev_comp_2500 = data_visual_500 + data_relation_500 + data_relative_position_1_500 + data_relative_position_2_500 + data_referent_500
    random.shuffle(dev_comp_2500)
    with open(f'{data_path}/dev_comp_2500.json', 'w') as f:
        for line in dev_comp_2500:
            f.write(json.dumps(line) + '\n')
def make_gscan_aux_data(data_path='./data/ReaSCAN-v1.1'):
    random.seed(1249)
    data_visual = [json.loads(line) for line in open(data_path + 'gSCAN-compositional_splits/visual.json', 'r')]
    data_visual_easier = [json.loads(line) for line in open(data_path + 'gSCAN-compositional_splits/visual_easier.json', 'r')]
    data_situational_1 = [json.loads(line) for line in open(data_path + 'gSCAN-compositional_splits/situational_1.json', 'r')]
    data_situational_2 = [json.loads(line) for line in open(data_path + 'gSCAN-compositional_splits/situational_2.json', 'r')]
    data_contextual = [json.loads(line) for line in open(data_path + 'gSCAN-compositional_splits/contextual.json', 'r')]
    data_adverb_1 = [json.loads(line) for line in open(data_path + 'gSCAN-compositional_splits/adverb_1.json', 'r')]
    data_adverb_2 = [json.loads(line) for line in open(data_path + 'gSCAN-compositional_splits/adverb_2.json', 'r')]
    data_visual_1000 = random.sample(data_visual, 1000)
    data_visual_easier_1000 = random.sample(data_visual_easier, 1000)
    data_situational_1_1000 = random.sample(data_situational_1, 1000)
    data_situational_2_1000 = random.sample(data_situational_2, 1000)
    data_contextual_1000 = random.sample(data_contextual, 1000)
    data_adverb_1_1000 = random.sample(data_adverb_1, 1000)
    data_adverb_2_1000 = random.sample(data_adverb_2, 1000)

    dev_comp_7000 = data_visual_1000 + data_visual_easier_1000 + data_situational_1_1000 + data_situational_2_1000 + data_contextual_1000 + data_adverb_1_1000 + data_adverb_2_1000
    random.shuffle(dev_comp_7000)
    with open(f'{data_path}/gSCAN-compositional_splits/dev_comp.json', 'w') as f:
        for line in dev_comp_7000:
            f.write(json.dumps(line) + '\n')
    data_visual_500 = random.sample(data_visual, 500)
    data_visual_easier_500 = random.sample(data_visual_easier, 500)
    data_situational_1_500 = random.sample(data_situational_1, 500)
    data_situational_2_500 = random.sample(data_situational_2, 500)
    data_contextual_500 = random.sample(data_contextual, 500)
    data_adverb_1_500 = random.sample(data_adverb_1, 500)
    data_adverb_2_500 = random.sample(data_adverb_2, 500)

    dev_comp_3500 = data_visual_500 + data_visual_easier_500 + data_situational_1_500 + data_situational_2_500 + data_contextual_500 + data_adverb_1_500 + data_adverb_2_500
    random.shuffle(dev_comp_3500)

    with open(f'{data_path}/gSCAN-compositional_splits/dev_comp_3500.json', 'w') as f:
        for line in dev_comp_3500:
            f.write(json.dumps(line) + '\n')