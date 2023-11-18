import json
import os
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from .vocabulary import *
from .object_vocabulary import *
from .world import *
from .grammer import *
from .simulator import *
from .relation_graph import *

os.environ['QT_QPA_PLATFORM']='offscreen'

class WorldPlotter():
    intransitive_verbs = ["walk"]
    transitive_verbs = ["push", "pull"]
    adverbs = ["while zigzagging", "while spinning", "cautiously", "hesitantly"]
    nouns = ["circle", "cylinder", "square", "box"]
    color_adjectives = ["red", "blue", "green", "yellow"]
    size_adjectives = ["big", "small"]
    relative_pronouns = ["that is"]
    relation_clauses = ["in the same row as", 
                        "in the same column as", 
                        "in the same color as", 
                        "in the same shape as", 
                        "in the same size as",
                        "inside of"]
    vocabulary = Vocabulary.initialize(intransitive_verbs=intransitive_verbs,
                                    transitive_verbs=transitive_verbs, adverbs=adverbs, nouns=nouns,
                                    color_adjectives=color_adjectives,
                                    size_adjectives=size_adjectives, 
                                    relative_pronouns=relative_pronouns, 
                                    relation_clauses=relation_clauses)

    # test out the object vocab
    object_vocabulary = ObjectVocabulary(shapes=vocabulary.get_semantic_shapes(),
                                        colors=vocabulary.get_semantic_colors(),
                                        min_size=1, max_size=4)
    def __init__(self,dataset_file):
 
        self.data = json.load(open(dataset_file))
    
    def show(self,example_id,key="test",commands=[],targets=[],show_target=True,show_size=False,wrong_target=None):
        example = self.data["examples"][key][example_id]
        command = " ".join(example["command"].split(","))
        action_sequence = example["target_commands"]
        grid_size = 6
        fig = plt.figure()
        ax = fig.add_subplot(111)


        situation = example["situation"]
        world = World(grid_size=grid_size, colors=self.vocabulary.get_semantic_colors(),
                    object_vocabulary=self.object_vocabulary,
                    shapes=self.vocabulary.get_semantic_shapes(),
                    save_directory="./tmp/")
        world.clear_situation()
        
        for obj_idx, obj in situation["placed_objects"].items():
            world.place_object(
                Object(size=int(obj["object"]["size"]), color=obj["object"]["color"], shape=obj["object"]["shape"]), 
                position=Position(row=int(obj["position"]["row"]), column=int(obj["position"]["column"]))
            )
            if show_size:
                if obj["object"]["shape"].lower() == "box":
                    ax.text(int(obj["position"]["column"])*60+55, (int(obj["position"]["row"])+1)*60-5, str(obj["object"]["size"]), fontsize=11, color="black")
                else:
                    ax.text(int(obj["position"]["column"])*60+10, (int(obj["position"]["row"])+1)*60-50, str(obj["object"]["size"]), fontsize=11, color="black")

        world.place_agent_at(
            Position(
                row=int(situation["agent_position"]["row"]), 
                column=int(situation["agent_position"]["column"])
        ))
        # world.render_simple()
        world_array = world.render_simple(array_only=True)
        ax.imshow(world_array)

        for command in commands:
            world.execute_command(command)
            col,row = world.agent_pos
            # ax.add_patch(
            #     patches.Circle(
            #         xy=(30 + col*60, 30 +  row*60),  # point of origin.
            #         radius=5,
            #         linewidth=3,
            #         color='brown',
            #         fill=False,
            #         alpha=1,
            #         hatch="/",
            #         linestyle="--"
            #     )
            # )
            ax.add_patch(
                patches.Polygon(
                    [
                        (30 + col*60 - 12, 30 +  row*60 + 10),
                        (30 + col*60 + 12, 30 +  row*60),
                        (30 + col*60 - 12, 30 +  row*60 - 10),
                    ],  # point of origin.
                    linewidth=3,
                    color='brown',
                    fill=False,
                    alpha=1,
                    hatch="/",
                    linestyle="--"
                )
            )
        from itertools import product
        if len(targets) == grid_size and len(targets[0]) == grid_size:
            for row, col in product(range(0,grid_size),range(0,grid_size)):
                if targets[row][col]:
                    ax.add_patch(
                    patches.Rectangle(
                        xy=(col*60, row*60),  # point of origin.
                        width=60,
                        height=60,
                        linewidth=3,
                        color='purple',
                        fill=False,
                        alpha=1,
                        hatch="/",
                        linestyle="--"
                    )
                )
        if show_target:
            row = int(example['situation']['target_object']['position']['row'])
            col = int(example['situation']['target_object']['position']['column'])
            ax.add_patch(
                patches.Rectangle(
                    xy=(col*60, row*60),  # point of origin.
                    width=60,
                    height=60,
                    linewidth=3,
                    color='brown',
                    fill=False,
                    alpha=1,
                    hatch="/",
                    linestyle="--"
                )
            )
            ax.text(col*60, (row+1)*60+20, "Target", fontsize=11, color="black")
        if wrong_target:
            row,col = wrong_target
            ax.add_patch(
                patches.Rectangle(
                    xy=(col*60, row*60),  # point of origin.
                    width=60,
                    height=60,
                    linewidth=3,
                    color='m',
                    fill=False,
                    alpha=1,
                    hatch="/",
                    linestyle="dashdot"
                )
            )
            ax.text(col*60 + 20, (row+1)*60+20, "Wrong Target", fontsize=11, color="black")
        plt.xticks([])
        plt.yticks([])
        # plt.plot()
        plt.show()

        print(command)
