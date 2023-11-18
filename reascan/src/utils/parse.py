import stanza
import numpy as np
from collections import OrderedDict


class ArgNode(object):
    def __init__(self, name):
        self.name = name
        self.children = []
        self.parent = None

    @property
    def narg(self):
        return len(self.children)
    
    def __str__(self):
        if self.narg > 0:
            children_str = ','.join([c.__str__() for c in self.children])
            return self.name + ',[,' + children_str + ',]'
        else:
            return self.name


class DKStanfordDependencyParser:
    
    def __init__(self):
        self.model = stanza.Pipeline(lang='en', processors='tokenize,pos,lemma,depparse')
        self.cache = {}
    
    def parse(self, text):
        text = text.replace(",", " ")
        doc = self.model(text)
        sub_tree = {}
        words = doc.sentences[0].words[:]
        counter = 0
        while len(words) > 0:
            word = words.pop(0)
            node = ArgNode(word.text)
            if word.head > 0:
                if word.head in sub_tree:
                    node.parent = sub_tree[word.head]
                    sub_tree[word.head].children.append(node)
                else:
                    words.append(word)
                    continue
            sub_tree[word.id] = node
            if counter == 200: 
                raise Exception(f"over the loop limit {200}")
            counter += 1
        return  (sub_tree[1])
    
    def get_parse_tree_masking(self,text):
        text_joined = " ".join(text)
        if text_joined in self.cache:
            return self.cache[text_joined]
        
        tokens = ["ROOT"] + text_joined.split()
        text_lenght = len(tokens)
        # masks = np.zeros((text_lenght,text_lenght))
        masks = np.eye(text_lenght)
        
        doc = self.model(text_joined)
        
        for dependency_parse_items in doc.sentences[0].dependencies:
            dependency_parse_item = dependency_parse_items[2]
            if dependency_parse_item.head is not None:
                masks[dependency_parse_item.head, dependency_parse_item.id] = 1
                masks[dependency_parse_item.id,dependency_parse_item.head] = 1
                # dependency_parse_item = dependency_parse_items[0]
                # if dependency_parse_item.head is not None:
                #     # masks[dependency_parse_item.id,dependency_parse_item.head] = 1
                #     masks[dependency_parse_item.head, dependency_parse_item.id] = 1  
        
        self.cache[text_joined] = tuple([tokens, masks])
        return tokens, masks



class DKConsituencyParser:
    
    def __init__(self):
        self.model = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
        self.cache = {}
        self.parse_cache = {}
    
    def get_full_parse(self,text):
        if "," in text:
            text = text.replace(",", " ")
        if text in self.cache:
            return self.cache[text]

        doc = self.model(text)
        parse_result = str(doc.sentences[0].constituency).replace(")"," )").replace("  "," ").split()
        stack = []
        for index,token in enumerate(parse_result):
            if "(" in token:
                stack.append(token)
            if token == ")":
                last_token = stack.pop(-1)
                parse_result[index] = last_token[1:]+token
        self.cache[text] = parse_result
        return parse_result

    def parse(self, text):
        if "," in text:
            text = text.replace(",", " ")
        if text in self.parse_cache:
            return self.parse_cache[text]
        doc = self.model(text)
        # return list(filter(lambda x:not x.isupper(),str(doc.sentences[0].constituency).replace("("," ( ").replace(")"," ) ").split()))
        self.parse_cache[text] = cleanup(doc.sentences[0].constituency)

        return self.parse_cache[text]

    def get_parse_tree_masking(self,text):
        text_joined = " ".join(text)
        if text_joined in self.cache:
            return self.cache[text_joined]
        
        doc = self.model(text_joined)
        new_text = str(doc.sentences[0].constituency).replace("(","").replace(")","")
        
        tokens = new_text.split()
        text_lenght = len(tokens)

        relations = OrderedDict()
        counter = {token:0 for token  in tokens}
        positions = {token:[] for token  in tokens}

        for token_idx, token in enumerate(tokens):
            positions[token].append(token_idx)

        def get_node_name(node):
            return f"{node.label}_{counter[node.label]}"
        
        def get_index(node_label):
            token, counter = node_label.split("_")
            counter = int(counter)
            return positions[token][counter-1]
        
        def extract_relations(node):
            node_label = get_node_name(node)
            if len(node.children):
                
                for child in node.children:
                    counter[child.label] += 1
                    relations[node_label].append(get_node_name(child))
                    relations[get_node_name(child)] = [node_label]
                    extract_relations(child)

        counter[doc.sentences[0].constituency.label] += 1
        relations[get_node_name(doc.sentences[0].constituency)] = []

        extract_relations(doc.sentences[0].constituency)
        masks = []
        for key, values in relations.items():
            mask = np.zeros((1,text_lenght))

            for value in values:
                index = get_index(value)    
                mask[0][index] = 1
            masks.append(mask)
            
        masks = np.concatenate(masks)
        masks = np.eye(masks.shape[0]) + masks

        self.cache[text_joined] = tokens, masks
        return tokens, masks
    

        
def cleanup(text):
    if not text.label.isupper():
        result = [text.label]
    else:
        result = []
    if len(text.children) == 1:
        return cleanup(text.children[0])
    elif len(text.children) > 1:
        result = result + ["("]
        for child in text.children:
            result += cleanup(child) 
        result += [")"]
    return result
        

if __name__ == '__main__':
    # texts = [
    #     'push the small object that is in the same color as a small circle and in the same column as the big circle while zigzagging',
    #     'push the small object that is in the same color as a small circle that is in the same column as the big circle while zigzagging',]
    # texts = ['push the small object that is in the same color as a small circle and in the same column as the big circle while zigzagging',
            # 'walk to a green big square while spinning',
            #  'walk to a red circle while spinning',
            #  'walk to a circle while zigzagging',
            #  'walk to a red big circle cautiously',
            #  'walk to a big circle cautiously',
            #  'walk to a circle cautiously',
            #  'walk to a red big circle',
            #  'walk to a red circle',
            #  'walk to a circle',
            #  'push a red big circle while spinning',
            #  'push a big circle while spinning',
            #  'push a circle while spinning',
            #  'push a square while spinning',
            #  'push a red big circle cautiously',
            #  'push a big circle cautiously',
            #  'push a circle cautiously',
            #  'push a green big square',
            #  'push a big square',
            #  'push a square',
            #  ]
    # c_parser = ConstituencyParser()
    # sd_parser = StanfordDependencyParser()
    # dk_parser = DKStanfordDependencyParser()
    # res = dk_parser.parse(text)
    # print(res)
    # print(res)
    # for text in texts:
    #     print(text)
    #     # print(' C: ', c_parser.parse(text))
    #     # print(' D: ', d_parser.parse(text))
    #     # print(' S: ', sd_parser.parse(text))
    #     print(' DK: ', dk_parser.parse(text))
    #     print("##############################################")
    # from allennlp.predictors.predictor import Predictor
    # predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/elmo-constituency-parser-2020.02.10.tar.gz")
    # res = predictor.predict(
    #     sentence=text
    # )
    # print(res["trees"])


        
        # text = text.replace("("," ( ").replace(")"," ) ")
        # print(text)
    # import stanza

    text = "walk,to,the,small,yellow,object,that,is,inside,of,a,small,green,box,that,is,in,the,same,row,as,a,small,green,circle,while,spinning"
    text = text.replace(",", " ")
    c_parser = DKConsituencyParser()
    d_parser = DKStanfordDependencyParser()
    print(c_parser.parse(text).split(","))
    print(d_parser.parse(text).split(","))
    text = "walk,to,the,small,yellow,object,that,is,,inside,of,a,small,green,box,and,in,the,same,row,as,a,small,green,circle,while,spinning"
    text = text.replace(",", " ")
    print(c_parser.parse(text).split(","))
    print(d_parser.parse(text).split(","))

    # doc = nlp(text)
    # print("".join(cleanup(doc.sentences[0].constituency)))
    # text = "walk,to,the,small,yellow,object,that,is,inside,of,a,small,green,box,that,is,in,the,same,row,as,a,small,green,circle,while,spinning"
    # text = text.replace(",", " ")
    # doc = nlp(text)
    # print("".join(cleanup(doc.sentences[0].constituency)))
