from random import randint
import codecs
import pickle

# First create a dictionary for each relation with the heads/tails
# Given a triple sample a negative triple by changing the head/tail from the corresponding triple dictionary


def load_binary_file(in_file, py_version=2):
    if py_version == 2:
        with open(in_file, 'rb') as f:
            embeddings = pickle.load(f)
            return embeddings
    else:
        with open(in_file, 'rb') as f:
            u = pickle._Unpickler(f)
            u.encoding = 'latin1'
            p = u.load()
            return p

def load_triples(triple_file):
    triples = []
    text_file = codecs.open(triple_file, "r", "utf-8")
    lines = text_file.readlines()
    for line in lines:
        line_arr = line.rstrip("\r\n").split("\t")
        head = line_arr[0]
        tail = line_arr[1]
        rel = line_arr[2]
        triples.append((head,tail,rel))
    return triples

def load_triples_with_labels(triple_file):
    triples = []
    text_file = codecs.open(triple_file, "r", "utf-8")
    lines = text_file.readlines()
    for line in lines:
        line_arr = line.rstrip("\r\n").split("\t")
        head = line_arr[0]
        tail = line_arr[1]
        rel = line_arr[2]
        label = line_arr[3]
        triples.append((head,tail,rel,label))
    return triples

def create_relation_dicts(triples):
    relation_head_dict = {}
    relation_tail_dict = {}

    for triple in triples:
        head = triple[0]
        tail = triple[1]
        rel = triple[2]

        if rel not in  relation_head_dict:
            relation_head_dict[rel] = [head]
        elif head not in relation_head_dict[rel]:
            relation_head_dict[rel].append(head)

        if rel not in  relation_tail_dict:
            relation_tail_dict[rel] = [tail]
        elif tail not in relation_tail_dict[rel]:
            relation_tail_dict[rel].append(tail)


    return relation_head_dict, relation_tail_dict

def create_negative_triple(triple,relation_head_dict,relation_tail_dict,corrupt_head = True):
    head = triple[0]
    tail = triple[1]
    rel = triple[2]
    if corrupt_head:
        neg_heads = [h for h in relation_head_dict[rel] if h != head]
        if len(neg_heads) > 0 :
            random_index = randint(0, len(neg_heads)-1)
            neg_head = neg_heads[random_index]
            return (neg_head,tail,rel)
        else:
            return None
    else:
        neg_tails = [t for t in relation_tail_dict[rel] if t != tail]
        if  len(neg_tails) > 0:
            random_index = randint(0, len(neg_tails)-1)
            neg_tail = neg_tails[random_index]
            return (head, neg_tail, rel)
        return None


