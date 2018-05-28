import os
triple_classification_base_dir = "/home/mousselly/MulitModal_KBC_IKRL/transE/triple_classification_results/"



# Triple files
triples_base_dir = "/home/mousselly/KBC_Datasets/FB/data/" #"/home/mousselly/MulitModal_KBC_IKRL/Data/WN9-IMG/triples/"
relation_file = "/home/mousselly/KBC_Datasets/FB/data/relations.txt"  #"/home/mousselly/MulitModal_KBC_IKRL/Data/WN9-IMG/relations.txt"


# Embeddings files
'''
text_entity_embeddings_file = "/home/mousselly/TF/IKRL_Text_Imagined/embeddings/structure/fast_TransX_100/k2b_unif_l1_100_normalized.pkl"
relation_embeddings_file = "/home/mousselly/TF/IKRL_Text_Imagined/embeddings/structure/fast_TransX_100/k2b_unif_l1_100_normalized.pkl"
'''

text_entity_embeddings_file =   "/home/mousselly/KBC_Datasets/FB/structure/FB_transE_100_norm.pkl"
relation_embeddings_file =  "/home/mousselly/KBC_Datasets/FB/structure/FB_transE_100_norm.pkl"

# weights files

model_name = "TRANSE"
dataset = "FBIMG"
model_results_dir = triple_classification_base_dir+ dataset+"_"+model_name+"/"
os.mkdir(model_results_dir)

# Validation and test triples score files
valid_triple_file = model_results_dir + dataset + "_" + model_name + "_valid.txt"
test_triple_file = model_results_dir + dataset + "_" + model_name + "_test.txt"

valid_triple_score_file = model_results_dir + dataset + "_" + model_name + "_valid_score.txt"
test_triple_score_file = model_results_dir + dataset + "_" + model_name + "_test_score.txt"

triple_classification_results_file = model_results_dir+ "triple_classification_results.txt"

