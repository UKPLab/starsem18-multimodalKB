import os
triple_classification_base_dir = "/home/mousselly/MulitModal_KBC_IKRL/gitlab/triple_classification_results/"



# Triple files
triples_base_dir = "/home/mousselly/KBC_Datasets/FB/data/"
relation_file = "/home/mousselly/KBC_Datasets/FB/data/relations.txt"


# Embeddings files

text_entity_embeddings_file =   "/home/mousselly/KBC_Datasets/FB/structure/FB_transE_100_norm.pkl"
image_entity_embeddings_file =   "/home/mousselly/KBC_Datasets/FB/multimodal/fb_vgg128_avg_fb_txt_normalized.pkl"
relation_embeddings_file =  "/home/mousselly/KBC_Datasets/FB/structure/FB_transE_100_norm.pkl"



# weights files
checkpoint_best_valid_dir = "/home/mousselly/MulitModal_KBC_IKRL/gitlab/paper_best/best_FBIMG_HMS_MM128_dropout0_m10_tanh_mapped_1_layer_02/"
best_valid_model_meta_file = checkpoint_best_valid_dir+"FBIMG_HMS_MM128_dropout0_m10_tanh_mapped_1_layer_02_best_hits.meta"

model_name = "HMS-02"
dataset = "FBIMG"
model_results_dir = triple_classification_base_dir+ dataset+"_"+model_name+"/"
os.mkdir(model_results_dir)

# Validation and test triples score files
valid_triple_file = model_results_dir + dataset + "_" + model_name + "_valid.txt"
test_triple_file = model_results_dir + dataset + "_" + model_name + "_test.txt"

valid_triple_score_file = model_results_dir + dataset + "_" + model_name + "_valid_score.txt"
test_triple_score_file = model_results_dir + dataset + "_" + model_name + "_test_score.txt"

triple_classification_results_file = model_results_dir+ "triple_classification_results.txt"

