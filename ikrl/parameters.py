import tensorflow as tf
import os

relation_structural_embeddings_size = 100
entity_structural_embeddings_size = 100
entity_multimodal_embeddings_size = 1128


margin = 4
training_epochs = 1000
batch_size = 100
display_step = 1
activation_function = None
initial_learning_rate = 0.001
mapping_size = 100


all_triples_file =   "/home/mousselly/KBC_Datasets/FB/data/all.txt" #"
train_triples_file = "/home/mousselly/KBC_Datasets/FB/data/train.txt" #
test_triples_file =  "/home/mousselly/KBC_Datasets/FB/data/test.txt"
valid_triples_file =  "/home/mousselly/KBC_Datasets/FB/data/valid.txt"

entity_structural_embeddings_file =   "/home/mousselly/KBC_Datasets/FB/structure/FB_transE_100_norm.pkl"
entity_multimodal_embeddings_file =    "/home/mousselly/KBC_Datasets/FB/multimodal/fb_vgg128_avg_fb_txt_normalized.pkl"
relation_structural_embeddings_file =  "/home/mousselly/KBC_Datasets/FB/structure/FB_transE_100_norm.pkl"


model_id = "IKRL_FB_MM_m4_lin_02"

checkpoint_best_valid_dir = "weights/best_"+model_id+"/"
checkpoint_current_dir ="weights/current_"+model_id+"/"
results_dir = "results/results_"+model_id+"/"

if not os.path.exists(checkpoint_best_valid_dir):
    os.mkdir(checkpoint_best_valid_dir)

if not os.path.exists(checkpoint_current_dir):
    os.mkdir(checkpoint_current_dir)


if not os.path.exists(results_dir):
    os.mkdir(results_dir)


model_current_weights_file = checkpoint_current_dir + model_id + "_current"
current_model_meta_file = checkpoint_current_dir + model_id + "_current.meta"

model_weights_best_valid_file = checkpoint_best_valid_dir + model_id + "_best_hits"
best_valid_model_meta_file = checkpoint_best_valid_dir + model_id + "_best_hits.meta"


result_file = results_dir+model_id+"_results.txt"
log_file = results_dir+model_id+"_log.txt"

