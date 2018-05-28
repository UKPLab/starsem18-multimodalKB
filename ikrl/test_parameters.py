# Specify the parameters for the test

entity_structural_embeddings_file =   "/home/mousselly/KBC_Datasets/FB/structure/FB_transE_100_norm.pkl"
entity_multimodal_embeddings_file =   "/home/mousselly/KBC_Datasets/FB/multimodal/fb_vgg128_avg_fb_txt_normalized.pkl" # "/home/mousselly/KBC_Datasets/FB/image/vgg128/fb_vgg128_avg_normalized.pkl" # "/home/mousselly/KBC_Datasets/FB/image/vgg19/fb_vgg19_avg_normalized.pkl"
relation_structural_embeddings_file =  "/home/mousselly/KBC_Datasets/FB/structure/FB_transE_100_norm.pkl"

all_triples_file =   "/home/mousselly/KBC_Datasets/FB/data/all.txt" #"
test_triples_file =  "/home/mousselly/KBC_Datasets/FB/data/test.txt"

model_id = "IKRL_FB_MM_m4_lin_02"

# where to load the weights for the model
checkpoint_best_valid_dir = "weights/best_"+model_id+"/"
model_weights_best_valid_file = checkpoint_best_valid_dir + model_id + "_best_hits"
best_valid_model_meta_file = checkpoint_best_valid_dir + model_id + "_best_hits.meta"

# Results location
results_dir = "results/results_"+model_id+"/"
result_file = results_dir+model_id+"_results.txt"