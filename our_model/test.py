import operator
import os

import numpy as np
import tensorflow as tf

import test_parameters as param
import util as u

graph = tf.get_default_graph()

batch_size = 2000

def predict_best_tail(test_triple, full_triple_list, full_entity_list, entity_embeddings_txt,
                      entity_embeddings_img,
                      full_relation_embeddings):
    results = {}
    gt_head = test_triple[0]
    gt_head_embeddings_txt = entity_embeddings_txt[gt_head]
    gt_head_embeddings_img = entity_embeddings_img[gt_head]

    gt_rel = test_triple[2]
    gt_relation_embeddings = full_relation_embeddings[gt_rel]
    gt_tail_org = test_triple[1]
    gt_tail = u.get_correct_tails(gt_head, gt_rel, full_triple_list)


    total_batches = len(full_entity_list)//batch_size +1

    predictions = []
    for batch_i in range(total_batches):
        start = batch_size * (batch_i)
        end = batch_size * (batch_i + 1)


        tails_embeddings_list_txt = []
        tails_embeddings_list_img = []

        head_embeddings_list_txt = np.tile(gt_head_embeddings_txt,(len(full_entity_list[start:end]),1))
        head_embeddings_list_img = np.tile(gt_head_embeddings_img,(len(full_entity_list[start:end]),1))
        full_relation_embeddings = np.tile(gt_relation_embeddings,(len(full_entity_list[start:end]),1))


        for i in range(len(full_entity_list[start:end])):

            tails_embeddings_list_txt.append(entity_embeddings_txt[full_entity_list[start+i]])
            tails_embeddings_list_img.append(entity_embeddings_img[full_entity_list[start+i]])

        sub_predictions = predict_tail(head_embeddings_list_txt, head_embeddings_list_img, full_relation_embeddings,tails_embeddings_list_txt, tails_embeddings_list_img)
        for p in sub_predictions:
            predictions.append(p)

    predictions = [predictions]
    for i in range(0, len(predictions[0])):
        if  full_entity_list[i] == gt_head  and gt_head not in gt_tail:
            pass
            #results[full_entity_list[i]] = 0
        else:
            results[full_entity_list[i]] = predictions[0][i]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    top_10_predictions = [x[0] for x in sorted_x[:10]]
    sorted_keys = [x[0] for x in sorted_x]
    index_correct_tail_raw = sorted_keys.index(gt_tail_org)

    gt_tail_to_filter = [x for x in gt_tail if x != gt_tail_org]
    # remove the correct tails from the predictions
    for key in gt_tail_to_filter:
        del results[key]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    sorted_keys = [x[0] for x in sorted_x]
    index_tail_head_filter = sorted_keys.index(gt_tail_org)

    return (index_correct_tail_raw + 1), (index_tail_head_filter + 1), top_10_predictions

def predict_tail(head_embedding_txt, head_embedding_img, relation_embedding, tails_embedding_txt, tails_embeddings_img):


    r_input = graph.get_tensor_by_name("input/r_input:0")
    h_pos_txt_input = graph.get_tensor_by_name("input/h_pos_txt_input:0")
    t_pos_txt_input = graph.get_tensor_by_name("input/t_pos_txt_input:0")

    h_pos_img_input = graph.get_tensor_by_name("input/h_pos_img_input:0")
    t_pos_img_input = graph.get_tensor_by_name("input/t_pos_img_input:0")


    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    h_r_t_pos = graph.get_tensor_by_name("cosine/h_r_t_pos:0")


    predictions = h_r_t_pos.eval(feed_dict={r_input: relation_embedding,
                                            h_pos_txt_input: np.asarray(head_embedding_txt),
                                            t_pos_txt_input: np.asarray(tails_embedding_txt),
                                            h_pos_img_input: np.asarray(head_embedding_img),
                                            t_pos_img_input: np.asarray(tails_embeddings_img),
                                            keep_prob: 1.0})
    return predictions



def predict_head(tail_embeddings_list_txt, tail_embeddings_list_img, full_relation_embeddings,
                               heads_embeddings_list_txt, heads_embeddings_list_img):


    r_input = graph.get_tensor_by_name("input/r_input:0")
    h_pos_txt_input = graph.get_tensor_by_name("input/h_pos_txt_input:0")
    t_pos_txt_input = graph.get_tensor_by_name("input/t_pos_txt_input:0")

    h_pos_img_input = graph.get_tensor_by_name("input/h_pos_img_input:0")
    t_pos_img_input = graph.get_tensor_by_name("input/t_pos_img_input:0")


    keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    t_r_h_pos = graph.get_tensor_by_name("cosine/t_r_h_pos:0")

    predictions = t_r_h_pos.eval(feed_dict={r_input: full_relation_embeddings,
                                            h_pos_txt_input: np.asarray(heads_embeddings_list_txt),
                                            t_pos_txt_input: np.asarray(tail_embeddings_list_txt),
                                            h_pos_img_input: np.asarray(heads_embeddings_list_img),
                                            t_pos_img_input: np.asarray(tail_embeddings_list_img),
                                            keep_prob: 1.0})




    return predictions



def predict_best_head(test_triple, full_triple_list, full_entity_list, entity_embeddings_txt,
                      entity_embeddings_img,
                      full_relation_embeddings):

    #triple: head, tail, relation
    results = {}
    gt_tail = test_triple[1] #tail
    gt_tail_embeddings_txt = entity_embeddings_txt[gt_tail] #tail embeddings
    gt_tail_embeddings_img = entity_embeddings_img[gt_tail]

    gt_rel = test_triple[2]
    gt_relation_embeddings = full_relation_embeddings[gt_rel]

    gt_head_org = test_triple[0]
    gt_head = u.get_correct_heads(gt_tail, gt_rel, full_triple_list)



    total_batches = len(full_entity_list) // batch_size + 1

    predictions = []
    for batch_i in range(total_batches):
        start = batch_size * (batch_i)
        end = batch_size * (batch_i + 1)
        heads_embeddings_list_txt = []
        heads_embeddings_list_img = []

        tail_embeddings_list_txt = np.tile(gt_tail_embeddings_txt,(len(full_entity_list[start:end]),1))
        tail_embeddings_list_img = np.tile(gt_tail_embeddings_img,(len(full_entity_list[start:end]),1))
        full_relation_embeddings = np.tile(gt_relation_embeddings,(len(full_entity_list[start:end]),1))


        for i in range(len(full_entity_list[start:end])):


            heads_embeddings_list_txt.append(entity_embeddings_txt[full_entity_list[start+i]])
            heads_embeddings_list_img.append(entity_embeddings_img[full_entity_list[start+i]])

        sub_predictions = predict_head(tail_embeddings_list_txt, tail_embeddings_list_img, full_relation_embeddings,
                                   heads_embeddings_list_txt, heads_embeddings_list_img)

        for p in sub_predictions:
            predictions.append(p)


    predictions = [predictions]

    for i in range(0, len(predictions[0])):
        if full_entity_list[i] == gt_tail  and gt_tail not in gt_head:

        #    #results[full_entity_list[i]] = 0
            pass
        else:
            results[full_entity_list[i]] = predictions[0][i]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    top_10_predictions = [x[0] for x in sorted_x[:10]]
    sorted_keys = [x[0] for x in sorted_x]
    index_correct_head_raw = sorted_keys.index(gt_head_org)

    gt_tail_to_filter = [x for x in gt_head if x != gt_head_org]
    # remove the correct tails from the predictions
    for key in gt_tail_to_filter:
        del results[key]

    sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=False)
    sorted_keys = [x[0] for x in sorted_x]
    index_head_filter = sorted_keys.index(gt_head_org)

    return (index_correct_head_raw + 1), (index_head_filter + 1), top_10_predictions

############ Testing Part #######################
relation_embeddings = u.load_binary_file(param.relation_structural_embeddings_file)
entity_embeddings = u.load_binary_file(param.entity_structural_embeddings_file)
entity_embeddings_img = u.load_binary_file(param.entity_multimodal_embeddings_file)

entity_list = u.load_entity_list(param.all_triples_file, entity_embeddings)

print("#Entities", len(entity_list))
all_triples = u.load_triples(param.all_triples_file, entity_list)
all_test_triples = u.load_triples(param.test_triples_file, entity_list)
#all_test_triples = all_test_triples[:1000]
print("#Test triples", len(all_test_triples))  # Triple: head, tail, relation


tail_ma_raw = 0
tail_ma_filter = 0
tail_hits_raw = 0
tail_hits_filter = 0
head_ma_raw = 0
head_ma_filter = 0
head_hits_raw = 0
head_hits_filter = 0

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
with tf.Session(config=sess_config) as sess:
        #print("Model restored from file: %s" % param.current_model_meta_file)
        avg_rank_raw = 0.0
        avg_rank_filter = 0.0
        hits_at_10_raw = 0.0
        hits_at_10_filter = 0.0
        lines = []

        #new_saver = tf.train.import_meta_graph(param.model_meta_file)
        # new_saver.restore(sess, param.model_weights_best_file)

        saver = tf.train.import_meta_graph(param.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(param.checkpoint_best_valid_dir))

        graph = tf.get_default_graph()
        #Warning only for relation classification
        #entity_list = u.load_relation_list(param.all_triples_file, entity_embeddings)
        counter = 1
        for triple in all_test_triples:
            rank_raw, rank_filter, top_10 = predict_best_tail(triple, all_triples, entity_list, entity_embeddings,
                                                              entity_embeddings_img,
                                                              relation_embeddings)

            line = triple[0] + "\t" + triple[2] + "\t" + triple[1] + "\t" + str(top_10) + "\t" + str(rank_raw) + "\t" + str(
                    rank_filter) + "\n"

            #print(line)
            lines.append(line)
            print(str(counter) + "/" + str(len(all_test_triples)) + " " + str(rank_raw) + " " + str(rank_filter))
            counter +=1
            avg_rank_raw += rank_raw
            avg_rank_filter += rank_filter
            if rank_raw <= 10:
                hits_at_10_raw += 1
            if rank_filter <= 10:
                hits_at_10_filter += 1

        avg_rank_raw /= len(all_test_triples)
        avg_rank_filter /= len(all_test_triples)
        hits_at_10_raw /= len(all_test_triples)
        hits_at_10_filter /= len(all_test_triples)

        print("MAR Raw", avg_rank_raw, "MAR Filter", avg_rank_filter)
        print("Hits@10 Raw", hits_at_10_raw, "Hits@10 Filter", hits_at_10_filter)

        # Write to a file
        #results_file = param.result_file
        results_file = param.result_file.replace(".txt","tail_prediction.txt")

        if os.path.isfile(results_file):
            results_file = results_file.replace(".txt", "_1.txt")

        print("write the results into", results_file)
        with open(results_file, "w") as f:
            f.write("MAR Raw" + "\t" + str(avg_rank_raw) + "\t" + "MAR Filter" + "\t" + str(avg_rank_filter) + "\n")
            f.write("Hits@10 Raw" + "\t" + str(hits_at_10_raw) + "\t" + "Hits@10 Filter" + "\t" + str(
                hits_at_10_filter) + "\n" + "\n")
            f.write("Head \t Relation \t Gold Tail \t Top Predicted Tails \t Raw Rank \t Filtered Rank\n")
            for l in lines:
                f.write(str(l))

        tail_ma_raw = avg_rank_raw
        tail_ma_filter = avg_rank_filter
        tail_hits_raw = hits_at_10_raw
        tail_hits_filter = hits_at_10_filter


        avg_rank_raw = 0.0
        avg_rank_filter = 0.0
        hits_at_10_raw = 0.0
        hits_at_10_filter = 0.0
        lines = []

        counter = 1
        for triple in all_test_triples:
            rank_raw, rank_filter, top_10 = predict_best_head(triple, all_triples, entity_list, entity_embeddings,
                                                              entity_embeddings_img,
                                                              relation_embeddings)

            line = triple[1] + "\t" + triple[2] + "\t" + triple[0] + "\t" + str(top_10) + "\t" + str(rank_raw) + "\t" + str(
                    rank_filter) + "\n"

            #print(line)
            lines.append(line)
            print(str(counter) + "/" + str(len(all_test_triples)) + " " + str(rank_raw) + " " + str(rank_filter))
            counter += 1
            avg_rank_raw += rank_raw
            avg_rank_filter += rank_filter
            if rank_raw <= 10:
                hits_at_10_raw += 1
            if rank_filter <= 10:
                hits_at_10_filter += 1

        avg_rank_raw /= len(all_test_triples)
        avg_rank_filter /= len(all_test_triples)
        hits_at_10_raw /= len(all_test_triples)
        hits_at_10_filter /= len(all_test_triples)

        print("MAR Raw", avg_rank_raw, "MAR Filter", avg_rank_filter)
        print("Hits@10 Raw", hits_at_10_raw, "Hits@10 Filter", hits_at_10_filter)

        # Write to a file
        results_file = param.result_file.replace(".txt","head_prediction.txt")
        if os.path.isfile(results_file):
            results_file = results_file.replace(".txt", "_1.txt")

        print("write the results into", results_file)
        with open(results_file, "w") as f:
            f.write("MAR Raw" + "\t" + str(avg_rank_raw) + "\t" + "MAR Filter" + "\t" + str(avg_rank_filter) + "\n")
            f.write("Hits@10 Raw" + "\t" + str(hits_at_10_raw) + "\t" + "Hits@10 Filter" + "\t" + str(
                hits_at_10_filter) + "\n" + "\n")
            f.write("Tail \t Relation \t Gold Head \t Top Predicted Heads \t Raw Rank \t Filtered Rank\n")
            for l in lines:
                f.write(str(l))

        head_ma_raw = avg_rank_raw
        head_ma_filter = avg_rank_filter
        head_hits_raw = hits_at_10_raw
        head_hits_filter = hits_at_10_filter

print("+++++++++++++++ Evaluation Summary ++++++++++++++++")
print("MA Raw Tail \t MA Filter Tail \t Hits Raw Tail \t Hits Filter Tail")
print(str(tail_ma_raw)+"\t"+str(tail_ma_filter)+"\t"+str(tail_hits_raw)+"\t"+str(tail_hits_filter))


print("MA Raw Head \t MA Filter Head \t Hits Raw Head \t Hits Filter Head")
print(str(head_ma_raw)+"\t"+str(head_ma_filter)+"\t"+str(head_hits_raw)+"\t"+str(head_hits_filter))


print("MA Raw AVG \t MA Filter AVG \t Hits Raw AVG \t Hits Filter AVG")
avg_ma_raw = (head_ma_raw+tail_ma_raw)/2
avg_ma_filter = (head_ma_filter+tail_ma_filter)/2
avg_hits_raw = (head_hits_raw+tail_hits_raw)/2
avg_hits_filter = (head_hits_filter+tail_hits_filter)/2

print(str(avg_ma_raw)+"\t"+str(avg_ma_filter)+"\t"+str(avg_hits_raw)+"\t"+str(avg_hits_filter))