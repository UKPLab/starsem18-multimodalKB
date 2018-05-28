import tensorflow as tf
import numpy as np
import helper_functions as u
import parameters as p




graph = tf.get_default_graph()



def calculate_triple_score(triple,rel_embd_dict,entity_txt_embd_dict,entity_img_embd_dict):
    head = triple[0]
    tail = triple[1]
    relation = triple[2]

    head_txt_embedding = [entity_txt_embd_dict[head]]
    head_image_embedding = [entity_img_embd_dict[head]]
    rel_embedding =  [rel_embd_dict[relation]]
    tail_txt_embedding =  [entity_txt_embd_dict[tail]]
    tail_img_embedding =  [entity_img_embd_dict[tail]]

    r_input = graph.get_tensor_by_name("input/r_input:0")
    h_pos_txt_input = graph.get_tensor_by_name("input/h_pos_txt_input:0")
    t_pos_txt_input = graph.get_tensor_by_name("input/t_pos_txt_input:0")

    h_pos_img_input = graph.get_tensor_by_name("input/h_pos_img_input:0")
    t_pos_img_input = graph.get_tensor_by_name("input/t_pos_img_input:0")



    h_r_t_pos = graph.get_tensor_by_name("cosine/pos_energy:0")
   # keep_prob = graph.get_tensor_by_name("input/keep_prob:0")

    score = h_r_t_pos.eval(feed_dict={r_input: rel_embedding,
                                            h_pos_txt_input: head_txt_embedding,
                                            t_pos_txt_input: tail_txt_embedding,
                                            h_pos_img_input: head_image_embedding,
                                            t_pos_img_input: tail_img_embedding,
                                           # keep_prob : 1
                                             })



    return [score]


relation_embeddings = u.load_binary_file(p.relation_embeddings_file)
entity_txt_embeddings = u.load_binary_file(p.text_entity_embeddings_file)
entity_embeddings_img = u.load_binary_file(p.image_entity_embeddings_file)



def main():

    modes = ["valid","test"]


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        saver = tf.train.import_meta_graph(p.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(p.checkpoint_best_valid_dir))
        print("model loaded", p.checkpoint_best_valid_dir)

        for mode in modes:
            print("Scoring " + mode + " triples")

            if mode == "valid":
                test_triples_file = p.valid_triple_file
                result_file = p.valid_triple_score_file
            else:
                test_triples_file = p.test_triple_file
                result_file = p.test_triple_score_file

            all_test_triples = u.load_triples_with_labels(test_triples_file)

            f = open(result_file,"w")

            for triple in all_test_triples:
                score = calculate_triple_score(triple, relation_embeddings, entity_txt_embeddings, entity_embeddings_img)
                #print(triple,score[0][0])
                f.write(triple[0] +"\t" + triple[2] + "\t" + triple[1] + "\t" + str(score[0][0]) + "\t" +triple[3] +"\n")
                f.flush()
            f.close()

if __name__ == '__main__':
    main()
