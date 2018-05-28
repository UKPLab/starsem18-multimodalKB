import tensorflow as tf
import numpy as np
import helper_functions as u
import parameters as p




graph = tf.get_default_graph()



def calculate_triple_score(triple,rel_embd_dict,entity_txt_embd_dict):
    head = triple[0]
    tail = triple[1]
    relation = triple[2]

    head_txt_embedding = [entity_txt_embd_dict[head]]
    rel_embedding =  [rel_embd_dict[relation]]
    tail_txt_embedding =  [entity_txt_embd_dict[tail]]

    h_r_sum = np.sum([head_txt_embedding, rel_embedding], axis=0)
    #print("h_r_sum",h_r_sum)
    score = np.sum(abs(np.subtract(h_r_sum,tail_txt_embedding)))
    #print("score",score)

    return [[score]]



relation_embeddings = u.load_binary_file(p.relation_embeddings_file)
entity_txt_embeddings = u.load_binary_file(p.text_entity_embeddings_file)


def main():

    modes = ["valid","test"]


    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:

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
                score = calculate_triple_score(triple, relation_embeddings, entity_txt_embeddings)
                #print(triple,score[0][0])
                f.write(triple[0] +"\t" + triple[2] + "\t" + triple[1] + "\t" + str(score[0][0]) + "\t" +triple[3] +"\n")
                f.flush()
            f.close()

if __name__ == '__main__':
    main()
