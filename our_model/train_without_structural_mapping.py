import os

import numpy as np
import tensorflow as tf

import parameters as param
import util as u

#tf.set_random_seed(1234)
#np.random.seed(7)

logs_path = "log"
# .... Loading the data ....
print("load all triples")
relation_embeddings = u.load_binary_file(param.relation_structural_embeddings_file)
entity_embeddings_txt = u.load_binary_file(param.entity_structural_embeddings_file)
entity_embeddings_img = u.load_binary_file(param.entity_multimodal_embeddings_file)

all_train_test_valid_triples, entity_list = u.load_training_triples(param.all_triples_file)
triples_set = [t[0] + "_" + t[1] + "_" + t[2] for t in all_train_test_valid_triples]
triples_set = set(triples_set)
entity_list_filtered = []
for e in entity_list:
    if e in entity_embeddings_txt:
        entity_list_filtered.append(e)
entity_list = entity_list_filtered
print("#entities", len(entity_list), "#total triples", len(all_train_test_valid_triples))

training_data = u.load_freebase_triple_data_multimodal(param.train_triples_file, entity_embeddings_txt,
                                                       entity_embeddings_img, relation_embeddings)

#training_data= training_data[:1000]
#training_data = training_data[:10000]
print("#training data", len(training_data))

valid_data = u.load_freebase_triple_data_multimodal(param.valid_triples_file, entity_embeddings_txt,
                                                    entity_embeddings_img,relation_embeddings)
#valid_data = valid_data[:len(valid_data)//3]

print("valid_data",len(valid_data))

def max_norm_regulizer(threshold,axes=1,name="max_norm",collection="max_norm"):
    def max_norm(weights):
        clipped = tf.clip_by_norm(weights,clip_norm=threshold,axes=axes)
        clip_weights = tf.assign(weights,clipped,name=name)
        tf.add_to_collection(collection,clip_weights)
        return None
    return max_norm

max_norm_reg = max_norm_regulizer(threshold=1.0)

def my_dense(x, nr_hidden, scope, activation_fn=param.activation_function,reuse=None):
    with tf.variable_scope(scope):
        h = tf.contrib.layers.fully_connected(x, nr_hidden,
                                              activation_fn=activation_fn,
                                              reuse=reuse,
                                              scope=scope#, weights_regularizer= max_norm_reg
                                              )

        return h



# ........... Creating the model
with tf.name_scope('input'):
    r_input = tf.placeholder(dtype=tf.float32, shape=[None, param.relation_structural_embeddings_size],name="r_input")

    h_pos_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="h_pos_txt_input")
    h_neg_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="h_neg_txt_input")

    h_pos_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size],name="h_pos_img_input")
    h_neg_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size], name="h_neg_img_input")

    t_pos_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="t_pos_txt_input")
    t_pos_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size], name="t_pos_img_input")

    t_neg_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size],   name="t_neg_txt_input")
    t_neg_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size], name="t_neg_img_input")

    keep_prob = tf.placeholder(tf.float32, name="keep_prob")

with tf.name_scope('head_relation'):
    # structure
     

    # Multimodal embeddings : image + ling concatented
    h_pos_img_mapped = my_dense(h_pos_img_input, param.mapping_size, activation_fn=param.activation_function, scope="img_proj", reuse=None)
    h_pos_img_mapped = tf.nn.dropout(h_pos_img_mapped, keep_prob)

    h_neg_img_mapped = my_dense(h_neg_img_input, param.mapping_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)
    h_neg_img_mapped = tf.nn.dropout(h_neg_img_mapped, keep_prob)

    # Tail image ....
    t_pos_img_mapped = my_dense(t_pos_img_input, param.mapping_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)
    t_pos_img_mapped = tf.nn.dropout(t_pos_img_mapped, keep_prob)

    t_neg_img_mapped = my_dense(t_neg_img_input, param.mapping_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)
    t_neg_img_mapped = tf.nn.dropout(t_neg_img_mapped, keep_prob)




with tf.name_scope('cosine'):

    # Head model
    energy_ss_pos = tf.reduce_sum(abs(h_pos_txt_input + r_input - t_pos_txt_input), 1, keep_dims=True, name="pos_s_s")
    energy_ss_neg = tf.reduce_sum(abs(h_pos_txt_input + r_input - t_neg_txt_input), 1, keep_dims=True, name="neg_s_s")

    energy_is_pos = tf.reduce_sum(abs(h_pos_img_mapped + r_input - t_pos_txt_input), 1, keep_dims=True, name="pos_i_i")
    energy_is_neg = tf.reduce_sum(abs(h_pos_img_mapped + r_input - t_neg_txt_input), 1, keep_dims=True, name="neg_i_i")

    energy_si_pos = tf.reduce_sum(abs(h_pos_txt_input + r_input - t_pos_img_mapped), 1, keep_dims=True, name="pos_s_i")
    energy_si_neg = tf.reduce_sum(abs(h_pos_txt_input + r_input - t_neg_img_mapped), 1, keep_dims=True, name="neg_s_i")

    energy_ii_pos = tf.reduce_sum(abs(h_pos_img_mapped + r_input - t_pos_img_mapped), 1, keep_dims=True, name="pos_i_i")
    energy_ii_neg = tf.reduce_sum(abs(h_pos_img_mapped + r_input - t_neg_img_mapped), 1, keep_dims=True, name="neg_i_i")

    energy_concat_pos = tf.reduce_sum(abs((h_pos_txt_input + h_pos_img_mapped) + r_input - (t_pos_txt_input + t_pos_img_mapped)), 1, keep_dims=True, name="energy_concat_pos")
    energy_concat_neg = tf.reduce_sum(abs((h_pos_txt_input + h_pos_img_mapped) + r_input - (t_neg_txt_input + t_neg_img_mapped)), 1, keep_dims=True, name="energy_concat_neg")

    h_r_t_pos = tf.reduce_sum([energy_ss_pos, energy_is_pos, energy_si_pos, energy_ii_pos, energy_concat_pos], 0,   name="h_r_t_pos")
    h_r_t_neg = tf.reduce_sum([energy_ss_neg, energy_is_neg, energy_si_neg, energy_ii_neg, energy_concat_neg], 0, name="h_r_t_neg")


    # Tail model

    score_t_t_pos = tf.reduce_sum(abs(t_pos_txt_input - r_input - h_pos_txt_input), 1, keep_dims=True, name="pos_s_s")
    score_t_t_neg = tf.reduce_sum(abs(t_pos_txt_input - r_input - h_neg_txt_input), 1, keep_dims=True, name="neg_s_s")

    score_i_t_pos = tf.reduce_sum(abs(t_pos_img_mapped - r_input - h_pos_txt_input), 1, keep_dims=True, name="pos_i_i")
    score_i_t_neg = tf.reduce_sum(abs(t_pos_img_mapped - r_input - h_neg_txt_input), 1, keep_dims=True, name="neg_i_i")

    score_t_i_pos = tf.reduce_sum(abs(t_pos_txt_input - r_input - h_pos_img_mapped), 1, keep_dims=True, name="pos_s_i")
    score_t_i_neg = tf.reduce_sum(abs(t_pos_txt_input - r_input - h_neg_img_mapped), 1, keep_dims=True, name="neg_s_i")

    score_i_i_pos = tf.reduce_sum(abs(t_pos_img_mapped - r_input - h_pos_img_mapped), 1, keep_dims=True, name="pos_i_i")
    score_i_i_neg = tf.reduce_sum(abs(t_pos_img_mapped - r_input - h_neg_img_mapped), 1, keep_dims=True, name="neg_i_i")

    energy_concat_pos_tail = tf.reduce_sum(abs((t_pos_txt_input + t_pos_img_mapped) - r_input - (h_pos_txt_input + h_pos_img_mapped)), 1, keep_dims=True, name="energy_concat_pos_tail")
    energy_concat_neg_tail = tf.reduce_sum(abs((t_pos_txt_input + t_pos_img_mapped) - r_input - (h_neg_txt_input + h_neg_img_mapped)), 1, keep_dims=True, name="energy_concat_neg_tail")

    t_r_h_pos = tf.reduce_sum([score_t_t_pos, score_i_t_pos, score_t_i_pos, score_i_i_pos,energy_concat_pos_tail], 0, name="t_r_h_pos")
    t_r_h_neg = tf.reduce_sum( [score_t_t_neg, score_i_t_neg, score_t_i_neg, score_i_i_neg,energy_concat_neg_tail], 0,name="t_r_h_neg")


    kbc_loss1 = tf.maximum(0., param.margin - h_r_t_neg + h_r_t_pos)
    kbc_loss2 = tf.maximum(0., param.margin - t_r_h_neg + t_r_h_pos)


    kbc_loss = kbc_loss1 + kbc_loss2

    tf.summary.histogram("loss", kbc_loss)

#epsilon= 0.1
optimizer = tf.train.AdamOptimizer().minimize(kbc_loss)

summary_op = tf.summary.merge_all()

#..... start the training
saver = tf.train.Saver()
log_file = open(param.log_file,"w")

log_file.write("relation_input_size = " + str(param.relation_structural_embeddings_size)+ "\n")
log_file.write("entity_input_size = " + str(param.entity_structural_embeddings_size) + "\n")
log_file.write("nr_neuron_dense_layer1 = " + str(param.nr_neuron_dense_layer_1) +"\n")
log_file.write("nr_neuron_dense_layer2 = " + str(param.nr_neuron_dense_layer_2) +"\n")
log_file.write("dropout_ratio = " + str(param.dropout_ratio) +"\n")
log_file.write("margin = " + str(param.margin) +"\n")
log_file.write("training_epochs = " + str(param.training_epochs) +"\n")
log_file.write("batch_size = " + str(param.batch_size) +"\n")
log_file.write("activation_function = " + str(param.activation_function) +"\n")
log_file.write("initial_learning_rate = " + str(param.initial_learning_rate) +"\n")

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

#np.random.shuffle(valid_data)

h_data_valid_txt, h_data_valid_img, r_data_valid, t_data_valid_txt, \
t_data_valid_img, t_neg_data_valid_txt, t_neg_data_valid_img, h_neg_data_valid_txt, h_neg_data_valid_img = \
    u.get_batch_with_neg_heads_and_neg_tails_multimodal(valid_data,
                                                        triples_set,
                                                        entity_list,
                                                        0, len(valid_data),
                                                        entity_embeddings_txt,
                                                        entity_embeddings_img)
#clip_all_weights = tf.get_collection("max_norm")

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())

    if os.path.isfile(param.best_valid_model_meta_file):
        print("restore the weights",param.checkpoint_best_valid_dir)
        saver = tf.train.import_meta_graph(param.best_valid_model_meta_file)
        saver.restore(sess, tf.train.latest_checkpoint(param.checkpoint_best_valid_dir))
    else:
        print("no weights to load :(")


    writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    initial_valid_loss = 100


    for epoch in range(param.training_epochs):

        np.random.shuffle(training_data)

        training_loss = 0.
        total_batch = len(training_data) // param.batch_size +1

        for i in range(total_batch):

            batch_loss = 0
            start = i * param.batch_size
            end = (i + 1) * param.batch_size

            h_data_txt, h_data_img, r_data, t_data_txt, \
            t_data_img, t_neg_data_txt, t_neg_data_img, h_neg_data_txt, h_neg_data_img = u.get_batch_with_neg_heads_and_neg_tails_multimodal(
                training_data, triples_set, entity_list, start,
                end, entity_embeddings_txt, entity_embeddings_img)

            _, loss, summary = sess.run(
                [optimizer, kbc_loss, summary_op],
                feed_dict={r_input: r_data,
                           h_pos_txt_input: h_data_txt,
                           h_pos_img_input: h_data_img,

                           t_pos_txt_input: t_data_txt,
                           t_pos_img_input: t_data_img,

                           t_neg_txt_input: t_neg_data_txt,
                           t_neg_img_input: t_neg_data_img,

                           h_neg_txt_input: h_neg_data_txt,
                           h_neg_img_input: h_neg_data_img,

                           keep_prob: 1 - param.dropout_ratio#,
                           #learning_rate : param.initial_learning_rate
                           })
            #sess.run(clip_all_weights)

            batch_loss = np.sum(loss)/param.batch_size

            training_loss += batch_loss

            writer.add_summary(summary, epoch * total_batch + i)

        training_loss = training_loss / total_batch

        # validating by sampling every epoch


        val_loss = sess.run([kbc_loss],
                            feed_dict={r_input: r_data_valid,
                                       h_pos_txt_input: h_data_valid_txt,
                                       h_pos_img_input: h_data_valid_img,

                                       t_pos_txt_input: t_data_valid_txt,
                                       t_pos_img_input: t_data_valid_img,

                                       t_neg_txt_input: t_neg_data_valid_txt,
                                       t_neg_img_input: t_neg_data_valid_img,

                                       h_neg_txt_input: h_neg_data_valid_txt,
                                       h_neg_img_input: h_neg_data_valid_img,

                                       keep_prob: 1
                                       })

        val_score = np.sum(val_loss) / len(valid_data)

 
        print("Epoch:", (epoch + 1), "loss=", str(round(training_loss, 4)), "val_loss", str(round(val_score, 4)))

        if val_score < initial_valid_loss :
            saver.save(sess, param.model_weights_best_valid_file)
            log_file.write("save model best validation loss: " + str(initial_valid_loss) + "==>" + str(val_score) + "\n")
            print("save model valid loss: ", str(initial_valid_loss), "==>", str(val_score))
            initial_valid_loss = val_score


        saver.save(sess, param.model_current_weights_file)

        log_file.write("Epoch:\t" + str(epoch + 1) + "\tloss:\t" + str(round(training_loss, 5)) + "\tval_loss:\t" + str(
            round(val_score, 5)) + "\n")
        log_file.flush()




