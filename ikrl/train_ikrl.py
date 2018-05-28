import os

import numpy as np
import tensorflow as tf

import parameters  as param
import util as u

logs_path = "log"
# .... Loading the data ....
print("load all triples")
relation_embeddings = u.load_binary_file(param.relation_structural_embeddings_file)
entity_embeddings_txt = u.load_binary_file(param.entity_structural_embeddings_file)
entity_embeddings_img = u.load_binary_file(param.entity_multimodal_embeddings_file)

# Remove triples that don't have embeddings
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
print("#training data", len(training_data))

valid_data = u.load_freebase_triple_data_multimodal(param.valid_triples_file, entity_embeddings_txt,
                                                    entity_embeddings_img,relation_embeddings)


h_data_valid_txt, h_data_valid_img, r_data_valid, t_data_valid_txt, \
t_data_valid_img, t_neg_data_valid_txt, t_neg_data_valid_img, h_neg_data_valid_txt, h_neg_data_valid_img, r_neg_data_valid = \
    u.get_batch_with_neg_heads_and_neg_tails_relation_multimodal(valid_data,
                                                        triples_set,
                                                        entity_list,
                                                        0, len(valid_data),
                                                        entity_embeddings_txt,
                                                        entity_embeddings_img,relation_embeddings)




def my_dense(x, nr_hidden, scope, activation_fn=param.activation_function,reuse=None):
    with tf.variable_scope(scope):


        h = tf.contrib.layers.fully_connected(x, nr_hidden,
                                               activation_fn=activation_fn,
                                               reuse=reuse,
                                               scope=scope,weights_regularizer=None
                                               )

        return h


# ........... Creating the model
with tf.name_scope('input'):
    # Relation
    r_input = tf.placeholder(dtype=tf.float32, shape=[None, param.relation_structural_embeddings_size],name="r_input")
    r_neg_input = tf.placeholder(dtype=tf.float32, shape=[None, param.relation_structural_embeddings_size],name="r_input")

    # Head input: negative and positive. The txt correspond to the structural embeddings and the img to the multimodal emebddings
    h_pos_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="h_pos_txt_input")
    h_neg_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="h_neg_txt_input")

    h_pos_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size],name="h_pos_img_input")
    h_neg_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size], name="h_neg_img_input")

    # Tail input: negative and positive. The txt correspond to the structural embeddings and the img to the multimodal emebddings
    t_pos_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size], name="t_pos_txt_input")
    t_pos_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size], name="t_pos_img_input")

    t_neg_txt_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_structural_embeddings_size],   name="t_neg_txt_input")
    t_neg_img_input = tf.placeholder(dtype=tf.float32, shape=[None, param.entity_multimodal_embeddings_size], name="t_neg_img_input")


with tf.name_scope('head_relation'):

    # Mapping the multimodal representation of the head and the tail entities
    h_pos_img_mapped = my_dense(h_pos_img_input, param.relation_structural_embeddings_size, activation_fn=param.activation_function, scope="img_proj", reuse=None)
    h_neg_img_mapped = my_dense(h_neg_img_input, param.relation_structural_embeddings_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)

    # Tail image ....
    t_pos_img_mapped = my_dense(t_pos_img_input, param.relation_structural_embeddings_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)
    t_neg_img_mapped = my_dense(t_neg_img_input, param.relation_structural_embeddings_size, activation_fn=param.activation_function, scope="img_proj", reuse=True)


with tf.name_scope('cosine'):

    # Energy calculation for postive and negative triples
    pos_s_s = tf.reduce_sum(abs(h_pos_txt_input + r_input - t_pos_txt_input), 1, keep_dims = True,  name="pos_s_s")
    neg_s_s = tf.reduce_sum(abs(h_neg_txt_input + r_neg_input - t_neg_txt_input), 1, keep_dims = True,   name="neg_s_s")

    pos_i_i = tf.reduce_sum(abs(h_pos_img_mapped + r_input - t_pos_img_mapped), 1, keep_dims = True,  name="pos_i_i")
    neg_i_i = tf.reduce_sum(abs(h_neg_img_mapped + r_neg_input - t_neg_img_mapped) , 1, keep_dims = True,  name="neg_i_i")

    pos_s_i = tf.reduce_sum(abs(h_pos_txt_input + r_input - t_pos_img_mapped) , 1, keep_dims = True, name="pos_s_i")
    neg_s_i = tf.reduce_sum(abs(h_neg_txt_input + r_neg_input - t_neg_img_mapped), 1, keep_dims = True,  name="neg_s_i")

    pos_i_s = tf.reduce_sum(abs(h_pos_img_mapped + r_input - t_pos_txt_input), 1, keep_dims = True, name="pos_i_i")
    neg_i_s = tf.reduce_sum(abs(h_neg_img_mapped + r_neg_input - t_neg_txt_input), 1, keep_dims = True,  name="neg_i_i")

    pos_energy = tf.reduce_sum([pos_s_s, pos_i_i, pos_s_i, pos_i_s],0, name="pos_energy")
    negative_energy = tf.reduce_sum([neg_s_s, neg_i_i, neg_s_i, neg_i_s],0, name="negative_energy")

    kbc_loss = tf.maximum(pos_energy - negative_energy + param.margin, 0)
    tf.summary.histogram("loss", kbc_loss)


optimizer = tf.train.AdamOptimizer(param.initial_learning_rate).minimize(kbc_loss)


summary_op = tf.summary.merge_all()

#..... start the training
saver = tf.train.Saver()
log_file = open(param.log_file,"w")
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

with tf.Session(config=sess_config) as sess:
    sess.run(tf.global_variables_initializer())

    # Load pre-trained weights if available
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
        total_batch = int(len(training_data) / param.batch_size)

        for i in range(total_batch):

            batch_loss = 0
            start = i * param.batch_size
            end = (i + 1) * param.batch_size

            h_data_txt, h_data_img, r_data, t_data_txt, \
            t_data_img, t_neg_data_txt, t_neg_data_img, h_neg_data_txt, h_neg_data_img, r_neg_data= u.get_batch_with_neg_heads_and_neg_tails_relation_multimodal(
                training_data, triples_set, entity_list, start,
                end, entity_embeddings_txt, entity_embeddings_img,relation_embeddings)

            _, loss, summary = sess.run(
                [optimizer, kbc_loss, summary_op],
                feed_dict={r_input: r_data,
                           r_neg_input: r_neg_data,
                           h_pos_txt_input: h_data_txt,
                           h_pos_img_input: h_data_img,

                           t_pos_txt_input: t_data_txt,
                           t_pos_img_input: t_data_img,

                           t_neg_txt_input: t_neg_data_txt,
                           t_neg_img_input: t_neg_data_img,

                           h_neg_txt_input: h_neg_data_txt,
                           h_neg_img_input: h_neg_data_img
                           })

            batch_loss = np.sum(loss)/param.batch_size

            training_loss += batch_loss

            writer.add_summary(summary, epoch * total_batch + i)

        training_loss = training_loss / total_batch

        val_loss = sess.run([kbc_loss],
                            feed_dict={r_input: r_data_valid,
                                       r_neg_input: r_neg_data_valid,

                                       h_pos_txt_input: h_data_valid_txt,
                                       h_pos_img_input: h_data_valid_img,

                                       t_pos_txt_input: t_data_valid_txt,
                                       t_pos_img_input: t_data_valid_img,

                                       t_neg_txt_input: t_neg_data_valid_txt,
                                       t_neg_img_input: t_neg_data_valid_img,

                                       h_neg_txt_input: h_neg_data_valid_txt,
                                       h_neg_img_input: h_neg_data_valid_img

                                       })

        val_score = np.sum(val_loss) / len(valid_data)


        print("Epoch:", (epoch + 1),  "loss=", str(round(training_loss, 4)), "val_loss", str(round(val_score, 4)))

        if val_score < initial_valid_loss :
            saver.save(sess, param.model_weights_best_valid_file)
            log_file.write("save model best validation loss: " + str(initial_valid_loss) + "==>" + str(val_score) + "\n")
            print("save model valid loss: ", str(initial_valid_loss), "==>", str(val_score))
            initial_valid_loss = val_score


        saver.save(sess, param.model_current_weights_file)

        log_file.write("Epoch:\t" + str(epoch + 1) + "\tloss:\t" + str(round(training_loss, 5)) + "\tval_loss:\t" + str(
            round(val_score, 5)) + "\n")
        log_file.flush()




