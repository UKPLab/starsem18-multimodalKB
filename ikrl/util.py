import tensorflow as tf
import codecs
import pickle
import numpy as np
import random
import operator

def save_into_binary_file(data, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(data, f,  protocol=2)

def linear(x, n_output, name=None, activation=None, reuse=None):
    """Fully connected layer.
    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply
    Returns
    -------
    op : tf.Tensor
        Output of fully connected layer.
    """

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope("shared", reuse=reuse):
        W = tf.get_variable(
            name=name + 'w',
            shape=[n_input, n_output],
            dtype=tf.float32,
            initializer=tf.contrib.layers.xavier_initializer())

        b = tf.get_variable(
            name=name + 'b',
            shape=[n_output],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0))

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h #, W


def linear_old(x, n_output, name=None, activation=None, reuse=None,pre_W=None,pre_B=None):
    """Fully connected layer.
    Parameters
    ----------
    x : tf.Tensor
        Input tensor to connect
    n_output : int
        Number of output neurons
    name : None, optional
        Scope to apply
    Returns
    -------
    op : tf.Tensor
        Output of fully connected layer.
    """

    n_input = x.get_shape().as_list()[1]

    with tf.variable_scope(name or "fc", reuse=reuse):
        if pre_W is None:

            W = tf.get_variable(
                name='W',
                shape=[n_input, n_output],
                dtype=tf.float32,
                initializer=tf.contrib.layers.xavier_initializer())
        else:
            W = pre_W

        if pre_B is None:
            b = tf.get_variable(
                name='b',
                shape=[n_output],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0))
        else:
            b = pre_B

        h = tf.nn.bias_add(
            name='h',
            value=tf.matmul(x, W),
            bias=b)

        if activation:
            h = activation(h)

        return h, W


def norm_distance_l1(v1, v2, name):

    distance = tf.reduce_sum(abs(v1 - v2), 1,keep_dims = True, name=name)
    return distance


def norm_distance_l2(v1, v2, name):
    distance = tf.reduce_sum((v1 - v2) ** 2, 1, name=name)

    #distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(v1, v2)),reduction_indices=1), name=name)

    #distance = tf.sqrt(tf.reduce_sum((v1 - v2) ** 2, 1), name=name)
    return distance

def combined_distance_l2(head_txt_relation, t_pos_txt_input,name):

    l2_distance = norm_distance_l2(head_txt_relation, t_pos_txt_input,name="l2_distance")

    cos_distance = cosine_similarity(head_txt_relation, head_txt_relation, name="cos_distance")

    total_distance = tf.add(l2_distance,cos_distance,name=name)

    return total_distance

def cosine_similarity(pred_vectors,true_vectors,name):
    # calc dot product
    dot_products = tf.reduce_sum(tf.multiply(pred_vectors, true_vectors), 1)

    # divide by magnitudes
    pred_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(pred_vectors, pred_vectors), 1))
    true_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(true_vectors, true_vectors), 1))

    cosines = tf.div(dot_products, tf.maximum(tf.multiply(pred_magnitudes, true_magnitudes),1e-08),name=name)
    cosines = tf.maximum(cosines,0)

    return 1-cosines

def cosine_similarity_real(pred_vectors,true_vectors,name):
    # calc dot product
    dot_products = tf.reduce_sum(tf.multiply(pred_vectors, true_vectors), 1)

    # divide by magnitudes
    pred_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(pred_vectors, pred_vectors), 1))
    true_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(true_vectors, true_vectors), 1))

    cosines = tf.div(dot_products, tf.maximum(tf.multiply(pred_magnitudes, true_magnitudes),1e-08),name=name)
    cosines = tf.maximum(cosines, 0)
    return cosines

def gesd(pred_vectors,true_vectors,name):
    #sigmoid: tanh(gamma * dot(a, b) + c)
    #euclidean: 1 / (1 + l2_norm(a - b))
    #        gesd: euclidean * sigmoid

    gamma = 1
    c = 1
    dot_products = tf.reduce_sum(tf.multiply(pred_vectors, true_vectors), 1)

    euclidean = 1 / (1 + tf.norm(pred_vectors - true_vectors))

    sigmoid =  tf.tanh(gamma *dot_products + c)

    gesd = tf.multiply(euclidean,sigmoid,name = name)

    return gesd

def bray_curtis_similarity(pred_vectors,true_vectors,name):

        diff_uv = tf.reduce_sum(tf.abs(pred_vectors-true_vectors))
        sum_uv =  tf.reduce_sum(tf.abs(pred_vectors + true_vectors))

        dist = tf.div(diff_uv, tf.maximum(sum_uv, 1e-08))
        sim =   tf.multiply(1 - dist,1,name = name)
        return sim #tf.div(diff_uv,sum_uv,name = name)



def mse(pred_vectors,true_vectors,name):
    #loss = tf.nn.l2_loss(tf.subtract(pred_vectors ,true_vectors))
    loss = tf.losses.mean_squared_error(true_vectors,pred_vectors)
    return loss

    #differences = pred_vectors - true_vectors
    #square_diff = tf.pow(differences,2)
    #mserror = tf.div(tf.sqrt(tf.reduce_sum(square_diff)),2)

    #return mserror


def dot_product(pred_vectors,true_vectors,name):
    # calc dot product
    dot_products = tf.reduce_sum(tf.multiply(pred_vectors, true_vectors), 1,name=name)

    # divide by magnitudes
    #pred_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(pred_vectors, pred_vectors), 1))
    #true_magnitudes = tf.sqrt(tf.reduce_sum(tf.multiply(true_vectors, true_vectors), 1))

    #cosines = tf.div(dot_products, tf.maximum(tf.multiply(pred_magnitudes, true_magnitudes),1e-08),name=name)

    return dot_products


def get_correct_tails(head, rel, triples):
    correct_tails = [t[1] for t in triples if t[0] == head and t[2] == rel]
    return correct_tails


def load_training_triples(triple_file):
    triple_list = []
    entity_list = []
    text_file = codecs.open(triple_file, "r", "utf-8")
    lines = text_file.readlines()
    for line in lines:
        line_arr = line.rstrip("\r\n").split("\t")
        head = line_arr[0]
        tail = line_arr[1]
        rel = line_arr[2]
        triple_list.append((head, tail, rel))
        entity_list.append(head)
        entity_list.append(tail)
    return triple_list, list(set(entity_list))


def load_entity_list(triple_file,entity_embeddings):
    entity_list = []
    text_file = open(triple_file, "r")
    lines = text_file.readlines()
    for line in lines:
        line_arr = line.rstrip("\r\n").split("\t")
        head = line_arr[0]
        tail = line_arr[1]

        if head in entity_embeddings and tail in entity_embeddings:
           entity_list.append(head)
           entity_list.append(tail)

    return list(set(entity_list))


def load_relation_list(triple_file,entity_embeddings):
    entity_list = []
    text_file = open(triple_file, "r")
    lines = text_file.readlines()
    for line in lines:
        line_arr = line.rstrip("\r\n").split("\t")
        #head = line_arr[0]
        relation = line_arr[1]

        if relation in entity_embeddings:
           entity_list.append(relation)


    return list(set(entity_list))

def load_triples(triple_file, entity_list):
    triple_list = []
    text_file = open(triple_file, "r")
    lines = text_file.readlines()
    for line in lines:
        line_arr = line.rstrip("\r\n").split("\t")
        head = line_arr[0]
        tail = line_arr[1]
        rel = line_arr[2]
        if head in entity_list and tail in entity_list:
            triple_list.append((head, tail, rel))

    return triple_list


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

def load_freebase_triple_data(train_triples_file, freebase_entity_embeddings, relation_fb_embeddings):
    training_intances = []

    training_triples, temp = load_training_triples(train_triples_file)

    for triple in training_triples:
        # head 	 relation 	 tail 	 negative tail 	 negative head
        head = triple[0]
        tail = triple[1]
        rel = triple[2]
        if head in freebase_entity_embeddings and tail in freebase_entity_embeddings:
            head_embd = freebase_entity_embeddings[head]
            tail_embd = freebase_entity_embeddings[tail]
            rel_embd = relation_fb_embeddings[rel]
            train_instance = (head_embd, rel_embd, tail_embd, head, rel, tail)

            training_intances.append(train_instance)

    return training_intances


def load_freebase_triple_data_multimodal(train_triples_file, entity_embeddings_txt,entity_embeddings_img, relation_embeddings):
    training_intances = []

    training_triples, temp = load_training_triples(train_triples_file)

    for triple in training_triples:
        # head 	 relation 	 tail 	 negative tail 	 negative head
        head = triple[0]
        tail = triple[1]
        rel = triple[2]
        if head in entity_embeddings_txt and tail in entity_embeddings_txt:
            head_embd_txt = entity_embeddings_txt[head]#/ np.linalg.norm(entity_embeddings_txt[head])
            tail_embd_txt = entity_embeddings_txt[tail]#/ np.linalg.norm( entity_embeddings_txt[tail])
            head_embd_img = entity_embeddings_img[head]#/ np.linalg.norm(entity_embeddings_img[head])
            tail_embd_img = entity_embeddings_img[tail]#/ np.linalg.norm( entity_embeddings_img[tail])
            rel_embd = relation_embeddings[rel]#/ np.linalg.norm( relation_embeddings[rel])
            train_instance = (head_embd_txt, rel_embd, tail_embd_txt,head_embd_img,tail_embd_img, head, rel, tail)

            training_intances.append(train_instance)

    return training_intances


def get_batch_with_neg_tails(training_data, triples_set, entity_list, start, end, entity_embedding_dict):

    h_data = []
    r_data = []
    t_data = []
    t_neg_data = []

    batch_data = training_data[start:end]

    # (head_embd,rel_embd,tail_embd,head,rel,tail)

    for triple in batch_data:
        h_data.append(triple[0])
        r_data.append(triple[1])
        t_data.append(triple[2])
        text_triple = (triple[3], triple[5], triple[4])
        # print text_triple

        t_neg = sample_negative_tail(triples_set, entity_list, text_triple)[1]
        # if t_neg in entity_embedding_dict:
        t_neg_embed = entity_embedding_dict[t_neg]
        # else:
        # t_neg_embed = np.random.uniform(-1,1,1000)

        t_neg_data.append(t_neg_embed)

    return np.asarray(h_data), np.asarray(r_data), np.asarray(t_data), np.asarray(t_neg_data)


def get_batch_with_neg_tails_multimodal(training_data, triples_set, entity_list, start, end, entity_embedding_txt,entity_embedding_img):
    h_data_txt = [] # head text embeddings
    h_data_img = [] # head image embeddings
    r_data = []  # relation embeddings
    t_data_txt = [] # tail text embeddings
    t_data_img = [] # tail image embeddings
    t_neg_data_txt = [] # negative tail text embeddings
    t_neg_data_img = [] # negative tail image embeddings

    batch_data = training_data[start:end]

    for triple in batch_data:
        # train_instance = (head_embd_txt, rel_embd, tail_embd_txt, head_embd_img, tail_embd_img, head, rel, tail)

        h_data_txt.append(triple[0])
        r_data.append(triple[1])
        t_data_txt.append(triple[2])
        h_data_img.append(triple[3])
        t_data_img.append(triple[4])

        text_triple = (triple[5], triple[7], triple[6])
        # print text_triple

        t_neg = sample_negative_tail(triples_set, entity_list, text_triple)[1]
        t_neg_embed_txt = entity_embedding_txt[t_neg]
        t_neg_embed_img = entity_embedding_img[t_neg]

        t_neg_data_txt.append(t_neg_embed_txt)
        t_neg_data_img.append(t_neg_embed_img)

    return np.asarray(h_data_txt),np.asarray(h_data_img), np.asarray(r_data), np.asarray(t_data_txt),\
           np.asarray(t_data_img), np.asarray(t_neg_data_txt),np.asarray(t_neg_data_img)

def get_batch_with_neg_tails_multimodal_top_k(training_data, triples_set, entity_list, start, end, entity_embedding_txt,entity_embedding_img,nr_neg_tails):
    h_data_txt = [] # head text embeddings
    h_data_img = [] # head image embeddings
    r_data = []  # relation embeddings
    t_data_txt = [] # tail text embeddings
    t_data_img = [] # tail image embeddings
    t_neg_data_txt = [] # negative tail text embeddings
    t_neg_data_img = [] # negative tail image embeddings

    batch_data = training_data[start:end]

    for triple in batch_data:
        # train_instance = (head_embd_txt, rel_embd, tail_embd_txt, head_embd_img, tail_embd_img, head, rel, tail)

        for i in range(nr_neg_tails):
            h_data_txt.append(triple[0])
            r_data.append(triple[1])
            t_data_txt.append(triple[2])
            h_data_img.append(triple[3])
            t_data_img.append(triple[4])

            text_triple = (triple[5], triple[7], triple[6])
            # print text_triple

            t_neg = sample_negative_tail(triples_set, entity_list, text_triple)[1]
            t_neg_embed_txt = entity_embedding_txt[t_neg]
            t_neg_embed_img = entity_embedding_img[t_neg]

            t_neg_data_txt.append(t_neg_embed_txt)
            t_neg_data_img.append(t_neg_embed_img)


    return np.asarray(h_data_txt),np.asarray(h_data_img), np.asarray(r_data), np.asarray(t_data_txt),\
           np.asarray(t_data_img), np.asarray(t_neg_data_txt),np.asarray(t_neg_data_img)


def get_batch_with_neg_heads_and_neg_tails_multimodal(training_data, triples_set, entity_list, start, end, entity_embedding_txt,entity_embedding_img):

    h_data_txt = []  # head text embeddings
    h_data_img = []  # head image embeddings
    h_neg_data_txt = []  # head text embeddings
    h_neg_data_img = []  # head image embeddings
    r_data = []  # relation embeddings
    t_data_txt = []  # tail text embeddings
    t_data_img = []  # tail image embeddings
    t_neg_data_txt = []  # negative tail text embeddings
    t_neg_data_img = []  # negative tail image embeddings

    batch_data = training_data[start:end]

    for triple in batch_data:
        # train_instance = (head_embd_txt, rel_embd, tail_embd_txt, head_embd_img, tail_embd_img, head, rel, tail)

        h_data_txt.append(triple[0])
        r_data.append(triple[1])
        t_data_txt.append(triple[2])
        h_data_img.append(triple[3])
        t_data_img.append(triple[4])

        text_triple = (triple[5], triple[7], triple[6])
        #print("org triple: ", text_triple)

        # print text_triple

        t_neg = sample_negative_tail(triples_set, entity_list, text_triple)[1]
        #print("negative tail: ", t_neg)
        t_neg_embed_txt = entity_embedding_txt[t_neg]
        t_neg_embed_img = entity_embedding_img[t_neg]
        t_neg_data_txt.append(t_neg_embed_txt)
        t_neg_data_img.append(t_neg_embed_img)

        h_neg = sample_negative_head(triples_set, entity_list, text_triple)[0]
        #print("negative head: ", h_neg)

        h_neg_embed_txt = entity_embedding_txt[h_neg]
        h_neg_embed_img = entity_embedding_img[h_neg]
        h_neg_data_txt.append(h_neg_embed_txt)
        h_neg_data_img.append(h_neg_embed_img)


    return np.asarray(h_data_txt), np.asarray(h_data_img), np.asarray(r_data), np.asarray(t_data_txt), \
           np.asarray(t_data_img), np.asarray(t_neg_data_txt), np.asarray(t_neg_data_img),\
           np.asarray(h_neg_data_txt),\
           np.asarray(h_neg_data_img)


def get_batch_with_neg_heads_and_neg_tails_relation_multimodal(training_data, triples_set, entity_list, start, end, entity_embedding_txt,entity_embedding_img,relation_embeddings):

    #chance = random.randint(0,1)
   # print("chance",chance)

    h_data_txt = []  # head text embeddings
    h_data_img = []  # head image embeddings
    h_neg_data_txt = []  # head text embeddings
    h_neg_data_img = []  # head image embeddings
    r_data = []  # relation embeddings
    r_neg_data = []
    t_data_txt = []  # tail text embeddings
    t_data_img = []  # tail image embeddings
    t_neg_data_txt = []  # negative tail text embeddings
    t_neg_data_img = []  # negative tail image embeddings

    batch_data = training_data[start:end]

    counter = 1

    x_0 = 1
    y_0 = 2
    z_0 = 3

    factor = 0

    for triple in batch_data:
        # train_instance = (head_embd_txt, rel_embd, tail_embd_txt, head_embd_img, tail_embd_img, head, rel, tail)

        if counter > 3 :
            factor = 3
        x_t = factor + x_0
        y_t = factor + y_0
        z_t = factor + z_0

        h_data_txt.append(triple[0])
        r_data.append(triple[1])
        t_data_txt.append(triple[2])
        h_data_img.append(triple[3])
        t_data_img.append(triple[4])

        text_triple = (triple[5], triple[7], triple[6])
        #print("org triple: ", text_triple)

        # print text_triple
        t_neg = triple[7]
        if counter == x_t :
            x_0 = x_t
            #print("negative tail", t_neg)
            t_neg = sample_negative_tail(triples_set, entity_list, text_triple)[1]
        #print("negative tail: ", t_neg)
        t_neg_embed_txt = entity_embedding_txt[t_neg]
        t_neg_embed_img = entity_embedding_img[t_neg]
        t_neg_data_txt.append(t_neg_embed_txt)
        t_neg_data_img.append(t_neg_embed_img)

        h_neg = triple[5]
        if counter  == y_t:
            y_0 = y_t
            #print("negative head", h_neg)
            h_neg = sample_negative_head(triples_set, entity_list, text_triple)[0]

        h_neg_embed_txt = entity_embedding_txt[h_neg]
        h_neg_embed_img = entity_embedding_img[h_neg]
        h_neg_data_txt.append(h_neg_embed_txt)
        h_neg_data_img.append(h_neg_embed_img)

        r_neg = triple[6]
        if counter == z_t:
            z_0 = z_t
            #print("negative relation", r_neg)
            r_neg = sample_negative_relation(triple[6],list(relation_embeddings.keys()))

        r_neg_embeddings = relation_embeddings[r_neg]
        r_neg_data.append(r_neg_embeddings)

        counter += 1
    return np.asarray(h_data_txt), np.asarray(h_data_img), np.asarray(r_data), np.asarray(t_data_txt), \
           np.asarray(t_data_img), np.asarray(t_neg_data_txt), np.asarray(t_neg_data_img),\
           np.asarray(h_neg_data_txt),np.asarray(h_neg_data_img),  np.asarray(r_neg_data)


def get_batch_with_neg_tails_hard_neg(training_data, triples_set,entity_list, start, end, entity_embedding_dict,
                                      h_r_t_pos,r_input,h_pos_input,t_pos_input,keep_prob,sess):
    h_data = []
    r_data = []
    t_data = []
    t_neg_data = []

    batch_data = training_data[start:end]

    processed_so_far = 0
    total_to_process = len(training_data)
    percent = 0
    for triple in batch_data:

        if int(total_to_process * percent) == processed_so_far:
            print("processed",percent)
            percent += 0.1

        processed_so_far +=1

        h = triple[3]
        t = triple[5]
        r = triple[4]

        #print("org triple",h,t,r)

        h_emb = entity_embedding_dict[h]
        t_emb = entity_embedding_dict[t]
        r_emb = entity_embedding_dict[r]

        h_data.append(h_emb)
        r_data.append(r_emb)
        t_data.append(t_emb)

        candid_entitys = [e for e in entity_list if e != h and e!=t and h + "_" + e + "_" + r not in triples_set]

        head_embeddings_list = np.tile(h_emb, (len(candid_entitys), 1))
        full_relation_embeddings = np.tile(r_emb, (len(candid_entitys), 1))

        tails_embeddings_list = []

        for i in range(len(candid_entitys)):
            #head_embeddings_list.append(h_emb)
            #full_relation_embeddings.append(r_emb)
            tails_embeddings_list.append(entity_embedding_dict[candid_entitys[i]])


        sim = sess.run([h_r_t_pos], feed_dict={r_input: full_relation_embeddings,
                                               h_pos_input: head_embeddings_list,
                                               t_pos_input: tails_embeddings_list, keep_prob: 1})
        #print("similarites",len(sim[0]))
        '''
        results = {}
        for i in range(0, len(sim[0])):
            results[candid_entitys[i]] = sim[0][i]

        sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        #print(sorted_x)
        hard_neg_tail = sorted_x[0][0]
        '''
        sim = sim[0].tolist()
        index_best_neg_tail = sim.index(max(sim))
        hard_neg_tail = candid_entitys[index_best_neg_tail]

        t_neg_embed = entity_embedding_dict[hard_neg_tail]
        t_neg_data.append(t_neg_embed)
        #print("best negative",hard_neg_tail,sim[index_best_neg_tail])

        '''
        temp_sim = -10
        for entity in entity_list:
            key = h + "_" + entity + "_" + r
            #print(key,list(triples_set)[0])

            if entity != h and entity != t  and key not in triples_set:
                entity_emb = entity_embedding_dict[entity]
                sim = sess.run([h_r_t_pos], feed_dict={r_input: [r_emb],
                           h_pos_input: [h_emb],
                           t_pos_input: [entity_emb], keep_prob: 1})

                if sim >= temp_sim:
                    hard_neg_tail = entity
                    temp_sim = sim
        #print("best negative",hard_neg_tail,sim)

        t_neg_embed = entity_embedding_dict[hard_neg_tail]
        t_neg_data.append(t_neg_embed)
    '''
    print("finished sampling")
    return np.asarray(h_data), np.asarray(r_data), np.asarray(t_data), np.asarray(t_neg_data)



def get_batch_with_neg_tails_hard_neg_top_k(training_data, triples_set,entity_list, start, end, entity_embedding_dict,
                                      h_r_t_pos,r_input,h_pos_input,t_pos_input,keep_prob,sess,k):
    h_data = []
    r_data = []
    t_data = []
    t_neg_data = []

    batch_data = training_data[start:end]
    #print("batch data",len(batch_data))
    processed_so_far = 0
    total_to_process = len(training_data)
    percent = 0
    for triple in batch_data:

        if int(total_to_process * percent) == processed_so_far:
            #print("processed",percent)
            percent += 0.1

        processed_so_far +=1

        h = triple[3]
        t = triple[5]
        r = triple[4]

        h_emb = entity_embedding_dict[h]
        t_emb = entity_embedding_dict[t]
        r_emb = entity_embedding_dict[r]

        #candid_entitys = [e for e in entity_list if e != h and e!=t and h + "_" + e + "_" + r not in triples_set]
        candid_entitys = [e for e in entity_list if e != t and h + "_" + e + "_" + r not in triples_set]

        head_embeddings_list = np.tile(h_emb, (len(candid_entitys), 1))
        relation_embeddings_list = np.tile(r_emb, (len(candid_entitys), 1))

        tails_embeddings_list = []

        for i in range(len(candid_entitys)):

            tails_embeddings_list.append(entity_embedding_dict[candid_entitys[i]])


        sim = sess.run([h_r_t_pos], feed_dict={r_input: relation_embeddings_list,
                                               h_pos_input: head_embeddings_list,
                                               t_pos_input: tails_embeddings_list, keep_prob: 1})

        results = {}
        for i in range(0, len(sim[0])):
            results[candid_entitys[i]] = sim[0][i]

        sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        #print(sorted_x)
        for top_k in range(k):
            h_data.append(h_emb)
            r_data.append(r_emb)
            t_data.append(t_emb)
            hard_neg_tail = sorted_x[k][0]
            t_neg_embed = entity_embedding_dict[hard_neg_tail]
            t_neg_data.append(t_neg_embed)

    #print("finished sampling")
    return np.asarray(h_data), np.asarray(r_data), np.asarray(t_data), np.asarray(t_neg_data)


def get_batch_with_neg_tails_hard_neg_top_k_multimodal(training_data, triples_set, entity_list, start, end, entity_embedding_txt,
                                                       entity_embedding_img, h_r_t_pos, r_input, h_pos_txt_input, t_pos_txt_input,
                                                       h_pos_img_input, t_pos_img_input,keep_prob, sess, k):

    # train_instance = (head_embd_txt, rel_embd, tail_embd_txt, head_embd_img, tail_embd_img, head, rel, tail)

    h_data_txt = []
    h_data_img = []  # head image embeddings
    r_data = []
    t_data_txt = []
    t_data_img = []  # tail image embeddings
    t_neg_data_txt = []
    t_neg_data_img = []  # negative tail image embeddings


    batch_data = training_data[start:end]
    #print("batch data",len(batch_data))
    processed_so_far = 0
    total_to_process = len(training_data)
    percent = 0
    for triple in batch_data:

        if int(total_to_process * percent) == processed_so_far:
            print("processed",percent)
            percent += 0.1

        processed_so_far +=1

        h = triple[5]
        t = triple[7]
        r = triple[6]
        #print(h,r,t)
        h_emb_txt = entity_embedding_txt[h]
        t_emb_txt = entity_embedding_txt[t]
        h_emb_img = entity_embedding_img[h]
        t_emb_img = entity_embedding_img[t]
        r_emb = entity_embedding_txt[r]

        #candid_entitys = [e for e in entity_list if e != h and e!=t and h + "_" + e + "_" + r not in triples_set]
        candid_entitys = [e for e in entity_list if e != t and h + "_" + e + "_" + r not in triples_set]

        head_embeddings_list_txt = np.tile(h_emb_txt, (len(candid_entitys), 1))
        head_embeddings_list_img = np.tile(h_emb_img, (len(candid_entitys), 1))

        relation_embeddings_list = np.tile(r_emb, (len(candid_entitys), 1))

        tails_embeddings_list_txt = []
        tails_embeddings_list_img = []

        for i in range(len(candid_entitys)):

            tails_embeddings_list_txt.append(entity_embedding_txt[candid_entitys[i]])
            tails_embeddings_list_img.append(entity_embedding_img[candid_entitys[i]])



        sim = sess.run([h_r_t_pos], feed_dict={r_input: relation_embeddings_list,
                                               h_pos_txt_input: head_embeddings_list_txt,
                                               h_pos_img_input: head_embeddings_list_img,
                                               t_pos_txt_input: tails_embeddings_list_txt,
                                               t_pos_img_input: tails_embeddings_list_img,
                                               keep_prob: 1})

        results = {}
        for i in range(0, len(sim[0])):
            results[candid_entitys[i]] = sim[0][i]

        sorted_x = sorted(results.items(), key=operator.itemgetter(1), reverse=True)
        #print(sorted_x)
        for top_k in range(k):
            h_data_txt.append(h_emb_txt)
            h_data_img.append(h_emb_img)

            r_data.append(r_emb)
            t_data_txt.append(t_emb_txt)
            t_data_img.append(t_emb_img)
            hard_neg_tail = sorted_x[k][0]
            t_neg_embed_txt = entity_embedding_txt[hard_neg_tail]
            t_neg_embed_img = entity_embedding_img[hard_neg_tail]
            t_neg_data_txt.append(t_neg_embed_txt)
            t_neg_data_img.append(t_neg_embed_img)

    #print("finished sampling")

    # train_instance = (head_embd_txt, rel_embd, tail_embd_txt, head_embd_img, tail_embd_img, head, rel, tail)

    return np.asarray(h_data_txt), np.asarray(h_data_img),np.asarray(r_data), np.asarray(t_data_txt),  np.asarray(t_data_img),np.asarray(t_neg_data_txt),np.asarray(t_neg_data_img)


def sample_negative_tail(triples_set, entity_list, triple_to_corrupt):
    for i in range(len(entity_list)):
        index = random.randint(0, len(entity_list) - 1)
        t_neg = entity_list[index]
        if t_neg != triple_to_corrupt[1]:
            new_tripe = (triple_to_corrupt[0], t_neg, triple_to_corrupt[2])
            key = new_tripe[0] + "_" + new_tripe[1] + "_" + new_tripe[2]
            if key not in triples_set:
                # print i, "break"
                return new_tripe
    new_tripe = (triple_to_corrupt[0], triple_to_corrupt[0], triple_to_corrupt[2])
    return new_tripe


def sample_negative_relation(source_relation,relations_list):


    index = random.randint(0, len(relations_list) - 1)

    neg_relation = relations_list[index]
    while neg_relation == source_relation :
        index = random.randint(0, len(relations_list) - 1)
        neg_relation = relations_list[index]

    return neg_relation



def sample_negative_head(triples_set, entity_list, triple_to_corrupt):
    for i in range(len(entity_list)):
        index = random.randint(0, len(entity_list) - 1)
        h_neg = entity_list[index]
        if h_neg != triple_to_corrupt[0]:
            new_tripe = (h_neg, triple_to_corrupt[1], triple_to_corrupt[2])
            key = new_tripe[0] + "_" + new_tripe[1] + "_" + new_tripe[2]
            if key not in triples_set:
                # print i, "break"
                return new_tripe
    new_tripe = (triple_to_corrupt[1], triple_to_corrupt[1], triple_to_corrupt[2])
    return new_tripe


def get_entity_index(entity_list):
    entity_index = {}
    index_entity = {}
    index = 0

    for e in entity_list:
        entity_index[e] = index
        index_entity[index] = e
        index += 1
    return entity_index,index_entity


def get_correct_tails(head, rel, triples):
    correct_tails = [t[1] for t in triples if t[0] == head and t[2] == rel]
    return correct_tails


def get_correct_heads(tail, rel, triples):
    correct_heads = [t[0] for t in triples if t[1] == tail and t[2] == rel]
    return correct_heads

def create_test_instance(triple, entity_list):
    head = triple[0]
    heads = np.repeat(head, len(entity_list))
    rel = triple[2]
    relations = np.repeat(rel, len(entity_list))
    tails = entity_list
    test_batch = (heads, relations, tails)
    return test_batch



def convert_txt_embeddings_to_binary(file_path,out_path,sep="\t",normalize=False):

    vec_dic = {}
    f = open(file_path)

    lines = f.readlines()
    i = 0
    for line in lines:

        line_arr = line.rstrip("\r\n").split(sep)
        id = line_arr[0]
        vector = line_arr[1:len(line_arr)]
        vector = np.asarray([float(x) for x in vector])
        if normalize:
            vector = vector/np.linalg.norm(vector)
        vec_dic[id] = vector

        print(id,len(vector),np.linalg.norm(vector))
    save_into_binary_file(vec_dic, out_path)
    return vec_dic


#convert_txt_embeddings_to_binary("embeddings/k2b_unif_l1_50_500_epoch.txt","embeddings/k2b_unif_l1_50_500_epoch.pkl")
#convert_txt_embeddings_to_binary("embeddings/k2b_unif_l1_50_500_epoch.txt","embeddings/k2b_unif_l1_50_500_epoch_normalized.pkl",normalize=True)

#x = load_binary_file("/home/mousselly/TF/IKRL_Text_Imagined/embeddings/IKRL_VGG_Glove.pkl")
#for k,vector in x.items():
#    print(k,vector[:4],np.linalg.norm(vector))

'''
norm_dict = {}
print(len(x))
for k,vector in x.items():
    vector = vector / np.linalg.norm(vector)
    norm_dict[k] = vector
    print(k,vector[:3],np.linalg.norm(vector))

#save_into_binary_file(norm_dict,"/home/mousselly/TF/IKRL_Text_Imagined/embeddings/IKRL_Embeddings_Glove_normalized.pkl")
#    print(a,np.linalg.norm(b))

glove_dict = load_binary_file("/home/mousselly/TF/IKRL_Text_Imagined/embeddings/IKRL_Embeddings_Glove_normalized.pkl")
vgg_dict = load_binary_file("/home/mousselly/TF/IKRL_Text_Imagined/VGG_128_IKRL/final/IKRL_vgg128_embeddings_avg_norm.pkl")
mixed_dict = {}
keys = glove_dict.keys()

for entity in keys:
    e_glove = glove_dict[entity]
    e_vgg = vgg_dict[entity]
    e_mm = np.concatenate((e_glove, e_vgg), axis=0)
    e_mm = e_mm / np.linalg.norm(e_mm)

    print(entity,len(e_glove),len(e_vgg),len(e_mm))
    mixed_dict[entity] = e_mm

save_into_binary_file(mixed_dict,"/home/mousselly/TF/IKRL_Text_Imagined/embeddings/IKRL_VGG_Glove.pkl")

'''