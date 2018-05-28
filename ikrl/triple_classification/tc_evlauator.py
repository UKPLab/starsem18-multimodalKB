import parameters as p

def load_relations(relation_file):
    relations = []
    f = open(relation_file,"r")
    lines = f.readlines()
    for l in lines:
        r = l.rstrip("\n\r")
        relations.append(r)
    return relations

def get_score_class_paris(triple_score_file,label_name):

    f = open(triple_score_file,"r")
    lines = f.readlines()
    pairs = []
    for line in lines :
        #print(line)
        line_arr = line.rstrip("\r\n").split("\t")
        score = float(line_arr[3].replace("[","").replace("]",""))
        label = line_arr[4]
        if label_name in label:
            pair = (score,label)
            pairs.append(pair)
    return pairs

def identify_threhold(pairs,label):
    pairs.sort(key=lambda tup: tup[0])  # sorts in place
    temp = 0
    final_th = 0
    for i in range(0,len(pairs)-1):
            j = i+1
            v1 = pairs[i][0]
            v2 = pairs[j][0]
            th = (v1 + v2)/2
            acc = calculate_accuracy(pairs, th, label)
           # print("acc",acc)
            if acc > temp:
                temp = acc
                final_th = th
    return final_th

def calculate_accuracy(pairs,threshold,label):
    total_pos = [x for x in pairs if x[1] in label+"_1"]
    total_pos_correct = [x for x in pairs if x[1] in label+"_1" and x[0] < threshold]

    total_neg = [x for x in pairs if x[1] in label+"_0"]
    total_neg_correct = [x for x in pairs if x[1] in label + "_0" and x[0] >= threshold]

    accuracy = (float(len(total_pos_correct))+float(len(total_neg_correct)))/(float(len(total_pos))+float(len(total_neg)))

    return accuracy


def main():

    relations = load_relations(p.relation_file)
    valid_avg_acc = 0.
    total_relations = 0
    relation_th_pair = dict()
    for r in relations:
        pairs = get_score_class_paris(p.valid_triple_score_file, r)
        if len(pairs) > 1:
            total_relations += 1
            final_threshold = identify_threhold(pairs,r)
            relation_th_pair[r] = final_threshold

            acc = calculate_accuracy(pairs,final_threshold,r)
            valid_avg_acc += acc
            print(r,final_threshold,acc,"total pairs",len(pairs))

    valid_avg_acc = valid_avg_acc/ total_relations
    print("thresholds", relation_th_pair)
    print("validation average accuracy",valid_avg_acc)

    ### Apply on the test data ####
    print("....... Testing ..........")
    test_avg_acc = 0.
    total_relations = 0
    for r,final_threshold in relation_th_pair.items():
        pairs = get_score_class_paris(p.test_triple_score_file, r)
        if len(pairs) > 1:
            total_relations += 1
            acc = calculate_accuracy(pairs,final_threshold,r)
            test_avg_acc += acc
            print(r, "#pairs:", len(pairs),"threshold:",final_threshold,"Accuracy:",acc)

    test_avg_acc = test_avg_acc/ total_relations
    print("test average accuracy",test_avg_acc)
    return test_avg_acc, valid_avg_acc

if __name__ == '__main__':
    main()