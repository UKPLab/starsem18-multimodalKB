import helper_functions as u
from random import randint
import  parameters as p
def main():

    modes = ["valid","test"]


    for mode in modes:
        triple_file = p.triples_base_dir+mode+".txt"
        triples = u.load_triples(triple_file)
        relation_head_dict, relation_tail_dict = u.create_relation_dicts(triples)

        print("............")

        if mode == "valid":
            file_name = p.valid_triple_file
        else:
            file_name = p.test_triple_file
        f = open(file_name,"w")
        #index = 0
        for triple in triples:
            index = randint(0, 9)

            if index % 2 == 0:

                f.write(triple[0]+"\t"+triple[1]+"\t"+triple[2]+"\t"+triple[2]+"_"+str(1)+"\n")
                neg_triple = u.create_negative_triple(triple, relation_head_dict, relation_tail_dict, corrupt_head=True)
                #print("org: ", triple, "negative head triple", neg_triple)

                if neg_triple == None:
                    neg_triple = u.create_negative_triple(triple, relation_head_dict, relation_tail_dict, corrupt_head=False)
                if neg_triple != None:
                    f.write(neg_triple[0] + "\t" + neg_triple[1] + "\t" + neg_triple[2] + "\t" + neg_triple[2] +"_" + str(0)+"\n")


            else:

                f.write(triple[0] + "\t" + triple[1] + "\t" + triple[2] + "\t" + triple[2] + "_" + str(1) + "\n")

                neg_triple = u.create_negative_triple(triple, relation_head_dict, relation_tail_dict, corrupt_head=False)
                #print("org: ", triple,"negative tail triple", neg_triple)

                if neg_triple == None:
                    neg_triple = u.create_negative_triple(triple, relation_head_dict, relation_tail_dict, corrupt_head=True)
                if neg_triple != None:
                    f.write(neg_triple[0] + "\t" + neg_triple[1] + "\t" + neg_triple[2] + "\t" + neg_triple[2] + "_" + str(0) + "\n")
            index +=1
        f.close()
        print("evaluation datasets created")

if __name__ == '__main__':
    main()


