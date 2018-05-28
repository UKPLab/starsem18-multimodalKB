import tc_dataset_creator
import triple_scorer
import tc_evlauator
import  parameters as p


f = open(p.triple_classification_results_file,"w")
for i in range(10):

    tc_dataset_creator.main()
    triple_scorer.main()
    test_avg_acc, avg_acc = tc_evlauator.main()
    f.write(str(test_avg_acc)+"\n")
    f.flush()

f.close()

