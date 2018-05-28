# starsem18-multimodalKB
# Multimodal Knowledge Graph Completion

> **Abstract:** 
Current methods for knowledge graph (KG) representation learning focus solely on the structure of the KG and do not exploit any kind of external information, such as visual and linguistic information corresponding to the KG entities. In this paper, we propose a multimodal translation-based approach that defines the energy of a KG triple as the sum of sub-energy functions that leverage both multimodal (visual and linguistic) and structural KG representations. Next, a ranking-based loss is minimized using a simple neural network architecture. Moreover, we introduce a new large-scale dataset for multimodal KG representation learning. We compared the performance of our approach to other baselines on two standard tasks, namely knowledge graph completion and triple classification, using our as well as the WN9-IMG dataset. The results demonstrate that our approach outperforms all baselines on both tasks and datasets.

Please use the following citation:
```
@InProceedings{moussellySergieh2018multimodal,
  title = {{A Multimodal Translation-Based Approach for Knowledge Graph Representation Learning}},
	author = {Mousselly Sergieh, Hatem and Botschen, Teresa and Gurevych, Iryna and Roth, Stefan},
	publisher = {Association for Computational Linguistics},
	booktitle = {Proceedings of the 7th Joint Conference on Lexical and Computational Semantics (*SEM 2018)},
	publisher = {Association for Computational Linguistics},
	pages = {to appear},
	month = jun,
	year = {2018},
	location = {New Orleans, USA}
}
```

Contact person: Hatem Mousselly Sergieh, h.m.sergieh@gmail.com ; Teresa Botschen, botschen@aiphes.tu-darmstadt.de

https://www.ukp.tu-darmstadt.de/

https://www.tu-darmstadt.de/


Don't hesitate to send us an e-mail or report an issue, if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication. 

### Folder structure

The reprository contains three folders. Each folder contains the code for traning and testing as well as a folder for conducting the triple classification experiment.
The implemented methods include:
- IKRL: This is an implementation of the IKRL model of (Xie et al., 2017) https://www.ijcai.org/proceedings/2017/0438.pdf
- Our mehthod
- Triple classfication for TransE method (Bordes et al., 2013) http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf

## Using the code
Each implementation contain a parameter file.
- For training you should edit the paramters.py file and run the training script.
- For testing edit the test_paramters file and call the test script.
- For the triple classification experiment you need to set the paramters in the corresponding file and then run the tc_batch.py scirpt.

## Dataset and Embeddings
This project uses different pretrained word embeddings. 
The datasets used in the experiments as well as the different embedding files can be found here:
https://fileserver.ukp.informatik.tu-darmstadt.de/starsem18-multimodalKB/


