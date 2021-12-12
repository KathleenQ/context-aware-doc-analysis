# Knowledge Graph Completion Based on Natural Language Processing Contents Analysis Techniques:
This repository corresponds to **Siying Qian's Honours Research Project "Context-Aware Document Analysis"**.

The codes are a runnable and extensional version of codes in ["**GPT-GNN: Generative Pre-Training of Graph Neural Networks**"](https://github.com/acbull/GPT-GNN).

Raw data for the pre-processing stage is the [data provided in the GPT-GNN repository](https://drive.google.com/drive/folders/1yDdVaartOCOSsQlUZs8cJcAUhmvRiBSz).

Chosen custom graphs from preprocessing *(preprocess_output)*, pre-trained models *(models)* and fine-tuned models *(finetune_models)* mentioned in the "Results Analysis" section of my thesis can be achieved via [this link](https://drive.google.com/drive/folders/1qqKUmyaTtAxsA0_RzIzCt1g_ZGiNcfhs?usp=sharing).

Different content analysis versions for titles and abstracts and various keyword extraction versions are listed in the *preprocess.py* file.
Whether abstract embedding is considered corresponds to two GNN model versions in the *pretrain.py*, *finetune_PV.py*, *finetune_AD.py* and *GPT_GNN/utils.py* files.
