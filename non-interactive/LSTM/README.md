This is an implementation of the LSTM model that is shown in [Tang et al. paper](https://aclanthology.info/papers/C16-1311/c16-1311).

This code is almost entirely taken from the work of Andrew Moore hosted at https://github.com/apmoore1/Bella. 

Here we present a simplfied version of Andrew's code which just contains enough code to fit LSTM models for different random partitionings. I have also remove parallelization though Keras to allow reproducible simualtions.

To load the dataset you need to set the config.yaml file so that `dong_twit_train_data` and `dong_twit_test_data` point to the corresponding data files which can be downloaded from 

https://github.com/bluemonk482/tdparse/tree/master/data/lidong/training

and

https://github.com/bluemonk482/tdparse/tree/master/data/lidong/testing


You also need to set the config.yaml to point to the pre-trained word-embeddings contained in sswe.tsv, stored at https://github.com/apmoore1/Bella/tree/master/data/word_vectors/vo_zhang



The code to run the simulations is contained in the simulation_scripts folder.

Note that Figure 4C is the combination of multiple runs of Figure_4C.py with different random seed and numbers of folds as arguments.
