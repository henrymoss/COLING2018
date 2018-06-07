# COLING2018
Accompanying code for "Using J-K-fold Cross Validation to Reduce Variance When Tuning NLP Models" from COLING 2018

Due to the heavy computational nature of this paper the code is split into interactive and non-interactive scripts (Python 3).

The non-interactive scripts consist of the repeated fitting of the machine learning models on the various NLP tasks. To collect the results from 1,000  different random partitionings we made use of a computing cluster resource. These scripts will not run in a reasonable time on a standard machine (without reducing the number of random partitionings). However, we include them here for completeness and reproducibility. We do however include the output from running these scripts for use with the interactive scripts. 

The interactive scripts are jupyter notebooks for Python3, which load in pre-computed data (the saved output from the non-interactive scripts). They allow the creation of all the plots and tables that rely on choices of K and J (allowing further experimentation at very low further computational cost). Figure 1 has no interaction elements and so is included as a non-interactive script.

Note that the non-interactive scripts require the download of the data. 

For IMDB https://github.com/jalbertbowden/large-movie-reviews-dataset/tree/master/acl-imdb-v1

For Reuters and Brown corpora use Python's NLTK package (see relevant script)

The set-up for the LSTM is significantly more complicated and so extra information for running that experiment is included in  non-interactive/LSTM folder (including instructions for obtaining data).
