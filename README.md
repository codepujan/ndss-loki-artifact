# ndss-loki-artifact
Companion repository for our NDSS'26 Loki paper 

## Setup 
Run the script load_models_and_datasets.py to download all the datasets and models used in the paper. Also make sure to re-use the same cache directory used in this script on subsequent experiments for caching purposes. 

## Metadata files download 
Please download these additional set of files containing keywords metadata before running **experiment_2.py**

* [keywords_categories.json](https://www.zenodo.org/records/16741269/files/keywords_categories.json?download=1)
* [query_ner_output.json](https://zenodo.org/records/16741269/files/query_ner_output.json?download=1)

## Experiment_1.py
Make sures the basic functionality tests are running. Demonstrates the core application of the contribution: a keyword scoring model that can predict toxicity score (between 0-1) for any search keyword, and subsequently used for ranking a list of keywords. 

## Experiment_2.py 
Detailed comparison of experimental evaluation discussed in the paper. Produces Table 2 , 3, and 5. Demonstrating Loki's superior performance over baseline models. 
