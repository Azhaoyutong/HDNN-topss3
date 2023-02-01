# HDNNtopss

This repository includes two ways, HDNNtopss and Co-HDNNtopss, HDNNtopss, which describes the actual force as the final predictor.

This repository contains Tensorflow implementations of the Hybrid DNN-CHMM models initially introduced in:

T. Gao and Y. Zhao, “Secondary and Topology Structure Merge Prediction of Alpha-Helical Transmembrane Protein using a Hybrid Model Based on Hidden Markov and Long Short-Term Memory Neural Networks” .

## Requirements

To run it, follow the instructions below.

Python ≥ 3.6 
tensorflow2.1.0  keras2.2.5

HH-suite(https://github.com/soedinglab/hh-suite) for generating HHblits files (with the file suffix of .hhm).
## Download

git-lfs clone https://github.com/NENUBioCompute/HDNNtopss.git

## Test & Evaluate in Command Line
For example of HDNNtopss:

```
cd ./HDNNtopss/HDNNtopss ("./" represents the local path of repository;)
python HDNNtopss_run.py -f ./datasets/test.txt -p ./datasets/test_hhm/ -o ./result
```
For example of Co-HDNNtopss:

```
cd ./HDNNtopss/Co-HDNNtopss ("./" represents the local path of repository;)
python HDNNtopss_run.py -f ./datasets/test.txt -p ./datasets/test_hhm/ -o ./result/
```
channels:
  - To set the path of fasta file, use or .```--fasta``` ```-f```
  - To set the path of generated HHblits files, use or .```--hhblits_path``` ```-p```
  - To set the outputs to a directory, use or .```--output``` ```-o```

