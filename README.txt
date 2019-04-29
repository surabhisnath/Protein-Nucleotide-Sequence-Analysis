The files need to be pulled from dockerhub. For this, run the command -

docker pull surabhisnath/bdmh_assignment2

With this, the image will get downloaded. Next, create a container from the image using the command:

docker run -it surabhisnath/bdmh_assignment2:final /bin/bash


The files can now be viewed and executed ------------------------------------------------------------------

The container consists of the following python3 files:

1) Ques1.py
2) Ques2.py
3) Ques3.py
4) Ques4.py

To run the files, the following calls need to be made:

1) Ques1.py

python3 Ques1.py -i seq.txt -o out1.txt
The input file should contain the nucleotide sequence for which repeats and inverse repeats are to be calculated
If multiple sequences are to be evaluated, each sequence should be on a newline
The output file will display the sequence, number of repeats and number of inverse repeats
Algorithm used - for repeats, the diagonals of half matrix are iterated and 4 or more consecutive 1s are noted as a repeating sequence. For inverse repeats, perpendiculars to diagonal are iterated and 4 or more consecutive 1s are noted as an inverse repeat.

2) Ques2.py

python3 Ques2.py -f ./fastafiles/ -o out2.txt

The path containing the fasta files needs to be provided as input. All files are concatenated together. The amino acid composition and atomic compositions are calculated from scratch and the feature vectors are concatenated to form a 25 dimentional vector. 2 types of clustering is performed - KMeans and Heirarchical. The number of clusters are varied and for each case, a Silhouette score is evaluated. The largest score represents the ideal number of clusters. The ideal num_clusters with corresponding score is written to the output file

3) Ques3.py

python3 Ques3.py -abci /gpsr/examples/example.fasta -abco out31.txt -ctli /gpsr/examples/example.fasta -ctlo out32.txt -proi /gpsr/examples/example.fasta -proo out33.txt -toxini /gpsr/examples/example.fasta -toxino out34.txt

Input and output fasta files are to be provided for each predictor - abcpred, ctlpred, propred, toxinpred. The corresponding perl codes are called and output files are created.

4) Ques4.py

python3 Ques4.py -p P.txt -n N.txt -o out4.txt

The Positive examples file and Negative examples file are passed as input. 3 classifiers - SVM, ANN and RF are implemented with 5 fold cross validation. Sensitivity, Specificity, Accuracy and MCC are written into output file for each fold of each classifier. It is observed that since the dataset is highly imbalanced, with P:N = 1:10, few classifiers do not perform well and classify all points to the N class.