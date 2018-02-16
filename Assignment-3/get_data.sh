#!/bin/bash

echo "Downloading the dataset"

wget https://www.cse.iitb.ac.in/~rdabral/CS763/Train/data.bin
wget https://www.cse.iitb.ac.in/~rdabral/CS763/Train/labels.bin
wget https://www.cse.iitb.ac.in/~rdabral/CS763/Test/test.bin

mkdir Data
mv test.bin Data
mv train.bin Data
mv labels.bin Data