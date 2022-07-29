#!/bin/bash

cd eval
echo "installing evaluation dependencies"
echo "downloading e2e-metrics..."
git clone https://github.com/tuetschek/e2e-metrics e2e
pip install -r e2e/requirements.txt

echo "downloading GenerationEval for webnlg and dart..."
git clone https://github.com/WebNLG/GenerationEval.git
cd GenerationEval

# INSTALL PYTHON DEPENDENCIES
pip install -r requirements.txt

# INSTALL BLEURT
pip install --upgrade pip
git clone https://github.com/google-research/bleurt.git
cd bleurt
pip install .
wget https://storage.googleapis.com/bleurt-oss/bleurt-base-128.zip
unzip bleurt-base-128.zip
rm bleurt-base-128.zip 
cd ../
mv bleurt metrics

# INSTALL METEOR
wget https://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz
tar -xvf meteor-1.5.tar.gz
mv meteor-1.5 metrics
rm meteor-1.5.tar.gz

rm -r data/en
rm -r data/ru
cd ..
mv eval.py GenerationEval/

echo "script complete!"
