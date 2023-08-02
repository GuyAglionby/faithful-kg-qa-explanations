#!/bin/bash

mv data data_old

wget https://nlp.stanford.edu/projects/myasu/QAGNN/data_preprocessed_release.zip
unzip data_preprocessed_release.zip
mv data_preprocessed_release data

mv data/csqa data/csqa-cpnet
mv data/obqa data/obqa-cpnet