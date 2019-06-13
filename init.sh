#!/bin/bash

mkdir input
mkdir data
mkdir script
mkdir script/log
mkdir temp

kaggle competitions download -q -c instant-gratification -p input
unzip "input/*.zip" -d input
chmod 444 input/*.csv