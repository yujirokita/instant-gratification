#!/bin/bash

mkdir input
mkdir script
mkdir script/log

kaggle competitions download -q -c instant-gratification -p input
unzip "input/*.zip" -d input
chmod 444 input/*.csv