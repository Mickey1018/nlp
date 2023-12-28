#!/bin/sh
sudo docker build -t ct-immd-nlp:v0.91 .
sudo docker save ct-immd-nlp:v0.91 | gzip > ct-immd-nlpv0.91.tar.gz
# scp ct-immd-nlpv0.9.tar.gz usnmp@nv01.vabot.org:/home/usnmp/Downloads