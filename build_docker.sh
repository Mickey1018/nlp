#!/bin/sh
docker build -t ct-immd-nlp:v0.9 .
docker save ct-immd-nlp:v0.9 | gzip > ct-immd-nlpv0.9.tar.gz
scp ct-immd-nlpv0.9.tar.gz usnmp@nv01.vabot.org:/home/usnmp/Downloads