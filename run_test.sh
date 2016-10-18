#!/bin/sh

python svhn_multi.py | tee out.txt
git commit -a -m 'Latest run'
git push origin master
