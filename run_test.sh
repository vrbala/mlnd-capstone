#!/bin/sh

python svhn_multi.py >out.txt
git commit -a -m 'Latest run'
git push origin master
