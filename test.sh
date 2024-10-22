#!/bin/sh
( for i in $(seq 1 100); do python test.py; done ) | tee results.txt
