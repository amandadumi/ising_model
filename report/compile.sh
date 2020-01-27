#!/bin/bash
cp ../*png .
latexmk -pdf main.tex
