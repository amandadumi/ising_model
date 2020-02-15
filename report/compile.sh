#!/bin/bash
echo = "To compile for specific cases, change the case name in compile.sh and file name in main.tex"
case=n_30_m_30_J_1.00_h_0.00
cp ../*$case*.png .
latexmk -pdf main.tex --jobname=$case
