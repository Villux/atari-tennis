#!/bin/bash 

neurons=0

echo Starting the script 

for value in {1..6} 
do
	neurons=$((neurons+32))
	echo Training the cartpool policy gradient with $neurons neurons
	python cartpole-pg.py --neurons $neurons --log_interval 100 
done