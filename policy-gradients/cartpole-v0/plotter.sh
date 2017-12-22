#!/bin/bash 



echo Starting the script 

for neurons in {3..200} 
do
	echo Training the cartPole policy gradient with $neurons neurons
	python cartpole-pg.py --neurons $neurons --log_interval 100 
done
