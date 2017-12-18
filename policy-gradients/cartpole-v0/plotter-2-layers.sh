#!/bin/bash 

neurons1=0
neurons2=0

echo Starting the script 

for value1 in {1..3} 
do
	neurons1=$((neurons1+8))
	for value2 in {1..3}
		do
			neurons2=$((neurons2+8))
			echo Training the cartpole policy gradient with $neurons1 and $neurons2 neurons
			python cartpole-pg-2-layers.py --neurons1 $neurons1 --neurons2 $neurons2  --log_interval 100 
		done
	neurons2=$((0))
done
