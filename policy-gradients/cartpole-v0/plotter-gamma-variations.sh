#!/bin/bash 

gamma=1.0

echo Starting the script 

python cartpole-pg.py --log_interval 100 --gamma $gamma --neurons 113

for value in {1..40} 
do
	gamma=$(echo "scale=9; $gamma - 0.01" | bc)
	echo Training the cartpool policy gradient with gamma $gamma
	python cartpole-pg.py --gamma $gamma --log_interval 100 --neurons 113
done
