#!\bin\bash

for BATCH in 32 1024
do
    for GPUS in 1 4 8
    do
        python3 experiment.py --function train --model resnet18 --batch $BATCH --gpus $GPUS
    done
done
