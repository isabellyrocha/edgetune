#!\bin\bash

for BATCH in 32 512 1024
do
    python3 experiment.py --function train --model resnet18 --batch $BATCH --gpus 8
done
