#!/bin/bash

# Script to reproduce results


for ((i=0;i<1;i+=1))
do
  python es_grad_dbcemtd3.py \
  --env "Ant-v2" \
  --use_td3\
	--seed $i
done
