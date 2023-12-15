# Double Buffers CEM-TD3: More Efficient Evolution and Richer Exploration (Accepted by AAAI 2024)
Pytorch implementation of DBCEM-TD3 based CEM-TD3: https://arxiv.org/pdf/1810.01222.pdf

To reproduce the results of the paper:

Use sh file to reproduce all results:
```terminal
./run.sh
```


Reproduce the results of a specified task:
```console
python es_grad_dbcemtd3.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```
