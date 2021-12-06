#!/bin/sh
for i in {0..8}; do
    python ModelSplit_client.py --id $i &:
done &
python ModelSplit_client.py --id 9
#
#for i in {0..8}; do
#    python clear_dense_client.py --id $i &:
#done &
#python clear_dense_client.py --id 9
#python mpc_dbscan.py --multiprocess --world_size 2