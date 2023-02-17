#! /bin/bash
# source run.sh
SCRIPT_NAME=$1
export  NCCL_DEBUG=INFO;export NCCL_IB_GID_INDEX=3;export NCCL_IB_DISABLE=0;export NCCL_IB_HCA=mlx5_bond;
if [ "$MASTER_ADDR" == "localhost" ] ; then MASTER_ADDR=`hostname`; fi
GPUS_PER_NODE=8
# Change for multinode config
NNODES=$WORLD_SIZE
NODE_RANK=$RANK
ALL_GPU_NUM=`expr ${WORLD_SIZE} \* 8`
echo "ALL_GPU_NUM" ${ALL_GPU_NUM}
echo $MASTER_ADDR
echo $MASTER_PORT

/opt/conda/bin/conda activate ldm
source activate ldm
pip install oss2 bezier opencv-python albumentations onnxruntime-gpu
pip install diffusers==0.10.0 --upgrade
pip install transformers --upgrade

DISTRIBUTED_ARGS="--multi_gpu --num_processes $ALL_GPU_NUM --num_machines $NNODES --machine_rank $NODE_RANK --main_process_ip $MASTER_ADDR --main_process_port $MASTER_PORT"
echo DISTRIBUTED_ARGS
echo ${DISTRIBUTED_ARGS}
echo "/opt/conda/envs/ldm/bin/accelerate launch $DISTRIBUTED_ARGS ${SCRIPT_NAME}"
/opt/conda/envs/ldm/bin/accelerate launch $DISTRIBUTED_ARGS ${SCRIPT_NAME}
