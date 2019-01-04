#!/bin/bash  

# Read environment 
export PASSWORD=${JUPYTER_PASSWORD} 
export PORT=${JUPYTER_PORT:-8888} 
export TENSORBOARD_LOGDIR=${TENSORBOARD_LOGDIR:-../logs/} 
export TENSORBOARD_PORT=${TENSORBOARD_PORT:-6006} 

# Launch tensorboard 
tensorboard --logdir=${TENSORBOARD_LOGDIR} --host 0.0.0.0 --port ${TENSORBOARD_PORT} & 

# Launch jupyter 
jupyter notebook --ip=0.0.0.0 --port=${PORT} --allow-root 

exec $@
