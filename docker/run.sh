NAME=$1

if [ $NAME ]; then
    docker run --gpus all -P \
           --name $NAME \
           -it --rm \
           --shm-size 80g \
           ns3-gym-csmarl:src bash
else
    docker run --gpus all -P \
           -it --rm \
           --shm-size 80g \
           ns3-gym-csmarl:src bash
fi
