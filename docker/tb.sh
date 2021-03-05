if [ $# -lt 1 ]; then
    echo "usage:" $0 "[container_name]"
    exit 1
fi

# run tensorboard inside container
docker exec -d $1 tensorboard --logdir=~/ray_results --bind_all