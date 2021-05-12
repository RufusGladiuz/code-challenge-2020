export PWD="$PWD"

echo $PWD

sudo sh docker-clean.sh
sh ./build-task-images.sh 0.1
sudo docker-compose up orchestrator

