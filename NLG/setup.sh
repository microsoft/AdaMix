sudo apt-get update
sudo apt-get -y install jq virtualenv

virtualenv -p `which python3` ./venv
. ./venv/bin/activate
pip install -r requirement.txt
source download_pretrained_checkpoints.sh
source create_datasets.sh
cd ./eval
source download_evalscript.sh
cd ..