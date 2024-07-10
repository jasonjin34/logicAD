# Makefile will include make test make clean make build make run 
include .env

# clean automatic generated files
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf *.egg-info
	rm -rf ./logs/*

sync:
	git pull
	git pull origin main

install_torch:
	${PYTHON} -m pip install torchvision==0.12.0+cu113 torch==1.11.0+cu113 -i https://download.pytorch.org/whl/cu113

# init submodules
submodules:
	git submodule update --init --recursive
	pip install -r GroundingDINO/

test:
	${PYTEST} -s