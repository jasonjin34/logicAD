# Makefile will include make test make clean make build make run 

# specify base python 3.10 path
# alternatively make one using:
# 	conda create -n python310 python=3.10

VENVNAME:= venv_temp
PYTHONBASE:= /work/scratch/$(USER)/bin/miniconda3/envs/python310/bin/python3

# specify desired location for adpy python binary 
VENV:= /work/scratch/$(USER)/miniconda3/envs/${VENVNAME}
PYTHON:= ${VENV}/bin/python

# clean automatic generated files
clean:
	find . | grep -E "(__pycache__|\.pyc|\.pyo)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -rf *.egg-info
	rm -rf ./logs/*

cleanvenv:
	rm -rf ${VENV}

sync:
	git pull
	git pull origin main

#test -d ${VENV} || python3 -m venv ${VENV}
$(VENV)/bin/activate: requirements.txt
	${PYTHONBASE} -m venv ${VENV}
	${PYTHON} -m pip install --upgrade pip
	${PYTHON} -m pip install -r requirements.txt
	${PYTHON} -m pip install -e .

install_torch:
	${PYTHON} -m pip install torchvision==0.12.0+cu113 torch==1.11.0+cu113 -i https://download.pytorch.org/whl/cu113

# check if the virtual environment is created or not, creating one
activate: ${VENV}/bin/activate

conda:
	. /home/staff/jin/miniconda3/bin/activate

test: activate
	$(PYTHON) -m unittest discover

run: activate
	$(PYTHON) adpy

ARGS+= task_name=debug
ARGS+= modules.optimizer.lr=0.0015
ARGS+= trainer.max_epochs=5

debug:
	$(PYTHON) -m customlib -cn train $(ARGS)
