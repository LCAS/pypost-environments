#Install simulators written in C
cd src/pypostEnvironments/dynamicalSystem/forwardModels
python3 installForwardModels.py install $1
#Install environment
cd ../../../..
python3 setup.py install $1

