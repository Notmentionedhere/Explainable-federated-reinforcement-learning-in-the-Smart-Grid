Computing Infrastructure
CPU: Intel Core i7
GPU: NVIDIA GTX TITAN X
Memory: 64 GB RAM
Operating System: Ubuntu 20.04 LTS and Win 10
Software Libraries and Frameworks: The experiments were conducted using the following libraries and frameworks:
TensorFlow version 1.12.0 
CUDA 10.0 and cuDNN 7.5
Python version 3.7.7

Install pip packages (Anaconda recommended)

    pip install -r requirements.txt

1.  Traning agents of FRL

python3 BESS/BESS_FRL/main.py train

2.  Traning agents of FRL-ESP

python3  BESS/ESP_grid/main.py train
