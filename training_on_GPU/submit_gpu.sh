#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J deeplearning
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 16GB of system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start --
##BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o gpu_%J.out
#BSUB -e gpu_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/10.2
#module swap cuda/10.2

#/appl/cuda/10.2/samples/NVIDIA_CUDA-10.2_Samples/bin/x86_64/linux/release/deviceQuery
echo  "Running script, activating venv"
source /zhome/91/a/164752/02456/venv/bin/activate
echo  "Pip version used:"
which pip3
#pip3 install -r requirements.txt
echo
echo  "All packages installed, running main.py"
echo
python3 main.py
