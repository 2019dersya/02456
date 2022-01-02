#!/bin/sh 
### General options 
### -- specify queue -- 
#BSUB -q hpc
### -- set the job Name -- 
#BSUB -J jobjob
### -- ask for number of cores (default: 1) -- 
#BSUB -n 16 
### -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
### -- specify that we need 2GB of memory per core/slot -- 
#BSUB -R "rusage[mem=2GB]"
### -- specify that we want the job to get killed if it exceeds 3 GB per core/slot -- 
#BSUB -M 3GB
### -- set walltime limit: hh:mm -- 
#BSUB -W 48:00 
### -- set the email address -- 
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u your_email_address
### -- send notification at start -- 
##BSUB -B 
### -- send notification at completion -- 
#BSUB -N 
### -- Specify the output and error file. %J is the job-id -- 
### -- -o and -e mean append, -oo and -eo mean overwrite -- 
#BSUB -o Output_%J.out 
#BSUB -e Error_%J.err

# here follow the commands you want to execute 
#myapplication.x < input.in > output.out

echo  "Running script, activating venv"
source /zhome/91/a/164752/02456/venv/bin/activate
echo  "Pip version used:"
which pip3
#pip3 install -r requirements.txt
echo
echo  "All packages installed, running main2.py"
echo
/zhome/91/a/164752/02456/venv/bin/python3 main2.py
