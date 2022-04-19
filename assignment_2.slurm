#!/bin/bash
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=4
#SBATCH --gres-flags=enforce-binding
#SBATCH --time=32:00:00
#SBATCH --job-name=assignment2
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user akashdeep26@knights.ucf.edu

module load cuda/cuda-11.4
module load gcc/gcc-11.2.0
#python3 -m pip install --upgrade pip --user
#python3 -m pip install split-folders --user
#python3 -m pip install numpy --user
#python3 -m pip install scikit-learn --user
#python3 -m pip install monai --user
#python3 -m pip install matplotlib --user
#python3 -m pip install nibabel --user
#python3 -m pip install scikit-image --user
#python3 -m pip install tensorflow==2.6.2 --upgrade  --user
#python3 -m pip install pandas --user
#python3 -m pip install keras --user
#python3 -m pip install segmentation-models-3D --user
#python3 -m pip install medpy --user

#tar -xzf Task01_BrainTumor.tar.gz
#python3 a2_data_prep.py

#python3 a2_custom_datagen.py

#python3 a2_unet.py

#python3 a2_training.py

python3 met_eval.py