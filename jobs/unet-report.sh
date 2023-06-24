#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00

module purge
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

cd $HOME/thesis-pat-reconstruction

while getopts m:d:l: flag
do
	case "${flag}" in
		m) model=${OPTARG};;
		d) data=${OPTARG};;
		l) loss=${OPTARG};;
	esac
done

for noise in 10 20 30 40 50
do
	python3 report.py $model-$data${noise}db-loss_$loss --model $model --checkpoint logs/$model-$data${noise}db-loss_$loss/version_0/checkpoints/best.ckpt --batch-size 8 --input-dir data/${data}/test/${noise}db --target-dir data/${data}/test/ground_truth
done
