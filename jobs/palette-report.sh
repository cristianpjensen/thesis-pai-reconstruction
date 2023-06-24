#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=24:00:00

module purge
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

cd $HOME/thesis-pat-reconstruction

while getopts d:n: flag
do
	case "${flag}" in
		d) data=${OPTARG};;
		n) noise=${OPTARG};;
	esac
done

python3 report.py palette-$data${noise}db --model palette --checkpoint logs/palette-$data${noise}db/version_0/checkpoints/best.ckpt --batch-size 64 --input-dir data/${data}/test/${noise}db --target-dir data/${data}/test/ground_truth
