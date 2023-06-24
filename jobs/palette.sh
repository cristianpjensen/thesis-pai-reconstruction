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
		d)
			data=${OPTARG};
			case "${data}" in
				drive) valepochs="200";;
				nne) valepochs="10";;
			esac;;
		n) noise=${OPTARG};;
	esac
done

python3 main.py palette-${data}${noise}db --model palette --channel-mults 1,1,2,2,4,4 --attention-res 16,32 --dropout 0.1 --ema --val-epochs $valepochs --epochs -1 --steps 100000 --val-size 16 --batch-size 8 --input-dir data/${data}/train/${noise}db --target-dir data/${data}/train/ground_truth --learn-variance --schedule-type cosine;
