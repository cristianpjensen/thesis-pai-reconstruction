#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --gpus=1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

module purge
module load 2022
module load PyTorch/1.12.0-foss-2022a-CUDA-11.7.0
module load torchvision/0.13.1-foss-2022a-CUDA-11.7.0

while getopts m:d:l: flag
do
	case "${flag}" in
		m)
			model=${OPTARG};
			case "${model}" in
				pix2pix) channelmults="1,2,4,8,8,8,8,8";;
				attention_unet) channelmults="1,2,4,8,8,8,8,8";;
				res18_unet) channelmults="1,2,4,8,8,8,8,8";;
				res50_unet) channelmults="1,2,4,8,8,8,8,8";;
				resv2_unet) channelmults="1,2,4,8,8,8,8,8";;
				resnext_unet) channelmults="1,2,4,8,8,8,8,8";;
				trans_unet) channelmults="1,2,2,4,4";;
			esac;;
		d)
			data=${OPTARG};
			case "${data}" in
				drive) valepochs="10";;
				nne) valepochs="1";;
			esac;;
		l) loss=${OPTARG};;
	esac
done

DIR=$HOME/thesis-pat-reconstruction;

cd /scratch-shared/$USER

name=$model-${data}_all-loss_$loss;

python3 $DIR/main.py $name --model $model --channel-mults $channelmults --dropout 0.5 --no-ema --val-epochs $valepochs --epochs -1 --steps 30000 --batch-size 8 --data data/${data}/train/all.yaml --val-data data/${data}/validation/all.yaml --loss-type $loss;

python3 $DIR/report.py $name --model $model --checkpoint logs/${name}/version_0/checkpoints/best.ckpt --data data/${data}/test/all.yaml --batch-size 8;

python3 $DIR/report.py experimental-$name --model $model --checkpoint logs/${name}/version_0/checkpoints/best.ckpt --data data/${data}/experimental/data.yaml --batch-size 8;
done
