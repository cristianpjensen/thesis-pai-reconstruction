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

cd $HOME/thesis-pat-reconstruction

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


for noise in 10 20 30 40 50
do
    name=$model-${data}${noise}db-loss_$loss;
	python3 main.py $name --model $model --channel-mults $channelmults --dropout 0.5 --no-ema --val-epochs $valepochs --epochs -1 --steps 6000 --batch-size 8 --data data/${data}/train/${noise}db.yaml --val-data data/${data}/validation/${noise}db.yaml --loss-type $loss;
    python3 report.py $name --model $model --checkpoint logs/${name}/version_0/checkpoints/best.ckpt --data data/${data}/test/${noise}db.yaml --batch-size 8;
    python3 report.py experimental-$name --model $model --checkpoint logs/${name}/version_0/checkpoints/best.ckpt --data data/${data}/experimental/${noise}db.yaml --batch-size 8;
done
