# Deep learning for deep photoacoustic image reconstruction

Required directories:
 - `data/` contains all data. The data should be structured such that `data/`
   contains directories that contain both the input and label directories
   containing those images.


Example of using the main script for training the conditional diffusion model:
```bash
python3 main.py palette-drive10db --model palette --schedule-type cosine --channel-mults 1,1,2,2,4,4 --attention-mults 16,8 --dropout 0.2 --ema --val-epochs 100 --epochs -1 --steps 200000 --val-size 8 --batch-size 8 --input-dir data/drive/train/10db --target-dir data/drive/train/ground_truth --learn-variance
```
