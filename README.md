# Attention for deep photoacoustic tomography reconstruction

Required directories:
 - `output/` contains the training data output, i.e. loss and metrics over
   epochs during training.
 - `checkpoints/` contains the checkpoints of the various models, so they can
   be used later. Their file names describe what model they are.
 - `data/` contains all data. The data should be structured such that `data/`
   contains directories that contain both the input and label directories
   containing those images.

`models/` contains all model architectures. If you want to use a custom model,
add its generator and discriminator in the dictionaries in `architectures.py`.

To train a model, just run
```
python main.py
```
Then, you get prompted what model you want to run and some more specific
information about the training loop.
