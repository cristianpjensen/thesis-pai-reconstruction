import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage


class EMACallback(pl.callbacks.Callback):
    """
    Exponential Moving Average callback to be used with any pytorch lightning
    module.

    """

    def __init__(self, decay=0.9999):
        self.decay = decay
        self.ema = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule):
        """Initialize EMA."""

        self.ema = ExponentialMovingAverage(
            pl_module.parameters(),
            decay=self.decay,
        )

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        *args,
        **kwargs,
    ):
        """Update the stored parameters using a moving average."""

        self.ema.update()

    def on_validation_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """Do validation using the stored parameters."""

        self.ema.store()
        self.ema.copy_to()

    def on_validation_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ):
        """Restore original parameters to resume training later."""

        self.ema.restore()

    def on_save_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        checkpoint: dict[str, any],
    ):
        """Save state dict on checkpoint."""

        return self.ema.state_dict()

    def on_load_checkpoint(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        callback_state: dict[str, any],
    ):
        """Load state dict on checkpoint."""

        self.ema.load_state_dict(callback_state)
