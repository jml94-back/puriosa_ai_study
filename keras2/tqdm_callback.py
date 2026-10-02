from tqdm.auto import tqdm
import tensorflow as tf


class TQDMProgress(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.pbar = tqdm(
            total=self.params["epochs"],
            desc="Training",
            unit="epoch"
        )

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # 현재 에포크의 주요 지표 표시
        postfix = {
            key: f"{value:.4f}"
            for key, value in logs.items()
            if isinstance(value, (int, float))
        }

        self.pbar.set_postfix(postfix)
        self.pbar.update(1)

    def on_train_end(self, logs=None):
        self.pbar.close()


