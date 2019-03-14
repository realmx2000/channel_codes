from tensorflow.keras.models import load_model


class ModelSaver(object):
    """Class to save tensorflow sessions"""

    def __init__(self, save_dir, logger):
        """Init saver/loader"""
        self.model_save_dir = save_dir
        self.logger = logger


    def save(self, model):
        model.trainable_encoder.save(self.model_save_dir / "encoder.h5")
        model.trainable_decoder.save(self.model_save_dir / "decoder.h5")
        self.logger.write(f"Saved model to: {self.model_save_dir}")


    def load(self, model):
        model.trainable_encoder.load_weights(str(self.model_save_dir / "encoder.h5"))
        model.trainable_decoder.load_weights(str(self.model_save_dir / "decoder.h5"))
