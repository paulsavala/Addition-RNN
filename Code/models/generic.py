from tensorflow.keras.models import Model, Sequential, load_model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

from pathlib import Path
from utils import file_io
from datetime import datetime


class GenericModel:
    '''
    A generic class for all models. Models which inherit from this gain the ability to keep notes, easily be saved
    and deleted, implement early stopping, etc.
    When defining a new model inheriting from this class be sure to specify the name. This name will be used when
    creating directories for saving.
    '''
    def __init__(self, name, version=1):
        self.name = name
        self.version = version

        self.model = None

        self.base_model_dir = Path(f'models/{self.name}')
        if not self.base_model_dir.exists(): self.base_model_dir.mkdir()
        self.model_dir = self.base_model_dir / Path(f'saved_models')
        if not self.model_dir.exists(): self.model_dir.mkdir()
        self.weights_dir = self.base_model_dir / Path('weights')
        if not self.weights_dir.exists(): self.weights_dir.mkdir()
        self.tensorboard_dir = self.base_model_dir / Path('tensorboard')
        if not self.tensorboard_dir.exists(): self.tensorboard_dir.mkdir()
        self.notes_dir = self.base_model_dir / Path('notes')
        if not self.notes_dir.exists(): self.notes_dir.mkdir()

        self.metrics_names = getattr(self.model, 'metrics_names', None)

    def build_model(self):
        raise NotImplementedError

    def train(self, X, y, epochs=1, early_stopping=False, validation_split=0.0,
                    optimizer=None, optimizer_params=dict(), loss=None, metrics=None):

        callbacks = [TensorBoard(log_dir=self.tensorboard_dir, histogram_freq=0, write_graph=True, write_images=True)]
        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=5))

        optimizer = optimizer(**optimizer_params)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

        self.model.fit(X, y,
                       epochs=epochs,
                       validation_split=validation_split,
                       callbacks=callbacks,
                       verbose=1)

    def save_model(self, notes=None, update_version=False):
        if update_version:
            self.version += 1

        try:
            weights_path = self.weights_dir / Path(f'v{self.version}.weights')
            self.model.save_weights(
                weights_path.as_posix()
            )
        except Exception as e:
            print('Error saving weights')
            print(e)
            raise
        
        try:
            model_path = self.model_dir / Path(f'v{self.version}.h5')
            self.model.save(
                model_path.as_posix()
            )
        except Exception as e:
            print('Error saving weights')
            print(e)
            raise

        if notes:
            version_notes_file = self.notes_dir / Path(f'v{self.version}.txt')
            global_notes_file = self.notes_dir / Path(f'version_notes.txt')

            formatted_notes = f'{"="*5} Version {self.version} ({datetime.now().strftime("%B %d, %Y - %H:%M:%S")}) {"="*5}'
            formatted_notes += '\n'
            formatted_notes += notes
            formatted_notes += '\n\n'

            file_io.append_or_write(version_notes_file, formatted_notes)
            file_io.append_or_write(global_notes_file, formatted_notes)

    def load_model(self, version):
        model_file = self.model_dir / Path(f'v{version}.h5')
        assert model_file.exists(), f'Model file does not exist for version {version} in directory {self.model_dir}'
        self.model = load_model(model_file.as_posix())

    def delete_model(self):
        file_io.delete_folder(self.base_model_dir)
