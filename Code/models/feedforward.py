from models.generic import GenericModel

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense


# Used mainly for testing and examples
class FeedForward(GenericModel):
    def __init__(self, n_layers, latent_units, input_shape=(2,), activation='relu', output_units=2):
        super()
        self.n_layers = n_layers
        if isinstance(latent_units, int):
            latent_units = [latent_units] * n_layers
        self.latent_units = latent_units
        self.input_shape = input_shape
        self.activation = activation
        self.output_units = output_units

    def build_model(self):
        model = Sequential()
        model.add(Input(shape=self.input_shape))
        for i in range(self.n_layers):
            model.add(Dense(self.latent_units[i], activation=self.activation))

        model.add(Dense(self.output_units, activation='softmax'))
        self.model = model