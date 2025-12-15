import keras
from keras import layers, regularizers

@keras.saving.register_keras_serializable()
class ConvBlock(layers.Layer):
    """Convolutional block with BatchNorm and optional residual connection."""
    
    def __init__(self, filters, kernel_size=3, strides=1, use_residual=False, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.use_residual = use_residual
        
        self.conv1 = layers.Conv2D(
            filters, kernel_size, strides=strides, padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            kernel_initializer='he_normal'
        )
        self.bn1 = layers.BatchNormalization()
        self.act1 = layers.ReLU()
        
        self.conv2 = layers.Conv2D(
            filters, kernel_size, padding='same',
            kernel_regularizer=regularizers.l2(1e-4),
            kernel_initializer='he_normal'
        )
        self.bn2 = layers.BatchNormalization()
        self.act2 = layers.ReLU()
        
        if use_residual:
            self.shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')
            self.shortcut_bn = layers.BatchNormalization()
    
    def call(self, inputs, training=False):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.act1(x)
        
        x = self.conv2(x)
        x = self.bn2(x, training=training)
        
        if self.use_residual:
            shortcut = self.shortcut(inputs)
            shortcut = self.shortcut_bn(shortcut, training=training)
            x = layers.add([x, shortcut])
        
        x = self.act2(x)
        return x

@keras.saving.register_keras_serializable()
class SEBlock(layers.Layer):
    """Squeeze-and-Excitation block for channel attention."""
    
    def __init__(self, filters, ratio=16, **kwargs):
        super().__init__(**kwargs)
        self.gap = layers.GlobalAveragePooling2D()
        self.dense1 = layers.Dense(filters // ratio, activation='relu')
        self.dense2 = layers.Dense(filters, activation='sigmoid')
        self.reshape = layers.Reshape((1, 1, filters))
    
    def call(self, inputs):
        x = self.gap(inputs)
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.reshape(x)
        return layers.multiply([inputs, x])