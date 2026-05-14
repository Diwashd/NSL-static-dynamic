"""
=========================================================
Nepali Sign Language Recognition Models (IMPROVED)

GRU vs BiGRU vs BiGRU + Attention
Optimized for small-to-medium datasets

Author: Diwash
=========================================================
"""

import tensorflow as tf
from tensorflow.keras import layers, models

# =========================================================
# 1. BASELINE GRU (REDUCED SIZE)
# =========================================================

def create_gru_model(input_shape, num_classes, dropout=0.3, units=[64, 32]):

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.GRU(
            units[0],
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.2
        ),
        layers.Dropout(dropout),

        layers.GRU(
            units[1],
            return_sequences=False,
            dropout=dropout,
            recurrent_dropout=0.2
        ),
        layers.Dropout(dropout),

        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),

        layers.Dense(num_classes, activation="softmax")
    ])

    return model


# =========================================================
# 2. BiGRU MODEL (STABLE VERSION)
# =========================================================

def create_bigru_model(input_shape, num_classes, dropout=0.3, units=[64, 32]):

    model = models.Sequential([
        layers.Input(shape=input_shape),

        layers.Bidirectional(
            layers.GRU(
                units[0],
                return_sequences=True,
                dropout=dropout,
                recurrent_dropout=0.2
            )
        ),
        layers.Dropout(dropout),

        layers.Bidirectional(
            layers.GRU(
                units[1],
                return_sequences=False,
                dropout=dropout,
                recurrent_dropout=0.2
            )
        ),
        layers.Dropout(dropout),

        layers.Dense(64, activation="relu"),
        layers.Dropout(dropout),

        layers.Dense(num_classes, activation="softmax")
    ])

    return model


# =========================================================
# 3. IMPROVED ATTENTION LAYER
# =========================================================

class AttentionLayer(layers.Layer):
    """
    Attention mechanism with 3 weights (matches saved models)
    """

    def build(self, input_shape):
        # 3 weights to match saved models
        self.W = self.add_weight(
            shape=(input_shape[-1], 64),
            initializer="glorot_uniform",
            trainable=True
        )

        self.b = self.add_weight(
            shape=(64,),
            initializer="zeros",
            trainable=True
        )

        self.V = self.add_weight(
            shape=(64, 1),
            initializer="glorot_uniform",
            trainable=True
        )

    def call(self, x):
        # x shape: (batch, time, features)
        score = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        attention = tf.tensordot(score, self.V, axes=1)
        weights = tf.nn.softmax(attention, axis=1)
        context = x * weights
        context = tf.reduce_sum(context, axis=1)
        return context


# =========================================================
# 4. BiGRU + ATTENTION (BEST MODEL)
# =========================================================

def create_bigru_attention_model(input_shape, num_classes, dropout=0.3, units=[64, 32]):

    inputs = layers.Input(shape=input_shape)

    x = layers.Bidirectional(
        layers.GRU(
            units[0],
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.2
        )
    )(inputs)

    x = layers.Dropout(dropout)(x)

    x = layers.Bidirectional(
        layers.GRU(
            units[1],
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.2
        )
    )(x)

    x = layers.Dropout(dropout)(x)

    x = AttentionLayer()(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(dropout)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return models.Model(inputs, outputs)


## =========================================================
# 5. LOSS FUNCTION (FIXED + COMPATIBLE)
# =========================================================


class LabelSmoothingLoss(tf.keras.losses.Loss):
    """
    Custom label smoothing for sparse labels.
    Works in all TF versions.
    """

    def __init__(self, num_classes=None, smoothing=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.smoothing = smoothing

    def call(self, y_true, y_pred):
        # Infer num_classes from prediction shape if not provided
        if self.num_classes is None:
            num_classes = tf.shape(y_pred)[-1]
        else:
            num_classes = self.num_classes
            
        # Convert labels → one-hot
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)

        # Apply smoothing
        y_true = y_true * (1 - self.smoothing) + (self.smoothing / tf.cast(num_classes, tf.float32))

        # Compute loss
        return tf.keras.losses.categorical_crossentropy(y_true, y_pred)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'smoothing': self.smoothing
        })
        return config


class TemporalConsistencyLoss(tf.keras.losses.Loss):
    def __init__(self, num_classes=None, label_smoothing=0.1, temporal_weight=0.05, **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.temporal_weight = temporal_weight
        self.label_smoothing = label_smoothing
        self.ce = tf.keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        )

    def call(self, y_true, y_pred):
        if self.num_classes is None:
            # Infer from shape if not provided
            num_classes = tf.shape(y_pred)[-1]
        else:
            num_classes = self.num_classes
            
        y_true = tf.one_hot(tf.cast(y_true, tf.int32), depth=num_classes)
        ce_loss = self.ce(y_true, y_pred)
        temporal_loss = tf.reduce_mean(tf.square(y_pred[1:] - y_pred[:-1]))
        return ce_loss + (self.temporal_weight * temporal_loss)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_classes': self.num_classes,
            'label_smoothing': self.label_smoothing,
            'temporal_weight': self.temporal_weight
        })
        return config


def get_loss(num_classes, label_smoothing=0.1):
    return LabelSmoothingLoss(num_classes=num_classes, smoothing=label_smoothing)

# =========================================================
# 6. MODEL BUILDER (FOR COMPARISON)
# =========================================================

def build_all_models(input_shape, num_classes):

    return {
        "GRU": create_gru_model(input_shape, num_classes),
        "BiGRU": create_bigru_model(input_shape, num_classes),
        "BiGRU_Attention": create_bigru_attention_model(input_shape, num_classes)
    }


# =========================================================
# 7. TRAINING CONFIG
# =========================================================

CONFIG = {
    "epochs": 120,
    "batch_size": 16,
    "learning_rate": 0.001,
    "dropout": 0.3,
    "sequence_length": 40
}


def get_training_config(num_classes):
    config = CONFIG.copy()
    config["num_classes"] = num_classes
    return config