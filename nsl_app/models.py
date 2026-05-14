import os
import joblib
import json
import tensorflow as tf
from tensorflow.keras import layers


class AttentionLayer(layers.Layer):
    def __init__(self, attention_dim=32, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(name="attention_weight", shape=(input_shape[-1], self.attention_dim), initializer="glorot_uniform", trainable=True)
        self.U = self.add_weight(name="attention_context", shape=(self.attention_dim, 1), initializer="glorot_uniform", trainable=True)
        self.b = self.add_weight(name="attention_bias", shape=(self.attention_dim,), initializer="zeros", trainable=True)
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.U, axes=1)
        ait = tf.squeeze(ait, -1)
        ait = tf.nn.softmax(ait, axis=-1)
        ait = tf.expand_dims(ait, -1)
        weighted = x * ait
        output = tf.reduce_sum(weighted, axis=1)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"attention_dim": self.attention_dim})
        return config


@tf.keras.utils.register_keras_serializable()
def label_smoothing_loss(y_true, y_pred, smoothing=0.1):
    num_classes = tf.cast(tf.shape(y_pred)[-1], tf.float32)
    y_true = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true, tf.cast(num_classes, tf.int32))
    y_true_smoothed = y_true_one_hot * (1.0 - smoothing) + (smoothing / num_classes)
    return tf.keras.losses.categorical_crossentropy(y_true_smoothed, y_pred)


def get_available_static_models(repo_root):
    static_models = {}
    primary_model = os.path.join(repo_root, "notebooks", "nsl_model_v2.pkl")
    if os.path.exists(primary_model):
        static_models["nsl_model_v2"] = primary_model
    models_dir = os.path.join(repo_root, "models")
    if os.path.exists(models_dir):
        try:
            for f in os.listdir(models_dir):
                if f.startswith("model_") and f.endswith(".pkl"):
                    model_path = os.path.join(models_dir, f)
                    name = f.replace("model_", "").replace(".pkl", "").title()
                    static_models[name] = model_path
        except:
            pass
    return static_models


def get_available_dynamic_models(repo_root):
    models_dir = os.path.join(repo_root, "models", "final")
    dynamic_models = {}
    if os.path.exists(models_dir):
        try:
            best_path = os.path.join(models_dir, "best_model.h5")
            if os.path.exists(best_path):
                dynamic_models["best_model (Default)"] = best_path
            for f in os.listdir(models_dir):
                if not f.endswith('.h5'):
                    continue
                model_path = os.path.join(models_dir, f)
                name = f.replace(".h5", "")
                dynamic_models[name] = model_path
                base = name
                for sep in ['_best', '-best', 'best']:
                    if base.lower().endswith(sep):
                        alias = base[: -len(sep)]
                        if alias:
                            dynamic_models.setdefault(alias, model_path)
                dynamic_models.setdefault(name.upper(), model_path)
        except:
            pass
    return dynamic_models


def load_selected_static_model(repo_root, model_name):
    available = get_available_static_models(repo_root)
    if model_name not in available:
        return None, None
    model_path = available[model_name]
    try:
        model = joblib.load(model_path)
        scaler = None
        scaler_path = os.path.join(repo_root, "models", "scaler_static.pkl")
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        return model, scaler
    except Exception:
        return None, None


def load_selected_dynamic_model(repo_root, model_name):
    available = get_available_dynamic_models(repo_root)
    # Resolve model_path with flexible matching
    if model_name not in available:
        lookup = {k.lower(): v for k, v in available.items()}
        if model_name.lower() in lookup:
            model_path = lookup[model_name.lower()]
        else:
            normalized = model_name.replace("(Default)", "").strip().lower()
            if normalized in lookup:
                model_path = lookup[normalized]
            else:
                found = None
                for k, v in available.items():
                    if model_name.lower() in k.lower():
                        found = v
                        break
                if found:
                    model_path = found
                else:
                    return None, None
    else:
        model_path = available[model_name]

    try:
        custom_objects = {"AttentionLayer": AttentionLayer, "label_smoothing_loss": label_smoothing_loss}
        dynamic_model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
        dynamic_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        config_path = model_path.replace('.h5', '_config.json')
        config = None
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        return dynamic_model, config
    except Exception:
        return None, None
