import os
import joblib
import json
import logging
import tensorflow as tf
from tensorflow.keras import layers

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# CUSTOM LAYER + LOSS
# ─────────────────────────────────────────────────────────────
class AttentionLayer(layers.Layer):
    def __init__(self, attention_dim=32, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        self.attention_dim = attention_dim

    def build(self, input_shape):
        self.W = self.add_weight(
            name="attention_weight",
            shape=(input_shape[-1], self.attention_dim),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.U = self.add_weight(
            name="attention_context",
            shape=(self.attention_dim, 1),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            name="attention_bias",
            shape=(self.attention_dim,),
            initializer="zeros",
            trainable=True,
        )
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        uit = tf.tanh(tf.tensordot(x, self.W, axes=1) + self.b)
        ait = tf.tensordot(uit, self.U, axes=1)
        ait = tf.squeeze(ait, -1)
        ait = tf.nn.softmax(ait, axis=-1)
        ait = tf.expand_dims(ait, -1)
        weighted = x * ait
        return tf.reduce_sum(weighted, axis=1)

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


# ─────────────────────────────────────────────────────────────
# STATIC MODEL DISCOVERY
# ─────────────────────────────────────────────────────────────
def get_available_static_models(repo_root):
    static_models = {}

    primary = os.path.join(repo_root, "notebooks", "nsl_model_v2.pkl")
    if os.path.exists(primary):
        static_models["nsl_model_v2"] = primary

    models_dir = os.path.join(repo_root, "models")
    if os.path.exists(models_dir):
        try:
            for f in os.listdir(models_dir):
                if f.startswith("model_") and f.endswith(".pkl"):
                    name = f.replace("model_", "").replace(".pkl", "").title()
                    static_models[name] = os.path.join(models_dir, f)
        except Exception as e:
            logger.warning(f"Error scanning static models dir: {e}")

    return static_models


# ─────────────────────────────────────────────────────────────
# DYNAMIC MODEL DISCOVERY  — FIXED
# ─────────────────────────────────────────────────────────────
def get_available_dynamic_models(repo_root):
    """
    Scans  <repo_root>/models/final/  for .h5 and .keras files.

    Fixes vs original:
    - Uses os.path.join() for cross-platform paths (no backslash issues on Windows)
    - Scans both .h5 AND .keras extensions
    - "best_model.h5 / best_model.keras" is registered once as "best_model (Default)"
      and NOT added a second time as plain "best_model" by the loop
    - Alias generation cleaned up so dict stays predictable
    - Returns {} (empty) if directory not found instead of silently missing models
    """
    # FIX: use os.path.join — never string-concatenate paths
    models_dir = os.path.join(repo_root, "models", "final")
    dynamic_models = {}

    if not os.path.exists(models_dir):
        logger.warning(
            f"Dynamic models directory not found: {models_dir}\n"
            f"Expected layout: <repo_root>/models/final/*.h5  or  *.keras"
        )
        return dynamic_models

    # Accepted extensions in priority order
    accepted_ext = (".keras", ".h5")

    # FIX: collect files first so we can de-duplicate before building the dict
    found_files = []
    try:
        for f in sorted(os.listdir(models_dir)):          # sorted = deterministic order
            if any(f.endswith(ext) for ext in accepted_ext):
                found_files.append(os.path.join(models_dir, f))
    except Exception as e:
        logger.error(f"Cannot list models dir {models_dir}: {e}")
        return dynamic_models

    # Track canonical stems already registered to avoid duplicates
    registered_stems = set()

    for model_path in found_files:
        filename = os.path.basename(model_path)

        # Strip extension(s) — handles both .h5 and .keras
        stem = filename
        for ext in accepted_ext:
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break

        # FIX: register "best_model" as "(Default)" ONCE, skip on second encounter
        if stem.lower() == "best_model":
            key = "best_model (Default)"
            if key not in dynamic_models:
                dynamic_models[key] = model_path
            registered_stems.add(stem.lower())
            continue

        # Skip stems already registered (e.g. both best_model.h5 and best_model.keras)
        if stem.lower() in registered_stems:
            continue
        registered_stems.add(stem.lower())

        # Primary key = stem as-is (e.g. "GRU_best", "GRU")
        dynamic_models.setdefault(stem, model_path)

        # Convenience aliases
        # "GRU_best" → alias "GRU" (strip trailing _best / -best)
        for suffix in ("_best", "-best", "_BEST", "-BEST"):
            if stem.endswith(suffix):
                alias = stem[: -len(suffix)]
                if alias:
                    dynamic_models.setdefault(alias, model_path)

        # UPPERCASE alias for legacy lookups
        dynamic_models.setdefault(stem.upper(), model_path)

    if not dynamic_models:
        logger.warning(
            f"No .h5 or .keras models found in {models_dir}. "
            f"Files present: {os.listdir(models_dir)}"
        )

    return dynamic_models


# ─────────────────────────────────────────────────────────────
# STATIC MODEL LOADING
# ─────────────────────────────────────────────────────────────
def load_selected_static_model(repo_root, model_name):
    available = get_available_static_models(repo_root)
    if model_name not in available:
        logger.error(f"Static model '{model_name}' not in available: {list(available)}")
        return None, None

    model_path = available[model_name]
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logger.error(f"Failed to joblib.load '{model_path}': {e}")
        return None, None

    scaler = None
    scaler_path = os.path.join(repo_root, "models", "scaler_static.pkl")
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            logger.warning(f"Scaler load failed (non-fatal): {e}")

    return model, scaler


# ─────────────────────────────────────────────────────────────
# DYNAMIC MODEL LOADING  — FIXED
# ─────────────────────────────────────────────────────────────
def load_selected_dynamic_model(repo_root, model_name):
    """
    Fixes vs original:
    - Bare except → specific Exception logging so errors are visible
    - Flexible name resolution kept but now logs what it resolves to
    - Tries .keras format first (TF2.12+ default), falls back to .h5
    - compile=False avoids loss-function conflicts on load
    - Re-compiles after load so the model is usable
    - Config JSON looked up via both .keras and .h5 stem variants
    """
    available = get_available_dynamic_models(repo_root)

    # ── Resolve model_name → file path ──────────────────────
    model_path = None

    if model_name in available:
        model_path = available[model_name]
    else:
        # Case-insensitive fallback
        lookup = {k.lower(): v for k, v in available.items()}
        normalized = model_name.replace("(Default)", "").strip().lower()

        for candidate in (model_name.lower(), normalized):
            if candidate in lookup:
                model_path = lookup[candidate]
                logger.info(f"Resolved '{model_name}' → '{model_path}' via case-insensitive match")
                break

        if model_path is None:
            # Substring match as last resort
            for k, v in available.items():
                if model_name.lower() in k.lower():
                    model_path = v
                    logger.info(f"Resolved '{model_name}' → '{v}' via substring match")
                    break

    if model_path is None:
        logger.error(
            f"Dynamic model '{model_name}' could not be resolved.\n"
            f"Available keys: {list(available.keys())}\n"
            f"Models dir: {os.path.join(repo_root, 'models', 'final')}"
        )
        return None, None

    # FIX: if the resolved path doesn't exist on disk, try swapping extension
    if not os.path.exists(model_path):
        stem = model_path
        for ext in (".h5", ".keras"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        for ext in (".keras", ".h5"):          # try .keras first
            alt = stem + ext
            if os.path.exists(alt):
                logger.info(f"Path '{model_path}' not found; using '{alt}' instead")
                model_path = alt
                break
        else:
            logger.error(f"Model file not found at '{model_path}' (or any extension variant)")
            return None, None

    # ── Load model ───────────────────────────────────────────
    custom_objects = {
        "AttentionLayer": AttentionLayer,
        "label_smoothing_loss": label_smoothing_loss,
    }

    dynamic_model = None
    load_error = None

    # FIX: try loading with compile=False to avoid custom-loss conflicts
    try:
        dynamic_model = tf.keras.models.load_model(
            model_path,
            custom_objects=custom_objects,
            compile=False,
        )
        logger.info(f"Loaded model from '{model_path}'")
    except Exception as e:
        load_error = e
        logger.error(f"tf.keras.models.load_model failed for '{model_path}': {e}")

    # FIX: if .h5 failed, try the .keras variant (and vice versa)
    if dynamic_model is None:
        stem = model_path
        for ext in (".h5", ".keras"):
            if stem.endswith(ext):
                stem = stem[: -len(ext)]
                break
        for ext in (".keras", ".h5"):
            alt = stem + ext
            if alt != model_path and os.path.exists(alt):
                try:
                    dynamic_model = tf.keras.models.load_model(
                        alt,
                        custom_objects=custom_objects,
                        compile=False,
                    )
                    logger.info(f"Loaded model from fallback path '{alt}'")
                    model_path = alt        # update so config JSON lookup uses correct stem
                    load_error = None
                    break
                except Exception as e2:
                    logger.error(f"Fallback load also failed for '{alt}': {e2}")

    if dynamic_model is None:
        return None, None

    # Re-compile after compile=False load
    try:
        dynamic_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
    except Exception as e:
        logger.warning(f"Re-compile failed (non-fatal, model still usable): {e}")

    # ── Load config JSON ─────────────────────────────────────
    config = None
    stem = model_path
    for ext in (".h5", ".keras"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break

    for config_path in (stem + "_config.json", stem + ".json"):
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    config = json.load(f)
                logger.info(f"Loaded config from '{config_path}'")
            except Exception as e:
                logger.warning(f"Config JSON parse error at '{config_path}': {e}")
            break

    if config is None:
        logger.info(
            f"No config JSON found for '{model_path}'. "
            f"Class names will fall back to index-based labels."
        )

    return dynamic_model, config