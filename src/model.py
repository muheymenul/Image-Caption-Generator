"""
src/model.py
EfficientNetB3 (CNN encoder) + Transformer decoder for image captioning.
Backend: TensorFlow with Keras 3.
"""
import numpy as np
import keras
from keras import layers, ops


# ============================================================
# CNN encoder (frozen EfficientNetB3 -> spatial grid tokens)
# ============================================================
def build_cnn_encoder(image_size: int = 300) -> keras.Model:
    """Return a frozen EfficientNetB3 producing (batch, H*W, C) grid features.
    Keras EfficientNet normalizes internally, so inputs are raw [0, 255] pixels."""
    base = keras.applications.EfficientNetB3(
        include_top=False, weights="imagenet", input_shape=(image_size, image_size, 3)
    )
    base.trainable = False
    h, w, c = base.output_shape[1:]
    inputs = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = base(inputs, training=False)
    x = layers.Reshape((h * w, c), name="grid_tokens")(x)
    return keras.Model(inputs, x, name="efficientnetb3_encoder")


def sinusoidal_position_encoding(max_len: int, dim: int) -> np.ndarray:
    """Classic 'Attention Is All You Need' sin/cos positional encoding."""
    positions = np.arange(max_len)[:, None]
    dims = np.arange(dim)[None, :]
    angles = positions / np.power(10000.0, (2 * (dims // 2)) / dim)
    pe = np.zeros((max_len, dim), dtype=np.float32)
    pe[:, 0::2] = np.sin(angles[:, 0::2])
    pe[:, 1::2] = np.cos(angles[:, 1::2])
    return pe


# ============================================================
# Building blocks
# ============================================================
@keras.saving.register_keras_serializable(package="captioner")
class FeedForward(layers.Layer):
    def __init__(self, embed_dim, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.ff_dim, self.dropout_rate = embed_dim, ff_dim, dropout
        self.fc1 = layers.Dense(ff_dim, activation="gelu")
        self.fc2 = layers.Dense(embed_dim)
        self.drop = layers.Dropout(dropout)

    def call(self, x, training=None):
        return self.fc2(self.drop(self.fc1(x), training=training))

    def get_config(self):
        return {**super().get_config(), "embed_dim": self.embed_dim,
                "ff_dim": self.ff_dim, "dropout": self.dropout_rate}


@keras.saving.register_keras_serializable(package="captioner")
class TransformerEncoderBlock(layers.Layer):
    """Pre-LN self-attention block over image grid tokens."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.ff_dim, self.dropout_rate = ff_dim, dropout
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = layers.MultiHeadAttention(num_heads, key_dim=embed_dim // num_heads,
                                              dropout=dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, training=None):
        h = self.ln1(x)
        x = x + self.drop1(self.attn(h, h, training=training), training=training)
        x = x + self.drop2(self.ffn(self.ln2(x), training=training), training=training)
        return x

    def get_config(self):
        return {**super().get_config(), "embed_dim": self.embed_dim, "num_heads": self.num_heads,
                "ff_dim": self.ff_dim, "dropout": self.dropout_rate}


@keras.saving.register_keras_serializable(package="captioner")
class TransformerDecoderBlock(layers.Layer):
    """Pre-LN decoder block: causal self-attention -> cross-attention -> FFN."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim, self.num_heads = embed_dim, num_heads
        self.ff_dim, self.dropout_rate = ff_dim, dropout
        key_dim = embed_dim // num_heads
        self.ln1 = layers.LayerNormalization(epsilon=1e-6)
        self.self_attn = layers.MultiHeadAttention(num_heads, key_dim=key_dim, dropout=dropout)
        self.ln2 = layers.LayerNormalization(epsilon=1e-6)
        self.cross_attn = layers.MultiHeadAttention(num_heads, key_dim=key_dim, dropout=dropout)
        self.ln3 = layers.LayerNormalization(epsilon=1e-6)
        self.ffn = FeedForward(embed_dim, ff_dim, dropout)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)
        self.drop3 = layers.Dropout(dropout)

    def call(self, x, enc_out, training=None):
        # 1) Masked self-attention: each position sees only earlier tokens
        h = self.ln1(x)
        x = x + self.drop1(self.self_attn(h, h, use_causal_mask=True, training=training),
                           training=training)
        # 2) Cross-attention: text queries attend over image grid tokens
        h = self.ln2(x)
        x = x + self.drop2(self.cross_attn(h, enc_out, training=training), training=training)
        # 3) Position-wise feed-forward
        x = x + self.drop3(self.ffn(self.ln3(x), training=training), training=training)
        return x

    def get_config(self):
        return {**super().get_config(), "embed_dim": self.embed_dim, "num_heads": self.num_heads,
                "ff_dim": self.ff_dim, "dropout": self.dropout_rate}


# ============================================================
# Image encoder head & caption decoder
# ============================================================
@keras.saving.register_keras_serializable(package="captioner")
class ImageEncoder(layers.Layer):
    """Projects CNN grid features to embed_dim, adds learned 2D position, refines with self-attention."""

    def __init__(self, num_tokens, embed_dim, num_heads, ff_dim, num_layers=1, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.num_tokens, self.embed_dim, self.num_heads = num_tokens, embed_dim, num_heads
        self.ff_dim, self.num_layers, self.dropout_rate = ff_dim, num_layers, dropout
        self.proj = layers.Dense(embed_dim)
        self.proj_ln = layers.LayerNormalization(epsilon=1e-6)
        self.pos_emb = layers.Embedding(num_tokens, embed_dim)
        self.drop = layers.Dropout(dropout)
        self.blocks = [TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout)
                       for _ in range(num_layers)]
        self.final_ln = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=None):
        x = self.proj_ln(self.proj(x))
        x = x + ops.expand_dims(self.pos_emb(ops.arange(self.num_tokens)), 0)
        x = self.drop(x, training=training)
        for block in self.blocks:
            x = block(x, training=training)
        return self.final_ln(x)

    def get_config(self):
        return {**super().get_config(), "num_tokens": self.num_tokens, "embed_dim": self.embed_dim,
                "num_heads": self.num_heads, "ff_dim": self.ff_dim,
                "num_layers": self.num_layers, "dropout": self.dropout_rate}


@keras.saving.register_keras_serializable(package="captioner")
class CaptionDecoder(layers.Layer):
    """Token embedding + sinusoidal positional encoding + N decoder blocks + LM head."""

    def __init__(self, vocab_size, max_len, embed_dim, num_heads, ff_dim,
                 num_layers=3, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size, self.max_len, self.embed_dim = vocab_size, max_len, embed_dim
        self.num_heads, self.ff_dim = num_heads, ff_dim
        self.num_layers, self.dropout_rate = num_layers, dropout
        self.token_emb = layers.Embedding(vocab_size, embed_dim)
        self.pos_encoding = ops.convert_to_tensor(sinusoidal_position_encoding(max_len, embed_dim))
        self.scale = float(np.sqrt(embed_dim))
        self.drop = layers.Dropout(dropout)
        self.blocks = [TransformerDecoderBlock(embed_dim, num_heads, ff_dim, dropout)
                       for _ in range(num_layers)]
        self.final_ln = layers.LayerNormalization(epsilon=1e-6)
        self.lm_head = layers.Dense(vocab_size, name="lm_head")

    def call(self, tokens, enc_out, training=None):
        seq_len = ops.shape(tokens)[1]
        x = self.token_emb(tokens) * self.scale
        x = x + self.pos_encoding[None, :seq_len, :]
        x = self.drop(x, training=training)
        for block in self.blocks:
            x = block(x, enc_out, training=training)
        return self.lm_head(self.final_ln(x))          # logits: (batch, seq_len, vocab_size)

    def get_config(self):
        return {**super().get_config(), "vocab_size": self.vocab_size, "max_len": self.max_len,
                "embed_dim": self.embed_dim, "num_heads": self.num_heads, "ff_dim": self.ff_dim,
                "num_layers": self.num_layers, "dropout": self.dropout_rate}


# ============================================================
# Model factory: training model + shared-weight encoder/decoder for fast inference
# ============================================================
def build_caption_model(vocab_size, max_len, num_img_tokens=100, feat_dim=1536,
                        embed_dim=512, num_heads=8, ff_dim=2048,
                        num_decoder_layers=3, num_encoder_layers=1, dropout=0.3):
    image_encoder = ImageEncoder(num_img_tokens, embed_dim, num_heads, ff_dim,
                                 num_encoder_layers, dropout, name="image_encoder")
    decoder = CaptionDecoder(vocab_size, max_len, embed_dim, num_heads, ff_dim,
                             num_decoder_layers, dropout, name="caption_decoder")

    # Training graph: (features, tokens) -> logits
    feats_in = keras.Input(shape=(num_img_tokens, feat_dim), name="image_features")
    tokens_in = keras.Input(shape=(None,), dtype="int32", name="decoder_tokens")
    enc_out = image_encoder(feats_in)
    logits = decoder(tokens_in, enc_out)
    train_model = keras.Model([feats_in, tokens_in], logits, name="image_captioner")

    # Inference graphs (same layer objects -> same weights)
    encoder_model = keras.Model(feats_in, enc_out, name="captioner_encoder")
    enc_in = keras.Input(shape=(num_img_tokens, embed_dim), name="encoded_image")
    step_tokens_in = keras.Input(shape=(None,), dtype="int32", name="decoder_tokens_step")
    decoder_model = keras.Model([enc_in, step_tokens_in], decoder(step_tokens_in, enc_in),
                                name="captioner_decoder")
    return train_model, encoder_model, decoder_model


# ============================================================
# Masked loss & metric (ignore <pad> positions)
# ============================================================
def make_masked_loss(vocab_size: int, pad_id: int = 0, label_smoothing: float = 0.1):
    def masked_loss(y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        mask = ops.cast(ops.not_equal(y_true, pad_id), y_pred.dtype)
        one_hot = ops.one_hot(y_true, vocab_size)
        loss = keras.losses.categorical_crossentropy(
            one_hot, y_pred, from_logits=True, label_smoothing=label_smoothing
        )                                                   # (batch, seq_len)
        return ops.sum(loss * mask) / ops.maximum(ops.sum(mask), 1.0)
    return masked_loss


def make_masked_accuracy(pad_id: int = 0):
    def masked_accuracy(y_true, y_pred):
        y_true = ops.cast(y_true, "int32")
        y_hat = ops.cast(ops.argmax(y_pred, axis=-1), "int32")
        mask = ops.not_equal(y_true, pad_id)
        correct = ops.logical_and(ops.equal(y_true, y_hat), mask)
        return ops.sum(ops.cast(correct, "float32")) / ops.maximum(
            ops.sum(ops.cast(mask, "float32")), 1.0)
    return masked_accuracy
