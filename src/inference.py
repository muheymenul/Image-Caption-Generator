"""
src/inference.py
Vocabulary, image preprocessing and Beam Search caption generation.
Framework-light: only needs TensorFlow/Keras + the saved artifacts:
    saved_models/vocab.json, saved_models/model_config.json, saved_models/caption_model.weights.h5
"""
import json
import os
from typing import Dict, List, Sequence, Union

import numpy as np
import tensorflow as tf

from .model import build_caption_model, build_cnn_encoder

ImageInput = Union[bytes, bytearray, str, np.ndarray]


# ============================================================
# Vocabulary (identical logic to training)
# ============================================================
class Vocabulary:
    PAD, START, END, UNK = "<pad>", "<start>", "<end>", "<unk>"
    SPECIAL_TOKENS = [PAD, START, END, UNK]

    def __init__(self, idx2word: List[str], max_len: int):
        self.idx2word, self.max_len = idx2word, max_len
        self.word2idx = {w: i for i, w in enumerate(idx2word)}
        self.pad_id, self.start_id = self.word2idx[self.PAD], self.word2idx[self.START]
        self.end_id, self.unk_id = self.word2idx[self.END], self.word2idx[self.UNK]

    def __len__(self) -> int:
        return len(self.idx2word)

    def encode(self, caption: str) -> List[int]:
        ids = [self.word2idx.get(w, self.unk_id) for w in caption.split()]
        ids = [self.start_id] + ids[: self.max_len - 2] + [self.end_id]
        return ids + [self.pad_id] * (self.max_len - len(ids))

    def decode(self, ids: Sequence[int]) -> str:
        words = []
        for i in map(int, ids):
            if i == self.end_id:
                break
            if i not in (self.pad_id, self.start_id):
                words.append(self.idx2word[i])
        return " ".join(words)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"idx2word": self.idx2word, "max_len": self.max_len}, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "Vocabulary":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data["idx2word"], data["max_len"])


# ============================================================
# Caption generator
# ============================================================
class CaptionGenerator:
    """Load artifacts once, then caption images (raw bytes / path / array) or precomputed features."""

    def __init__(self, model_dir: str, weights_file: str = "caption_model.weights.h5",
                 load_cnn: bool = True):
        with open(os.path.join(model_dir, "model_config.json")) as f:
            arch = json.load(f)
        self.image_size = int(arch.pop("image_size"))
        self.trained_on = arch.pop("dataset", "unknown")
        self.arch = arch

        self.vocab = Vocabulary.load(os.path.join(model_dir, "vocab.json"))
        if len(self.vocab) != arch["vocab_size"]:
            raise ValueError("vocab.json does not match model_config.json (vocab_size mismatch).")

        self.train_model, self.encoder_model, self.decoder_model = build_caption_model(**arch)
        self.train_model.load_weights(os.path.join(model_dir, weights_file))

        n_tok, feat_dim, emb = arch["num_img_tokens"], arch["feat_dim"], arch["embed_dim"]
        self._encode_step = tf.function(
            lambda f: self.encoder_model(f, training=False),
            input_signature=[tf.TensorSpec([None, n_tok, feat_dim], tf.float32)],
        )
        self._decode_step = tf.function(
            lambda enc, tok: self.decoder_model([enc, tok], training=False),
            input_signature=[tf.TensorSpec([None, n_tok, emb], tf.float32),
                             tf.TensorSpec([None, None], tf.int32)],
        )

        self.cnn = None
        if load_cnn:
            self.cnn = build_cnn_encoder(self.image_size)
            self._cnn_step = tf.function(
                lambda x: self.cnn(x, training=False),
                input_signature=[tf.TensorSpec([None, self.image_size, self.image_size, 3], tf.float32)],
            )

        v = self.vocab
        self.banned_ids = [v.pad_id, v.start_id, v.unk_id]

    # ---------------- Preprocessing ----------------
    def preprocess(self, image: ImageInput) -> tf.Tensor:
        """Return a (1, S, S, 3) float32 tensor in [0, 255] (EfficientNet normalizes internally)."""
        if isinstance(image, (bytes, bytearray)):
            img = tf.io.decode_image(bytes(image), channels=3, expand_animations=False)
        elif isinstance(image, str):
            img = tf.io.decode_image(tf.io.read_file(image), channels=3, expand_animations=False)
        elif isinstance(image, np.ndarray):
            img = tf.convert_to_tensor(image)
            if img.shape.rank == 2:
                img = tf.stack([img] * 3, axis=-1)
            img = img[..., :3]
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        img = tf.image.resize(tf.cast(img, tf.float32), (self.image_size, self.image_size))
        return img[None]

    def extract_features(self, image: ImageInput) -> tf.Tensor:
        if self.cnn is None:
            raise RuntimeError("CNN encoder not loaded. Use CaptionGenerator(..., load_cnn=True).")
        feats = self._cnn_step(self.preprocess(image))
        return tf.cast(tf.cast(feats, tf.float16), tf.float32)    # match float16 training cache

    # ---------------- Beam Search ----------------
    @staticmethod
    def _length_normalize(log_prob: float, length: int, alpha: float) -> float:
        return log_prob / (max(length, 1) ** alpha)

    @staticmethod
    def _banned_ngram_tokens(seq: List[int], n: int) -> List[int]:
        """Tokens that would complete an n-gram already present in seq."""
        if n <= 0 or len(seq) < n:
            return []
        prefix = tuple(seq[-(n - 1):]) if n > 1 else ()
        return [seq[i + n - 1] for i in range(len(seq) - n + 1)
                if tuple(seq[i:i + n - 1]) == prefix]

    def beam_search(self, enc: tf.Tensor, beam_width: int = 3, max_len: int = None,
                    length_alpha: float = 0.7, no_repeat_ngram_size: int = 3) -> List[Dict]:
        v = self.vocab
        max_len = max_len or v.max_len
        beams = [([v.start_id], 0.0)]           # (token sequence, cumulative log-prob)
        finished = []

        for _ in range(max_len - 1):
            tokens = tf.constant([seq for seq, _ in beams], dtype=tf.int32)
            enc_rep = tf.repeat(enc, repeats=len(beams), axis=0)
            logits = self._decode_step(enc_rep, tokens)[:, -1, :]
            log_probs = tf.nn.log_softmax(logits, axis=-1).numpy()
            log_probs[:, self.banned_ids] = -np.inf

            candidates = []
            for i, (seq, score) in enumerate(beams):
                lp = log_probs[i]
                for tok in self._banned_ngram_tokens(seq, no_repeat_ngram_size):
                    lp[tok] = -np.inf
                k = min(2 * beam_width, lp.shape[0])
                for tok in np.argpartition(-lp, k - 1)[:k]:
                    if np.isfinite(lp[tok]):
                        candidates.append((seq + [int(tok)], score + float(lp[tok])))

            candidates.sort(key=lambda c: c[1], reverse=True)
            beams = []
            for seq, score in candidates:
                if seq[-1] == v.end_id:
                    finished.append((seq, self._length_normalize(score, len(seq) - 1, length_alpha)))
                else:
                    beams.append((seq, score))
                if len(beams) == beam_width:
                    break
            if not beams or len(finished) >= beam_width:
                break

        if not finished:                         # hit max_len without <end>
            finished = [(seq, self._length_normalize(s, len(seq) - 1, length_alpha)) for seq, s in beams]

        finished.sort(key=lambda c: c[1], reverse=True)
        return [{"caption": v.decode(seq), "score": round(float(s), 4)} for seq, s in finished]

    # ---------------- Public API ----------------
    def caption_from_features(self, features: np.ndarray, beam_width: int = 3, **kwargs) -> List[Dict]:
        feats = np.asarray(features, dtype=np.float32)
        if feats.ndim == 2:
            feats = feats[None]
        enc = self._encode_step(tf.convert_to_tensor(feats))
        return self.beam_search(enc, beam_width=beam_width, **kwargs)

    def caption(self, image: ImageInput, beam_width: int = 3, **kwargs) -> List[Dict]:
        enc = self._encode_step(self.extract_features(image))
        return self.beam_search(enc, beam_width=beam_width, **kwargs)
