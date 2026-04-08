"""
DIEN（Deep Interest Evolution Network）TensorFlow/Keras 兼容实现。

与原始版本保持的核心语义：
1. 使用 GRU + 注意力 + AUGRU 建模用户兴趣演化。
2. 保留辅助损失（正负样本点击序列）以增强训练信号。
3. 保留负采样列命名（negtive_*）以兼容原代码字段。
"""

import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# 基础配置
# =========================
RECENT_MOVIES = 5
EMBEDDING_SIZE = 10
MOVIE_ID_VOCAB_SIZE = 1001
USER_ID_VOCAB_SIZE = 30001
BATCH_SIZE = 12
EPOCHS = 5
RANDOM_SEED = 2026

# 训练/测试使用不同随机种子，保持与原逻辑一致
TRAIN_NEGATIVE_SEED = 2020
TEST_NEGATIVE_SEED = 2021

# 类型定义：所有模型输入字段
MODEL_INPUT_DTYPES = {
    "movieAvgRating": tf.float32,
    "movieRatingStddev": tf.float32,
    "movieRatingCount": tf.int32,
    "userAvgRating": tf.float32,
    "userRatingStddev": tf.float32,
    "userRatingCount": tf.int32,
    "releaseYear": tf.int32,
    "movieId": tf.int32,
    "userId": tf.int32,
    "userRatedMovie1": tf.int32,
    "userRatedMovie2": tf.int32,
    "userRatedMovie3": tf.int32,
    "userRatedMovie4": tf.int32,
    "userRatedMovie5": tf.int32,
    "userGenre1": tf.string,
    "userGenre2": tf.string,
    "userGenre3": tf.string,
    "userGenre4": tf.string,
    "userGenre5": tf.string,
    "movieGenre1": tf.string,
    "movieGenre2": tf.string,
    "movieGenre3": tf.string,
    # 负样本字段保留原始拼写（negtive_*）避免破坏兼容性
    "negtive_userRatedMovie2": tf.int32,
    "negtive_userRatedMovie3": tf.int32,
    "negtive_userRatedMovie4": tf.int32,
    "negtive_userRatedMovie5": tf.int32,
    "label": tf.int32,
}

INT_COLUMNS = [
    "movieRatingCount",
    "userRatingCount",
    "releaseYear",
    "movieId",
    "userId",
    "userRatedMovie1",
    "userRatedMovie2",
    "userRatedMovie3",
    "userRatedMovie4",
    "userRatedMovie5",
    "label",
]

FLOAT_COLUMNS = [
    "movieAvgRating",
    "movieRatingStddev",
    "userAvgRating",
    "userRatingStddev",
]

STRING_COLUMNS = [
    "userGenre1",
    "userGenre2",
    "userGenre3",
    "userGenre4",
    "userGenre5",
    "movieGenre1",
    "movieGenre2",
    "movieGenre3",
]

NEGATIVE_SOURCE_COLUMNS = [
    "userRatedMovie2",
    "userRatedMovie3",
    "userRatedMovie4",
    "userRatedMovie5",
]

NEGATIVE_OUTPUT_COLUMNS = [
    "negtive_userRatedMovie2",
    "negtive_userRatedMovie3",
    "negtive_userRatedMovie4",
    "negtive_userRatedMovie5",
]

GENRE_VOCAB = [
    "Film-Noir",
    "Action",
    "Adventure",
    "Horror",
    "Romance",
    "War",
    "Comedy",
    "Western",
    "Documentary",
    "Sci-Fi",
    "Drama",
    "Thriller",
    "Crime",
    "Fantasy",
    "Animation",
    "IMAX",
    "Mystery",
    "Children",
    "Musical",
]


def _resolve_sample_path(file_name: str) -> Path:
    """自动定位样本文件，避免硬编码绝对路径。"""
    relative_candidates = [
        Path("src/main/resources/webroot/sampledata") / file_name,
        Path("target/classes/webroot/sampledata") / file_name,
    ]

    script_path = Path(__file__).resolve()
    search_roots = [Path.cwd(), script_path.parent, *script_path.parents]

    for root in search_roots:
        for relative_path in relative_candidates:
            candidate = (root / relative_path).resolve()
            if candidate.exists():
                return candidate

    raise FileNotFoundError(f"未找到样本文件: {file_name}")


def _sample_negative_movie_id(positive_movie_id: int, rng: random.Random) -> int:
    """在 [0, MOVIE_ID_VOCAB_SIZE) 范围内采样一个不等于正样本的 movieId。"""
    positive_movie_id = int(max(0, min(positive_movie_id, MOVIE_ID_VOCAB_SIZE - 1)))
    negative_movie_id = rng.randint(0, MOVIE_ID_VOCAB_SIZE - 1)
    while negative_movie_id == positive_movie_id:
        negative_movie_id = rng.randint(0, MOVIE_ID_VOCAB_SIZE - 1)
    return negative_movie_id


def _prepare_dataframe(csv_path: Path, seed_num: int) -> pd.DataFrame:
    """读取并规范化样本数据，同时构建负采样列。"""
    frame = pd.read_csv(csv_path).fillna(0)

    for column in INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(np.int32)
    for column in FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(np.float32)
    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype(str)

    rng = random.Random(seed_num)
    negative_data = {}
    for source_column, output_column in zip(NEGATIVE_SOURCE_COLUMNS, NEGATIVE_OUTPUT_COLUMNS):
        source_values = frame[source_column].to_numpy()
        negative_data[output_column] = [
            _sample_negative_movie_id(int(value), rng) for value in source_values
        ]

    negative_frame = pd.DataFrame(negative_data, index=frame.index, dtype=np.int32)
    frame = pd.concat([frame, negative_frame], axis=1)
    frame = frame[list(MODEL_INPUT_DTYPES.keys())]
    return frame


def build_dataset_with_negative_movie(
    csv_path: Path, batch_size: int, seed_num: int, shuffle: bool
) -> tf.data.Dataset:
    """构建 DIEN 所需的 tf.data.Dataset。"""
    frame = _prepare_dataframe(csv_path, seed_num=seed_num)
    dataset = tf.data.Dataset.from_tensor_slices(
        {name: frame[name].to_numpy() for name in MODEL_INPUT_DTYPES}
    )
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(frame), seed=seed_num, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _create_model_inputs() -> Dict[str, tf.keras.layers.Input]:
    """创建 Keras Functional 输入定义。"""
    return {
        name: tf.keras.layers.Input(name=name, shape=(), dtype=dtype)
        for name, dtype in MODEL_INPUT_DTYPES.items()
    }


def _expand_and_cast_column(input_tensor: tf.Tensor, dtype: tf.dtypes.DType, name: str) -> tf.Tensor:
    """将标量特征扩展为 [batch, 1] 并转换类型。"""
    return tf.keras.layers.Lambda(
        lambda x: tf.cast(tf.expand_dims(x, axis=-1), dtype), name=name
    )(input_tensor)


def _stack_int_features(inputs: Dict[str, tf.Tensor], feature_names: List[str], name: str) -> tf.Tensor:
    """将多个标量 int 特征堆叠为 [batch, time]。"""
    return tf.keras.layers.Lambda(
        lambda tensors: tf.stack([tf.cast(t, tf.int32) for t in tensors], axis=1),
        name=name,
    )([inputs[feature_name] for feature_name in feature_names])


class AttentionLayer(tf.keras.layers.Layer):
    """候选物品与行为序列的注意力计算层。"""

    def __init__(self, embedding_size: int = EMBEDDING_SIZE, time_length: int = RECENT_MOVIES):
        super().__init__()
        self.time_length = time_length
        self.embedding_size = embedding_size
        self.repeat_by_time = tf.keras.layers.RepeatVector(self.time_length)
        self.repeat_by_embedding = tf.keras.layers.RepeatVector(self.embedding_size)
        self.multiply = tf.keras.layers.Multiply()
        self.dense_32 = tf.keras.layers.Dense(32, activation="sigmoid")
        self.dense_1 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.permute = tf.keras.layers.Permute((2, 1))

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        candidate_inputs, gru_hidden_state = inputs
        repeated_candidate_layer = self.repeat_by_time(candidate_inputs)
        activation_product_layer = self.multiply([gru_hidden_state, repeated_candidate_layer])
        activation_unit = self.dense_32(activation_product_layer)
        activation_unit = self.dense_1(activation_unit)
        attention_score = tf.squeeze(activation_unit, axis=2)
        attention_score = self.repeat_by_embedding(attention_score)
        attention_score = self.permute(attention_score)
        return attention_score


class GRUGateParameter(tf.keras.layers.Layer):
    """AUGRU 门控参数计算层。"""

    def __init__(self, embedding_size: int = EMBEDDING_SIZE):
        super().__init__()
        self.embedding_size = embedding_size
        self.multiply = tf.keras.layers.Multiply()
        self.dense_sigmoid = tf.keras.layers.Dense(self.embedding_size, activation="sigmoid")
        self.dense_tanh = tf.keras.layers.Dense(self.embedding_size, activation="tanh")
        self.input_w = tf.keras.layers.Dense(self.embedding_size, activation=None, use_bias=True)
        self.hidden_w = tf.keras.layers.Dense(self.embedding_size, activation=None, use_bias=False)

    def call(self, inputs: List[tf.Tensor], z_t_inputs: tf.Tensor = None) -> tf.Tensor:
        gru_inputs, hidden_inputs = inputs
        if z_t_inputs is None:
            return self.dense_sigmoid(self.input_w(gru_inputs) + self.hidden_w(hidden_inputs))
        return self.dense_tanh(
            self.input_w(gru_inputs) + self.hidden_w(self.multiply([hidden_inputs, z_t_inputs]))
        )


class AUGRULayer(tf.keras.layers.Layer):
    """Attentional Update Gate GRU。"""

    def __init__(self, embedding_size: int = EMBEDDING_SIZE, time_length: int = RECENT_MOVIES):
        super().__init__()
        self.time_length = time_length
        self.embedding_size = embedding_size
        self.multiply = tf.keras.layers.Multiply()
        self.add = tf.keras.layers.Add()
        self.r_t = GRUGateParameter(embedding_size=self.embedding_size)
        self.z_t = GRUGateParameter(embedding_size=self.embedding_size)
        self.h_t_next = GRUGateParameter(embedding_size=self.embedding_size)

    def call(self, inputs: List[tf.Tensor]) -> tf.Tensor:
        gru_hidden_state_inputs, attention_s = inputs
        batch_size = tf.shape(gru_hidden_state_inputs)[0]
        limit = (6.0 / (2.0 * self.embedding_size)) ** 0.5
        augru_hidden_state = tf.random.uniform(
            shape=(batch_size, self.embedding_size),
            minval=-limit,
            maxval=limit,
            dtype=gru_hidden_state_inputs.dtype,
        )

        for time_index in range(self.time_length):
            r_t = self.r_t([gru_hidden_state_inputs[:, time_index, :], augru_hidden_state])
            z_t = self.z_t([gru_hidden_state_inputs[:, time_index, :], augru_hidden_state])
            h_t_next = self.h_t_next(
                [gru_hidden_state_inputs[:, time_index, :], augru_hidden_state], z_t
            )
            rt_attention = self.multiply([attention_s[:, time_index, :], r_t])
            augru_hidden_state = self.add(
                [
                    self.multiply([(1.0 - rt_attention), augru_hidden_state]),
                    self.multiply([rt_attention, h_t_next]),
                ]
            )
        return augru_hidden_state


class AuxiliaryLossLayer(tf.keras.layers.Layer):
    """DIEN 辅助损失层，直接通过 add_loss 注入总损失。"""

    def __init__(self, time_length: int = RECENT_MOVIES):
        super().__init__()
        self.time_length = time_length
        self.positive_dense_32 = tf.keras.layers.Dense(32, activation="sigmoid")
        self.positive_dense_1 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.negative_dense_32 = tf.keras.layers.Dense(32, activation="sigmoid")
        self.negative_dense_1 = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs: List[tf.Tensor], alpha: float = 0.5) -> tf.Tensor:
        negative_movie_t1, positive_movie_t0, movie_hidden_state, y_true, y_pred = inputs

        positive_concat_layer = tf.concat(
            [movie_hidden_state[:, 0:4, :], positive_movie_t0[:, 1:5, :]], axis=-1
        )
        positive_concat_layer = self.positive_dense_32(positive_concat_layer)
        positive_loss = self.positive_dense_1(positive_concat_layer)

        negative_concat_layer = tf.concat(
            [movie_hidden_state[:, 0:4, :], negative_movie_t1[:, :, :]], axis=-1
        )
        negative_concat_layer = self.negative_dense_32(negative_concat_layer)
        negative_loss = self.negative_dense_1(negative_concat_layer)

        auxiliary_loss_values = positive_loss + negative_loss
        main_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
        auxiliary_term = tf.reduce_mean(tf.reduce_sum(auxiliary_loss_values, axis=1))
        final_loss = main_loss - alpha * auxiliary_term
        self.add_loss(final_loss)
        return final_loss


def build_dien_model() -> tf.keras.Model:
    """构建 DIEN 模型（新版 tf.keras 兼容）。"""
    inputs = _create_model_inputs()

    # movieId 序列与负样本序列（显式拼接，替代 DenseFeatures）
    candidate_movie_ids = _stack_int_features(inputs, ["movieId"], "candidate_movie_ids")
    user_behavior_movie_ids = _stack_int_features(
        inputs,
        ["userRatedMovie1", "userRatedMovie2", "userRatedMovie3", "userRatedMovie4", "userRatedMovie5"],
        "user_behavior_movie_ids",
    )
    negative_movie_ids = _stack_int_features(
        inputs,
        [
            "negtive_userRatedMovie2",
            "negtive_userRatedMovie3",
            "negtive_userRatedMovie4",
            "negtive_userRatedMovie5",
        ],
        "negative_movie_ids",
    )

    # ID / 类别 embedding
    movie_embedding_layer = tf.keras.layers.Embedding(
        input_dim=MOVIE_ID_VOCAB_SIZE,
        output_dim=EMBEDDING_SIZE,
        mask_zero=True,
        name="movie_embedding",
    )
    user_id_embedding_layer = tf.keras.layers.Embedding(
        input_dim=USER_ID_VOCAB_SIZE,
        output_dim=EMBEDDING_SIZE,
        name="user_id_embedding",
    )
    genre_lookup_layer = tf.keras.layers.StringLookup(
        vocabulary=GENRE_VOCAB,
        mask_token=None,
        num_oov_indices=1,
        name="genre_lookup",
    )
    genre_embedding_layer = tf.keras.layers.Embedding(
        input_dim=len(GENRE_VOCAB) + 1,
        output_dim=EMBEDDING_SIZE,
        name="genre_embedding",
    )

    user_behaviors_emb_layer = movie_embedding_layer(user_behavior_movie_ids)
    candidate_emb_layer = movie_embedding_layer(candidate_movie_ids)
    candidate_emb_layer = tf.keras.layers.Lambda(
        lambda x: tf.squeeze(x, axis=1), name="candidate_embedding_squeeze"
    )(candidate_emb_layer)
    negative_movie_emb_layer = movie_embedding_layer(negative_movie_ids)

    user_behaviors_hidden_state = tf.keras.layers.GRU(
        EMBEDDING_SIZE, return_sequences=True, name="behavior_gru"
    )(user_behaviors_emb_layer)

    attention_score = AttentionLayer()([candidate_emb_layer, user_behaviors_hidden_state])
    augru_emb = AUGRULayer()([user_behaviors_hidden_state, attention_score])

    # 用户特征（与原始语义一致：userId emb + userGenre1 emb + 3 个数值）
    user_id_emb = user_id_embedding_layer(inputs["userId"])
    user_genre_emb = genre_embedding_layer(genre_lookup_layer(inputs["userGenre1"]))
    user_rating_count = _expand_and_cast_column(
        inputs["userRatingCount"], tf.float32, "user_rating_count_expand"
    )
    user_avg_rating = _expand_and_cast_column(inputs["userAvgRating"], tf.float32, "user_avg_rating_expand")
    user_rating_stddev = _expand_and_cast_column(
        inputs["userRatingStddev"], tf.float32, "user_rating_stddev_expand"
    )
    user_profile_layer = tf.keras.layers.Concatenate(name="user_profile_concat")(
        [user_id_emb, user_genre_emb, user_rating_count, user_avg_rating, user_rating_stddev]
    )

    # 上下文特征（与原始语义一致：movieGenre1 emb + 4 个数值）
    item_genre_emb = genre_embedding_layer(genre_lookup_layer(inputs["movieGenre1"]))
    release_year = _expand_and_cast_column(inputs["releaseYear"], tf.float32, "release_year_expand")
    movie_rating_count = _expand_and_cast_column(
        inputs["movieRatingCount"], tf.float32, "movie_rating_count_expand"
    )
    movie_avg_rating = _expand_and_cast_column(
        inputs["movieAvgRating"], tf.float32, "movie_avg_rating_expand"
    )
    movie_rating_stddev = _expand_and_cast_column(
        inputs["movieRatingStddev"], tf.float32, "movie_rating_stddev_expand"
    )
    context_features_layer = tf.keras.layers.Concatenate(name="context_features_concat")(
        [item_genre_emb, release_year, movie_rating_count, movie_avg_rating, movie_rating_stddev]
    )

    # 主干 DNN
    concat_layer = tf.keras.layers.Concatenate(name="dien_concat")(
        [augru_emb, candidate_emb_layer, user_profile_layer, context_features_layer]
    )
    output_layer = tf.keras.layers.Dense(128)(concat_layer)
    output_layer = tf.keras.layers.PReLU()(output_layer)
    output_layer = tf.keras.layers.Dense(64)(output_layer)
    output_layer = tf.keras.layers.PReLU()(output_layer)
    y_pred = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(output_layer)

    # 辅助损失：保留原公式，通过 add_loss 注入训练
    y_true = _expand_and_cast_column(inputs["label"], tf.float32, "label_expand")
    _ = AuxiliaryLossLayer()(
        [negative_movie_emb_layer, user_behaviors_emb_layer, user_behaviors_hidden_state, y_true, y_pred]
    )

    model = tf.keras.Model(inputs=inputs, outputs=y_pred, name="dien_model")
    model.compile(optimizer=tf.keras.optimizers.Adam())
    return model


def _collect_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """从 dataset 中按顺序收集标签，用于离线 AUC 计算与结果展示。"""
    labels = []
    for batch_features in dataset:
        labels.append(tf.cast(batch_features["label"], tf.float32))
    return tf.concat(labels, axis=0).numpy().reshape(-1)


def run() -> None:
    """训练 + 评估 + 打印示例预测。"""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    training_samples_file_path = _resolve_sample_path("trainingSamples.csv")
    test_samples_file_path = _resolve_sample_path("testSamples.csv")

    train_dataset = build_dataset_with_negative_movie(
        training_samples_file_path, batch_size=BATCH_SIZE, seed_num=TRAIN_NEGATIVE_SEED, shuffle=True
    )
    test_dataset = build_dataset_with_negative_movie(
        test_samples_file_path, batch_size=BATCH_SIZE, seed_num=TEST_NEGATIVE_SEED, shuffle=False
    )

    model = build_dien_model()
    model.fit(train_dataset, epochs=EPOCHS, verbose=2)

    test_loss = model.evaluate(test_dataset, verbose=2)
    test_loss_value = float(test_loss[0] if isinstance(test_loss, (list, tuple, np.ndarray)) else test_loss)

    predictions = model.predict(test_dataset, verbose=0).reshape(-1)
    labels = _collect_labels(test_dataset)

    roc_auc_metric = tf.keras.metrics.AUC(curve="ROC")
    roc_auc_metric.update_state(labels, predictions)
    test_roc_auc = float(roc_auc_metric.result().numpy())

    print(f"\n\nTest Loss {test_loss_value:.6f}, Test ROC AUC {test_roc_auc:.6f}")

    first_batch_labels = labels[:12]
    first_batch_predictions = predictions[:12]
    for prediction, label in zip(first_batch_predictions, first_batch_labels):
        print(
            f"Predicted good rating: {prediction:.2%} | Actual rating label: "
            f"{'Good Rating' if bool(label) else 'Bad Rating'}"
        )


if __name__ == "__main__":
    run()
