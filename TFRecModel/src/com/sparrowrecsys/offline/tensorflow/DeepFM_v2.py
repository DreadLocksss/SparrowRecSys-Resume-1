"""
DeepFM_v2（TensorFlow/Keras 新版兼容实现）

与原始 DeepFM_v2 保持一致的核心语义：
1. 一阶部分：类别特征与数值特征分开建模后再相加。
2. 二阶部分：使用“全交叉”FM 形式处理类别 embedding 与数值映射向量。
3. 深度部分：在二阶拼接特征上接 MLP。
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf

# =========================
# 基础配置
# =========================
BATCH_SIZE = 12
EPOCHS = 5
RANDOM_SEED = 2026

MOVIE_ID_VOCAB_SIZE = 1001
USER_ID_VOCAB_SIZE = 30001
EMBEDDING_SIZE = 10
FM_LATENT_DIM = 64

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
    "userGenre1": tf.string,
    "movieGenre1": tf.string,
}

FLOAT_COLUMNS = [
    "movieAvgRating",
    "movieRatingStddev",
    "userAvgRating",
    "userRatingStddev",
]

INT_COLUMNS = [
    "movieRatingCount",
    "userRatingCount",
    "releaseYear",
    "movieId",
    "userId",
    "label",
]

STRING_COLUMNS = ["userGenre1", "movieGenre1"]

DEEP_NUMERIC_COLUMNS = [
    "releaseYear",
    "movieRatingCount",
    "movieAvgRating",
    "movieRatingStddev",
    "userRatingCount",
    "userAvgRating",
    "userRatingStddev",
]


def _resolve_sample_path(file_name: str) -> Path:
    """自动定位项目内 sampledata 文件，避免硬编码本地绝对路径。"""
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


def _prepare_dataframe(csv_path: Path) -> pd.DataFrame:
    """读取并完成类型规范化，只保留本模型所需字段。"""
    frame = pd.read_csv(csv_path).fillna(0)

    for column in FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(np.float32)
    for column in INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0).astype(np.int32)
    for column in STRING_COLUMNS:
        frame[column] = frame[column].astype(str)

    selected_columns = list(MODEL_INPUT_DTYPES.keys()) + ["label"]
    return frame[selected_columns]


def build_dataset(csv_path: Path, batch_size: int, shuffle: bool, seed: int) -> tf.data.Dataset:
    """构建 tf.data.Dataset，输出格式为 (features, label)。"""
    frame = _prepare_dataframe(csv_path)
    features = {name: frame[name].to_numpy() for name in MODEL_INPUT_DTYPES}
    labels = frame["label"].to_numpy(dtype=np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size=len(frame), seed=seed, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _create_inputs() -> Dict[str, tf.keras.layers.Input]:
    """创建 Keras Functional 输入。"""
    return {
        name: tf.keras.layers.Input(name=name, shape=(), dtype=dtype)
        for name, dtype in MODEL_INPUT_DTYPES.items()
    }


def _expand_and_cast(input_tensor: tf.Tensor, dtype: tf.dtypes.DType, name: str) -> tf.Tensor:
    """将标量特征扩展成 [batch, 1] 并转成指定类型。"""
    return tf.keras.layers.Lambda(
        lambda x: tf.cast(tf.expand_dims(x, axis=-1), dtype=dtype),
        name=name,
    )(input_tensor)


class ReduceLayer(tf.keras.layers.Layer):
    """对指定轴做 reduce 操作，兼容原始实现语义。"""

    def __init__(self, axis: int, op: str = "sum", **kwargs):
        super().__init__(**kwargs)
        self.axis = axis
        self.op = op
        if self.op not in ("sum", "mean"):
            raise ValueError("op 仅支持 'sum' 或 'mean'")

    def call(self, inputs: tf.Tensor, **kwargs) -> tf.Tensor:
        if self.op == "sum":
            return tf.reduce_sum(inputs, axis=self.axis)
        return tf.reduce_mean(inputs, axis=self.axis)


def build_deepfm_v2_model() -> tf.keras.Model:
    """构建 DeepFM_v2 模型。"""
    inputs = _create_inputs()

    # 数值特征（Deep 与 FM 一致使用同一组输入）
    deep_numeric_tensors = [
        _expand_and_cast(inputs[column_name], tf.float32, f"{column_name}_expand")
        for column_name in DEEP_NUMERIC_COLUMNS
    ]
    deep_numeric_feature = tf.keras.layers.Concatenate(name="deep_numeric_concat")(deep_numeric_tensors)

    # 类别特征编码（对应原代码中的 indicator column）
    genre_lookup = tf.keras.layers.StringLookup(
        vocabulary=GENRE_VOCAB,
        mask_token=None,
        num_oov_indices=1,
        name="genre_lookup",
    )
    genre_token_size = len(GENRE_VOCAB) + 1
    one_hot_encoder_genre = tf.keras.layers.CategoryEncoding(
        num_tokens=genre_token_size,
        output_mode="one_hot",
        name="genre_one_hot",
    )
    one_hot_encoder_movie = tf.keras.layers.CategoryEncoding(
        num_tokens=MOVIE_ID_VOCAB_SIZE,
        output_mode="one_hot",
        name="movie_one_hot",
    )
    one_hot_encoder_user = tf.keras.layers.CategoryEncoding(
        num_tokens=USER_ID_VOCAB_SIZE,
        output_mode="one_hot",
        name="user_one_hot",
    )

    movie_one_hot = one_hot_encoder_movie(inputs["movieId"])
    user_one_hot = one_hot_encoder_user(inputs["userId"])
    user_genre_ids = genre_lookup(inputs["userGenre1"])
    movie_genre_ids = genre_lookup(inputs["movieGenre1"])
    user_genre_one_hot = one_hot_encoder_genre(user_genre_ids)
    movie_genre_one_hot = one_hot_encoder_genre(movie_genre_ids)

    first_order_cat_feature = tf.keras.layers.Concatenate(name="first_order_cat_concat")(
        [movie_one_hot, user_one_hot, user_genre_one_hot, movie_genre_one_hot]
    )
    first_order_cat_feature = tf.keras.layers.Dense(1, activation=None, name="first_order_cat_dense")(
        first_order_cat_feature
    )
    first_order_deep_feature = tf.keras.layers.Dense(1, activation=None, name="first_order_deep_dense")(
        deep_numeric_feature
    )
    first_order_feature = tf.keras.layers.Add(name="first_order_add")(
        [first_order_cat_feature, first_order_deep_feature]
    )

    # 二阶类别 embedding（与原始语义一致：movie/user/genre 先 embedding，再映射到 64 维）
    movie_embedding = tf.keras.layers.Embedding(
        input_dim=MOVIE_ID_VOCAB_SIZE,
        output_dim=EMBEDDING_SIZE,
        name="movie_embedding",
    )
    user_embedding = tf.keras.layers.Embedding(
        input_dim=USER_ID_VOCAB_SIZE,
        output_dim=EMBEDDING_SIZE,
        name="user_embedding",
    )
    genre_embedding = tf.keras.layers.Embedding(
        input_dim=genre_token_size,
        output_dim=EMBEDDING_SIZE,
        name="genre_embedding",
    )

    movie_emb = movie_embedding(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs["movieId"]))
    movie_emb = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="movie_emb_squeeze")(movie_emb)

    user_emb = user_embedding(tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(inputs["userId"]))
    user_emb = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1), name="user_emb_squeeze")(user_emb)

    user_genre_emb = genre_embedding(user_genre_ids)
    movie_genre_emb = genre_embedding(movie_genre_ids)

    second_order_cat_embs: List[tf.Tensor] = [movie_genre_emb, movie_emb, user_genre_emb, user_emb]
    second_order_cat_columns: List[tf.Tensor] = []
    for index, feature_emb in enumerate(second_order_cat_embs):
        feature = tf.keras.layers.Dense(FM_LATENT_DIM, activation=None, name=f"second_cat_dense_{index}")(
            feature_emb
        )
        feature = tf.keras.layers.Reshape((1, FM_LATENT_DIM), name=f"second_cat_reshape_{index}")(feature)
        second_order_cat_columns.append(feature)

    second_order_deep_columns = tf.keras.layers.Dense(
        FM_LATENT_DIM, activation=None, name="second_deep_dense"
    )(deep_numeric_feature)
    second_order_deep_columns = tf.keras.layers.Reshape(
        (1, FM_LATENT_DIM), name="second_deep_reshape"
    )(second_order_deep_columns)

    second_order_all_features = tf.keras.layers.Concatenate(axis=1, name="second_order_feature_concat")(
        second_order_cat_columns + [second_order_deep_columns]
    )

    # 深度分支（原始结构：Flatten -> 32 -> 16）
    deep_feature = tf.keras.layers.Flatten(name="deep_flatten")(second_order_all_features)
    deep_feature = tf.keras.layers.Dense(32, activation="relu", name="deep_dense_32")(deep_feature)
    deep_feature = tf.keras.layers.Dense(16, activation="relu", name="deep_dense_16")(deep_feature)

    # FM 二阶交叉项（保持与原代码一致：sum^2 - square_sum）
    second_order_sum_feature = ReduceLayer(axis=1, op="sum", name="second_sum")(second_order_all_features)
    second_order_sum_square_feature = tf.keras.layers.Multiply(name="second_sum_square")(
        [second_order_sum_feature, second_order_sum_feature]
    )
    second_order_square_feature = tf.keras.layers.Multiply(name="second_square")(
        [second_order_all_features, second_order_all_features]
    )
    second_order_square_sum_feature = ReduceLayer(axis=1, op="sum", name="second_square_sum")(
        second_order_square_feature
    )
    second_order_fm_feature = tf.keras.layers.Subtract(name="second_fm_subtract")(
        [second_order_sum_square_feature, second_order_square_sum_feature]
    )

    concatenated_outputs = tf.keras.layers.Concatenate(axis=1, name="final_concat")(
        [first_order_feature, second_order_fm_feature, deep_feature]
    )
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid", name="prediction")(concatenated_outputs)

    model = tf.keras.Model(inputs=inputs, outputs=output_layer, name="deepfm_v2_model")
    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(curve="ROC", name="roc_auc"),
            tf.keras.metrics.AUC(curve="PR", name="pr_auc"),
        ],
    )
    return model


def _collect_labels(dataset: tf.data.Dataset) -> np.ndarray:
    """从 dataset 按顺序提取 label，便于打印预测示例。"""
    labels = []
    for _, batch_labels in dataset:
        labels.append(tf.cast(batch_labels, tf.float32))
    return tf.concat(labels, axis=0).numpy().reshape(-1)


def run() -> None:
    """训练、评估并打印预测样例。"""
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    training_samples_file_path = _resolve_sample_path("trainingSamples.csv")
    test_samples_file_path = _resolve_sample_path("testSamples.csv")

    train_dataset = build_dataset(
        training_samples_file_path,
        batch_size=BATCH_SIZE,
        shuffle=True,
        seed=RANDOM_SEED,
    )
    test_dataset = build_dataset(
        test_samples_file_path,
        batch_size=BATCH_SIZE,
        shuffle=False,
        seed=RANDOM_SEED,
    )

    model = build_deepfm_v2_model()
    model.fit(train_dataset, epochs=EPOCHS, verbose=2)

    test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset, verbose=2)
    print(
        "\n\nTest Loss {:.6f}, Test Accuracy {:.6f}, Test ROC AUC {:.6f}, Test PR AUC {:.6f}".format(
            float(test_loss), float(test_accuracy), float(test_roc_auc), float(test_pr_auc)
        )
    )

    predictions = model.predict(test_dataset, verbose=0).reshape(-1)
    labels = _collect_labels(test_dataset)

    for prediction, good_rating in zip(predictions[:12], labels[:12]):
        print(
            "Predicted good rating: {:.2%} | Actual rating label: {}".format(
                prediction, "Good Rating" if bool(good_rating) else "Bad Rating"
            )
        )


if __name__ == "__main__":
    run()
