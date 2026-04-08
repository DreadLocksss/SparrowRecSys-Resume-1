from pathlib import Path

import tensorflow as tf

# -----------------------------
# 常量定义
# -----------------------------
BATCH_SIZE = 12
EPOCHS = 20
RECENT_MOVIES = 5  # userRatedMovie1~userRatedMovie5
EMBEDDING_SIZE = 10
MOVIE_ID_BUCKET_SIZE = 1001
USER_ID_BUCKET_SIZE = 30001

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


def locate_project_root() -> Path:
    """从当前文件向上查找项目根目录（包含 sampledata）。"""
    current_file = Path(__file__).resolve()
    for parent in current_file.parents:
        sample_data_dir = parent / "src" / "main" / "resources" / "webroot" / "sampledata"
        if sample_data_dir.exists():
            return parent
    raise FileNotFoundError("未找到 sampledata 目录，请检查项目目录结构。")


PROJECT_ROOT = locate_project_root()
SAMPLE_DATA_DIR = PROJECT_ROOT / "src" / "main" / "resources" / "webroot" / "sampledata"

# 训练集与测试集路径
training_samples_origin = (SAMPLE_DATA_DIR / "trainingSamples.csv").as_uri()
test_samples_origin = (SAMPLE_DATA_DIR / "testSamples.csv").as_uri()

training_samples_file_path = tf.keras.utils.get_file(
    "trainingSamples.csv",
    training_samples_origin,
)
test_samples_file_path = tf.keras.utils.get_file(
    "testSamples.csv",
    test_samples_origin,
)


# -----------------------------
# 数据集加载
# -----------------------------
def get_dataset(file_path, shuffle=True, num_epochs=1):
    """将 CSV 文件读取为 TensorFlow Dataset。"""
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=BATCH_SIZE,
        label_name="label",
        na_value="0",
        num_epochs=num_epochs,
        shuffle=shuffle,
        ignore_errors=True,
    )
    return dataset


train_dataset = get_dataset(training_samples_file_path, shuffle=True, num_epochs=1)
test_dataset = get_dataset(test_samples_file_path, shuffle=False, num_epochs=1)


# -----------------------------
# 模型输入定义
# -----------------------------
inputs = {
    "movieAvgRating": tf.keras.Input(name="movieAvgRating", shape=(), dtype="float32"),
    "movieRatingStddev": tf.keras.Input(name="movieRatingStddev", shape=(), dtype="float32"),
    "movieRatingCount": tf.keras.Input(name="movieRatingCount", shape=(), dtype="int32"),
    "userAvgRating": tf.keras.Input(name="userAvgRating", shape=(), dtype="float32"),
    "userRatingStddev": tf.keras.Input(name="userRatingStddev", shape=(), dtype="float32"),
    "userRatingCount": tf.keras.Input(name="userRatingCount", shape=(), dtype="int32"),
    "releaseYear": tf.keras.Input(name="releaseYear", shape=(), dtype="int32"),
    "movieId": tf.keras.Input(name="movieId", shape=(), dtype="int32"),
    "userId": tf.keras.Input(name="userId", shape=(), dtype="int32"),
    "userRatedMovie1": tf.keras.Input(name="userRatedMovie1", shape=(), dtype="int32"),
    "userRatedMovie2": tf.keras.Input(name="userRatedMovie2", shape=(), dtype="int32"),
    "userRatedMovie3": tf.keras.Input(name="userRatedMovie3", shape=(), dtype="int32"),
    "userRatedMovie4": tf.keras.Input(name="userRatedMovie4", shape=(), dtype="int32"),
    "userRatedMovie5": tf.keras.Input(name="userRatedMovie5", shape=(), dtype="int32"),
    "userGenre1": tf.keras.Input(name="userGenre1", shape=(), dtype="string"),
    "movieGenre1": tf.keras.Input(name="movieGenre1", shape=(), dtype="string"),
}

MODEL_INPUT_KEYS = list(inputs.keys())


def keep_model_inputs(features, label):
    """筛选模型输入特征，并将标签转为 float32。"""
    selected = {key: features[key] for key in MODEL_INPUT_KEYS}
    return selected, tf.cast(label, tf.float32)


train_dataset = train_dataset.map(keep_model_inputs)
test_dataset = test_dataset.map(keep_model_inputs)


def scalar_to_float_column(tensor):
    """将标量特征转为 shape=(1,) 的 float32 张量，便于拼接。"""
    casted = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(tensor)
    return tf.keras.layers.Reshape((1,))(casted)


# -----------------------------
# 特征编码层
# -----------------------------
# 电影 ID 的 Embedding，同时用于 candidate movie 和近期行为序列
movie_embedding_layer = tf.keras.layers.Embedding(
    input_dim=MOVIE_ID_BUCKET_SIZE,
    output_dim=EMBEDDING_SIZE,
    # mask_zero=True,
)

# 用户 ID 的 Embedding（用于用户画像）
user_embedding_layer = tf.keras.layers.Embedding(
    input_dim=USER_ID_BUCKET_SIZE,
    output_dim=EMBEDDING_SIZE,
)

# 类型特征查表与 Embedding（沿用原始语义，仅使用 userGenre1 与 movieGenre1）
genre_lookup = tf.keras.layers.StringLookup(
    vocabulary=GENRE_VOCAB,
    mask_token=None,
    num_oov_indices=1,
)
genre_embedding_layer = tf.keras.layers.Embedding(
    input_dim=genre_lookup.vocabulary_size(),
    output_dim=EMBEDDING_SIZE,
)


# -----------------------------
# DIN 输入张量构建
# -----------------------------
# candidate movie（等价于原始 candidate_movie_col + DenseFeatures）
candidate_movie_ids = tf.keras.layers.Reshape((1,))(inputs["movieId"])
candidate_emb_layer = movie_embedding_layer(candidate_movie_ids)  # (batch, 1, emb)
candidate_emb_layer = tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=1))(candidate_emb_layer)

# 用户近期行为序列（等价于原始 recent_rate_col + DenseFeatures）
recent_behavior_ids = tf.keras.layers.Concatenate(axis=1)(
    [
        tf.keras.layers.Reshape((1,))(inputs["userRatedMovie1"]),
        tf.keras.layers.Reshape((1,))(inputs["userRatedMovie2"]),
        tf.keras.layers.Reshape((1,))(inputs["userRatedMovie3"]),
        tf.keras.layers.Reshape((1,))(inputs["userRatedMovie4"]),
        tf.keras.layers.Reshape((1,))(inputs["userRatedMovie5"]),
    ]
)
user_behaviors_emb_layer = movie_embedding_layer(recent_behavior_ids)  # (batch, 5, emb)

# 用户画像（等价于原始 user_profile DenseFeatures）
user_id_emb = user_embedding_layer(inputs["userId"])  # (batch, emb)
user_genre1_ids = genre_lookup(inputs["userGenre1"])
user_genre1_emb = genre_embedding_layer(user_genre1_ids)  # (batch, emb)
user_profile_layer = tf.keras.layers.Concatenate()(
    [
        user_id_emb,
        user_genre1_emb,
        scalar_to_float_column(inputs["userRatingCount"]),
        scalar_to_float_column(inputs["userAvgRating"]),
        scalar_to_float_column(inputs["userRatingStddev"]),
    ]
)

# 上下文特征（等价于原始 context_features DenseFeatures）
movie_genre1_ids = genre_lookup(inputs["movieGenre1"])
movie_genre1_emb = genre_embedding_layer(movie_genre1_ids)  # (batch, emb)
context_features_layer = tf.keras.layers.Concatenate()(
    [
        movie_genre1_emb,
        scalar_to_float_column(inputs["releaseYear"]),
        scalar_to_float_column(inputs["movieRatingCount"]),
        scalar_to_float_column(inputs["movieAvgRating"]),
        scalar_to_float_column(inputs["movieRatingStddev"]),
    ]
)


# -----------------------------
# DIN Activation Unit
# -----------------------------

# 这一行之后维度是 (batch_size, RECENT_MOVIES, EMBEDDING_SIZE)
repeated_candidate_emb_layer = tf.keras.layers.RepeatVector(RECENT_MOVIES)(candidate_emb_layer)

activation_sub_layer = tf.keras.layers.Subtract()(
    [user_behaviors_emb_layer, repeated_candidate_emb_layer]
)
activation_product_layer = tf.keras.layers.Multiply()(
    [user_behaviors_emb_layer, repeated_candidate_emb_layer]
)
activation_all = tf.keras.layers.Concatenate(axis=-1)(
    [
        activation_sub_layer,
        user_behaviors_emb_layer,
        repeated_candidate_emb_layer,
        activation_product_layer,
    ]
)

activation_unit = tf.keras.layers.Dense(32)(activation_all)
activation_unit = tf.keras.layers.PReLU()(activation_unit)
activation_unit = tf.keras.layers.Dense(1, activation="sigmoid")(activation_unit)
activation_unit = tf.keras.layers.Flatten()(activation_unit) # (B, 5)
activation_unit = tf.keras.layers.RepeatVector(EMBEDDING_SIZE)(activation_unit)
activation_unit = tf.keras.layers.Permute((2, 1))(activation_unit)
activation_unit = tf.keras.layers.Multiply()([user_behaviors_emb_layer, activation_unit])

# 对加权行为序列做 sum pooling
user_behaviors_pooled_layers = tf.keras.layers.Lambda(lambda x: tf.reduce_sum(x, axis=1))(
    activation_unit
)


# -----------------------------
# 预测头（与原始结构一致）
# -----------------------------
concat_layer = tf.keras.layers.Concatenate()(
    [
        user_profile_layer,
        user_behaviors_pooled_layers,
        candidate_emb_layer,
        context_features_layer,
    ]
)
output_layer = tf.keras.layers.Dense(128)(concat_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(64)(output_layer)
output_layer = tf.keras.layers.PReLU()(output_layer)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(output_layer)

model = tf.keras.Model(inputs, output_layer)

# 编译模型：损失函数、优化器和评估指标
model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(curve="ROC"),
        tf.keras.metrics.AUC(curve="PR"),
    ],
)

# 训练模型（训练集共训练 5 个 epoch）
model.fit(train_dataset, epochs=EPOCHS)

# 在测试集上评估
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print(
    "\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}".format(
        test_loss,
        test_accuracy,
        test_roc_auc,
        test_pr_auc,
    )
)

# 打印部分预测结果
predictions = model.predict(test_dataset)
for prediction, good_rating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print(
        "Predicted good rating: {:.2%}".format(prediction[0]),
        " | Actual rating label: ",
        ("Good Rating" if bool(good_rating) else "Bad Rating"),
    )
