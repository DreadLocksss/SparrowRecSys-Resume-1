from pathlib import Path

import tensorflow as tf

# -----------------------------
# 常量定义
# -----------------------------
BATCH_SIZE = 12
EPOCHS = 5
EMBEDDING_DIM = 10
MOVIE_ID_BUCKET_SIZE = 1001
USER_ID_BUCKET_SIZE = 30001

NUMERIC_FEATURES = [
    "releaseYear",
    "movieRatingCount",
    "movieAvgRating",
    "movieRatingStddev",
    "userRatingCount",
    "userAvgRating",
    "userRatingStddev",
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
# 说明：仅保留原始代码中实际参与 DeepFM 计算的字段
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


def to_float_column(tensor):
    """将标量特征转换为 shape=(1,) 的 float32 张量。"""
    casted = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.float32))(tensor)
    return tf.keras.layers.Reshape((1,))(casted)


def int_to_one_hot(tensor, depth):
    """将整型 ID 特征转换为 one-hot（等价于原始 indicator_column）。"""
    indices = tf.keras.layers.Lambda(lambda x: tf.cast(x, tf.int32))(tensor)
    return tf.keras.layers.CategoryEncoding(
        num_tokens=depth,
        output_mode="one_hot",
    )(indices)


# -----------------------------
# 类别编码与 Embedding
# -----------------------------
genre_lookup = tf.keras.layers.StringLookup(
    vocabulary=GENRE_VOCAB,
    mask_token=None,
    num_oov_indices=1,
)

movie_embedding_layer = tf.keras.layers.Embedding(
    input_dim=MOVIE_ID_BUCKET_SIZE,
    output_dim=EMBEDDING_DIM,
)
user_embedding_layer = tf.keras.layers.Embedding(
    input_dim=USER_ID_BUCKET_SIZE,
    output_dim=EMBEDDING_DIM,
)

# 保持原语义：用户类型和电影类型使用独立的 embedding 参数
user_genre_embedding_layer = tf.keras.layers.Embedding(
    input_dim=genre_lookup.vocabulary_size(),
    output_dim=EMBEDDING_DIM,
)
item_genre_embedding_layer = tf.keras.layers.Embedding(
    input_dim=genre_lookup.vocabulary_size(),
    output_dim=EMBEDDING_DIM,
)

# ID/类型的一阶 one-hot（对应原始 fm_first_order_columns）
movie_id_one_hot = int_to_one_hot(inputs["movieId"], MOVIE_ID_BUCKET_SIZE)
user_id_one_hot = int_to_one_hot(inputs["userId"], USER_ID_BUCKET_SIZE)

genre_one_hot_layer = tf.keras.layers.CategoryEncoding(
    num_tokens=genre_lookup.vocabulary_size(),
    output_mode="one_hot",
)
user_genre1_ids = genre_lookup(inputs["userGenre1"])
movie_genre1_ids = genre_lookup(inputs["movieGenre1"])
user_genre1_one_hot = genre_one_hot_layer(user_genre1_ids)
movie_genre1_one_hot = genre_one_hot_layer(movie_genre1_ids)

# FM 一阶项
fm_first_order_layer = tf.keras.layers.Concatenate()(
    [movie_id_one_hot, user_id_one_hot, user_genre1_one_hot, movie_genre1_one_hot]
)

# 二阶交叉所需的 embedding 向量
movie_emb = movie_embedding_layer(inputs["movieId"])
user_emb = user_embedding_layer(inputs["userId"])
movie_genre_emb = item_genre_embedding_layer(movie_genre1_ids)
user_genre_emb = user_genre_embedding_layer(user_genre1_ids)

# FM 二阶交叉项（保持原始 4 组点积语义）
product_layer_item_user = tf.keras.layers.Dot(axes=1)([movie_emb, user_emb])
product_layer_item_genre_user_genre = tf.keras.layers.Dot(axes=1)(
    [movie_genre_emb, user_genre_emb]
)
product_layer_item_genre_user = tf.keras.layers.Dot(axes=1)([movie_genre_emb, user_emb])
product_layer_user_genre_item = tf.keras.layers.Dot(axes=1)([movie_emb, user_genre_emb])


# -----------------------------
# Deep 部分（数值特征 + movie/user embedding）
# -----------------------------
numeric_tensors = [to_float_column(inputs[name]) for name in NUMERIC_FEATURES]
deep_inputs = tf.keras.layers.Concatenate()(numeric_tensors + [movie_emb, user_emb])
deep = tf.keras.layers.Dense(64, activation="relu")(deep_inputs)
deep = tf.keras.layers.Dense(64, activation="relu")(deep)


# -----------------------------
# 输出层
# -----------------------------
concat_layer = tf.keras.layers.Concatenate(axis=1)(
    [
        fm_first_order_layer,
        product_layer_item_user,
        product_layer_item_genre_user_genre,
        product_layer_item_genre_user,
        product_layer_user_genre_item,
        deep,
    ]
)
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(concat_layer)

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
