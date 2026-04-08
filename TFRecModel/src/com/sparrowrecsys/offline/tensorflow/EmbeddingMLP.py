import tensorflow as tf
from pathlib import Path

# 训练集路径（如本地路径变化，请同步修改）
training_samples_origin = Path(
    r"C:\Users\32116\Desktop\ut\algo\resume\1\SparrowRecSys-Resume-1\src\main\resources\webroot\sampledata\trainingSamples.csv"
).as_uri()
training_samples_file_path = tf.keras.utils.get_file(
    "trainingSamples.csv", training_samples_origin
)

# 测试集路径（如本地路径变化，请同步修改）
test_samples_origin = Path(
    r"C:\Users\32116\Desktop\ut\algo\resume\1\SparrowRecSys-Resume-1\src\main\resources\webroot\sampledata\testSamples.csv"
).as_uri()
test_samples_file_path = tf.keras.utils.get_file(
    "testSamples.csv", test_samples_origin
)


# -----------------------------
# 数据集加载与特征定义
# -----------------------------
def get_dataset(file_path, shuffle=True):
    """将 CSV 文件读取为 TensorFlow Dataset。"""
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name="label",
        na_value="0",
        num_epochs=1,
        shuffle=shuffle,
        ignore_errors=True,
    )
    return dataset


# 构建训练集与测试集
train_dataset = get_dataset(training_samples_file_path, shuffle=True)
test_dataset = get_dataset(test_samples_file_path, shuffle=False)

# 类型特征词表
genre_vocab = [
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

GENRE_FEATURES = [
    "userGenre1",
    "userGenre2",
    "userGenre3",
    "userGenre4",
    "userGenre5",
    "movieGenre1",
    "movieGenre2",
    "movieGenre3",
]

NUMERIC_FEATURES = [
    "releaseYear",
    "movieRatingCount",
    "movieAvgRating",
    "movieRatingStddev",
    "userRatingCount",
    "userAvgRating",
    "userRatingStddev",
]

ALL_MODEL_FEATURES = GENRE_FEATURES + ["movieId", "userId"] + NUMERIC_FEATURES


def select_model_features(features, label):
    """筛选模型输入特征，并将标签转为 float32。"""
    selected = {name: features[name] for name in ALL_MODEL_FEATURES}
    return selected, tf.cast(label, tf.float32)


train_dataset = train_dataset.map(select_model_features)
test_dataset = test_dataset.map(select_model_features)

# -----------------------------
# 模型结构定义（Keras 3 兼容）
# -----------------------------
inputs = {}

# 类别特征输入（字符串）
for feature in GENRE_FEATURES:
    inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.string)

# ID 特征输入（整数）
inputs["movieId"] = tf.keras.Input(shape=(1,), name="movieId", dtype=tf.int64)
inputs["userId"] = tf.keras.Input(shape=(1,), name="userId", dtype=tf.int64)

# 数值特征输入（浮点）
for feature in NUMERIC_FEATURES:
    inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)

genre_lookup = tf.keras.layers.StringLookup(
    vocabulary=genre_vocab,
    mask_token=None,
    num_oov_indices=1,
)
genre_embedding_layer = tf.keras.layers.Embedding(
    input_dim=genre_lookup.vocabulary_size(),
    output_dim=10,
)

# 对每个类型特征做 StringLookup + Embedding + Flatten
genre_embeddings = []
for feature in GENRE_FEATURES:
    genre_ids = genre_lookup(inputs[feature])
    genre_vec = tf.keras.layers.Flatten()(genre_embedding_layer(genre_ids))
    genre_embeddings.append(genre_vec)

# movieId 与 userId 各自单独做 Embedding
movie_embedding = tf.keras.layers.Flatten()(
    tf.keras.layers.Embedding(1001, 10)(inputs["movieId"])
)
user_embedding = tf.keras.layers.Flatten()(
    tf.keras.layers.Embedding(30001, 10)(inputs["userId"])
)

# 合并所有特征张量：数值特征 + 类型特征向量 + ID 向量
all_feature_tensors = (
    [inputs[name] for name in NUMERIC_FEATURES]
    + genre_embeddings
    + [movie_embedding, user_embedding]
)
concat_features = tf.keras.layers.Concatenate()(all_feature_tensors)

# MLP 主干
x = tf.keras.layers.Dense(128, activation="relu")(concat_features)
x = tf.keras.layers.Dense(128, activation="relu")(x)
output = tf.keras.layers.Dense(1, activation="sigmoid")(x)

model = tf.keras.Model(inputs=inputs, outputs=output)

# -----------------------------
# 训练与评估
# -----------------------------
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

# 训练模型
model.fit(train_dataset, epochs=5)

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
