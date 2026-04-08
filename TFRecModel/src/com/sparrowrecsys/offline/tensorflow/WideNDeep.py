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
def get_dataset(file_path, shuffle=True, num_epochs=1):
    """将 CSV 文件读取为 TensorFlow Dataset。"""
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name="label",
        na_value="0",
        num_epochs=num_epochs,
        shuffle=shuffle,
        ignore_errors=True,
    )
    return dataset


# 构建训练集与测试集
# 训练集：每次遍历 1 轮，训练总轮次由 model.fit(epochs=5) 控制
train_dataset = get_dataset(training_samples_file_path, shuffle=True, num_epochs=1)
# 测试集：只读取 1 轮，evaluate/predict 只执行一遍
test_dataset = get_dataset(test_samples_file_path, shuffle=False, num_epochs=1)

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

# -----------------------------
# 模型输入定义
# -----------------------------
inputs = {
    # 数值特征
    "movieAvgRating": tf.keras.layers.Input(
        name="movieAvgRating", shape=(), dtype="float32"
    ),
    "movieRatingStddev": tf.keras.layers.Input(
        name="movieRatingStddev", shape=(), dtype="float32"
    ),
    "movieRatingCount": tf.keras.layers.Input(
        name="movieRatingCount", shape=(), dtype="int32"
    ),
    "userAvgRating": tf.keras.layers.Input(
        name="userAvgRating", shape=(), dtype="float32"
    ),
    "userRatingStddev": tf.keras.layers.Input(
        name="userRatingStddev", shape=(), dtype="float32"
    ),
    "userRatingCount": tf.keras.layers.Input(
        name="userRatingCount", shape=(), dtype="int32"
    ),
    "releaseYear": tf.keras.layers.Input(name="releaseYear", shape=(), dtype="int32"),
    # ID 特征
    "movieId": tf.keras.layers.Input(name="movieId", shape=(), dtype="int32"),
    "userId": tf.keras.layers.Input(name="userId", shape=(), dtype="int32"),
    "userRatedMovie1": tf.keras.layers.Input(
        name="userRatedMovie1", shape=(), dtype="int32"
    ),
    # 类型特征
    "userGenre1": tf.keras.layers.Input(name="userGenre1", shape=(), dtype="string"),
    "userGenre2": tf.keras.layers.Input(name="userGenre2", shape=(), dtype="string"),
    "userGenre3": tf.keras.layers.Input(name="userGenre3", shape=(), dtype="string"),
    "userGenre4": tf.keras.layers.Input(name="userGenre4", shape=(), dtype="string"),
    "userGenre5": tf.keras.layers.Input(name="userGenre5", shape=(), dtype="string"),
    "movieGenre1": tf.keras.layers.Input(name="movieGenre1", shape=(), dtype="string"),
    "movieGenre2": tf.keras.layers.Input(name="movieGenre2", shape=(), dtype="string"),
    "movieGenre3": tf.keras.layers.Input(name="movieGenre3", shape=(), dtype="string"),
}

MODEL_INPUT_KEYS = list(inputs.keys())


def keep_model_inputs(features, label):
    """仅保留模型声明过的输入特征。"""
    return {k: features[k] for k in MODEL_INPUT_KEYS}, label


train_dataset = train_dataset.map(keep_model_inputs)
test_dataset = test_dataset.map(keep_model_inputs)

genre_lookup = tf.keras.layers.StringLookup(
    vocabulary=genre_vocab,
    mask_token=None,
    num_oov_indices=1,
)
genre_embedding_layer = tf.keras.layers.Embedding(
    input_dim=genre_lookup.vocabulary_size(),
    output_dim=10,
)

# 类型特征：StringLookup + Embedding + Flatten
genre_embeddings = []
for feature in GENRE_FEATURES:
    genre_ids = genre_lookup(inputs[feature])
    genre_vec = tf.keras.layers.Flatten()(genre_embedding_layer(genre_ids))
    genre_embeddings.append(genre_vec)

# movieId 与 userId 的 Embedding
movie_embedding = tf.keras.layers.Flatten()(
    tf.keras.layers.Embedding(1001, 10)(inputs["movieId"])
)
user_embedding = tf.keras.layers.Flatten()(
    tf.keras.layers.Embedding(30001, 10)(inputs["userId"])
)

# 数值特征 reshape 后与 embedding 特征拼接
numeric_tensors = [tf.keras.layers.Reshape((1,))(inputs[name]) for name in NUMERIC_FEATURES]
all_feature_tensors = numeric_tensors + genre_embeddings + [movie_embedding, user_embedding]
concat_features = tf.keras.layers.Concatenate()(all_feature_tensors)

# 当前电影与用户历史电影的交叉特征（wide 部分）
crossed_feature = tf.keras.layers.HashedCrossing(num_bins=10000)(
    [inputs["movieId"], inputs["userRatedMovie1"]]
)

# movie_feature = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=1001)
# rated_movie_feature = tf.feature_column.categorical_column_with_identity(key='userRatedMovie1', num_buckets=1001)
# crossed_feature = tf.feature_column.indicator_column(tf.feature_column.crossed_column([movie_col, rated_movie], 10000))

# -----------------------------
# Wide & Deep 模型结构
# -----------------------------
# deep 分支：处理稠密特征与 embedding 拼接结果
deep = tf.keras.layers.Dense(128, activation="relu")(concat_features)
deep = tf.keras.layers.Dense(128, activation="relu")(deep)


# wide 分支：对交叉特征做 one-hot 编码
wide = tf.keras.layers.CategoryEncoding(
    num_tokens=10000, output_mode="one_hot"
)(crossed_feature)

# wide = tf.keras.layers.DenseFeatures(crossed_feature)(inputs)


both = tf.keras.layers.concatenate([deep, wide])
output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(both)
model = tf.keras.Model(inputs, output_layer)

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

# 训练模型（训练集共训练 5 个 epoch）
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
