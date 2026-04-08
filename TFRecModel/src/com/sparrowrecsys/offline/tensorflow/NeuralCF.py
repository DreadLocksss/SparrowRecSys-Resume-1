import argparse
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
HIDDEN_UNITS = [10, 10]


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


# 训练集训练 5 个 epoch；测试集只评估 1 轮
train_dataset = get_dataset(training_samples_file_path, shuffle=True, num_epochs=1)
test_dataset = get_dataset(test_samples_file_path, shuffle=False, num_epochs=1)

# 仅保留 NeuralCF 需要的输入字段，避免多余列干扰模型输入映射
MODEL_INPUT_KEYS = ["movieId", "userId"]


def keep_model_inputs(features, label):
    """筛选模型输入特征，并将标签转为 float32。"""
    selected = {key: features[key] for key in MODEL_INPUT_KEYS}
    return selected, tf.cast(label, tf.float32)


train_dataset = train_dataset.map(keep_model_inputs)
test_dataset = test_dataset.map(keep_model_inputs)


# -----------------------------
# 模型输入定义
# -----------------------------
inputs = {
    "movieId": tf.keras.Input(name="movieId", shape=(1,), dtype=tf.int64),
    "userId": tf.keras.Input(name="userId", shape=(1,), dtype=tf.int64),
}


def build_id_embedding(input_tensor, bucket_size, embedding_dim):
    """构建 ID 特征的 Embedding 向量并展平为二维张量。"""
    embedding = tf.keras.layers.Embedding(bucket_size, embedding_dim)(input_tensor)
    return tf.keras.layers.Flatten()(embedding)


# neural cf 模型结构一：双塔 embedding 拼接后进入 MLP
def neural_cf_model_1(feature_inputs, hidden_units):
    movie_tower = build_id_embedding(
        feature_inputs["movieId"],
        bucket_size=MOVIE_ID_BUCKET_SIZE,
        embedding_dim=EMBEDDING_DIM,
    )
    user_tower = build_id_embedding(
        feature_inputs["userId"],
        bucket_size=USER_ID_BUCKET_SIZE,
        embedding_dim=EMBEDDING_DIM,
    )

    interact_layer = tf.keras.layers.Concatenate()([movie_tower, user_tower])
    for num_nodes in hidden_units:
        interact_layer = tf.keras.layers.Dense(num_nodes, activation="relu")(interact_layer)

    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(interact_layer)
    return tf.keras.Model(feature_inputs, output_layer, name="neural_cf_model_1")


# neural cf 模型结构二：双塔各自 MLP 后做点积
def neural_cf_model_2(feature_inputs, hidden_units):
    movie_tower = build_id_embedding(
        feature_inputs["movieId"],
        bucket_size=MOVIE_ID_BUCKET_SIZE,
        embedding_dim=EMBEDDING_DIM,
    )
    user_tower = build_id_embedding(
        feature_inputs["userId"],
        bucket_size=USER_ID_BUCKET_SIZE,
        embedding_dim=EMBEDDING_DIM,
    )

    for num_nodes in hidden_units:
        movie_tower = tf.keras.layers.Dense(num_nodes, activation="relu")(movie_tower)
    for num_nodes in hidden_units:
        user_tower = tf.keras.layers.Dense(num_nodes, activation="relu")(user_tower)

    dot_output = tf.keras.layers.Dot(axes=1)([movie_tower, user_tower])
    output_layer = tf.keras.layers.Dense(1, activation="sigmoid")(dot_output)
    return tf.keras.Model(feature_inputs, output_layer, name="neural_cf_model_2")


def parse_model_type():
    """解析命令行参数，决定使用哪种 NeuralCF 结构。"""
    parser = argparse.ArgumentParser(description="NeuralCF 训练脚本")
    parser.add_argument(
        "model_type",
        nargs="?",
        default=1,
        type=int,
        choices=[1, 2],
        help="模型编号：1 表示模型结构一；2 表示模型结构二（默认 1）",
    )
    args = parser.parse_args()
    return args.model_type


def build_model(model_type):
    """根据命令行参数构建模型。"""
    if model_type == 1:
        return neural_cf_model_1(inputs, HIDDEN_UNITS)
    return neural_cf_model_2(inputs, HIDDEN_UNITS)


def main():
    # 用命令行参数选择模型：1 -> 模型1，2 -> 模型2
    model_type = parse_model_type()
    model = build_model(model_type)
    print(f"使用 NeuralCF 模型结构: {model_type}")

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

    # 保存模型（Keras 3 推荐保存为 .keras 文件）
    model_output_dir = (
        PROJECT_ROOT / "src" / "main" / "resources" / "webroot" / "modeldata" / "neuralcf" / "002"
    )
    model_output_dir.mkdir(parents=True, exist_ok=True)
    model_output_path = model_output_dir / "neuralcf.keras"
    model.save(str(model_output_path))

if __name__ == "__main__":
    main()


