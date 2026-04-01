import tensorflow as tf
from pathlib import Path

# Training samples path, change to your local path
training_samples_origin = Path(
    r"C:\Users\32116\Desktop\ut\algo\resume\1\SparrowRecSys-Resume-1\src\main\resources\webroot\sampledata\trainingSamples.csv"
).as_uri()
training_samples_file_path = tf.keras.utils.get_file(
    "trainingSamples.csv", training_samples_origin
)

# Test samples path, change to your local path
test_samples_origin = Path(
    r"C:\Users\32116\Desktop\ut\algo\resume\1\SparrowRecSys-Resume-1\src\main\resources\webroot\sampledata\testSamples.csv"
).as_uri()
test_samples_file_path = tf.keras.utils.get_file(
    "testSamples.csv", test_samples_origin
)


# load sample as tf dataset
def get_dataset(file_path, shuffle=True):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=12,
        label_name='label',
        na_value="0",
        num_epochs=1,
        shuffle=shuffle,
        ignore_errors=True)
    return dataset


# split as test dataset and training dataset
train_dataset = get_dataset(training_samples_file_path, shuffle=True)
test_dataset = get_dataset(test_samples_file_path, shuffle=False)

# genre features vocabulary
genre_vocab = ['Film-Noir', 'Action', 'Adventure', 'Horror', 'Romance', 'War', 'Comedy', 'Western', 'Documentary',
               'Sci-Fi', 'Drama', 'Thriller',
               'Crime', 'Fantasy', 'Animation', 'IMAX', 'Mystery', 'Children', 'Musical']

GENRE_FEATURES = [
    'userGenre1',
    'userGenre2',
    'userGenre3',
    'userGenre4',
    'userGenre5',
    'movieGenre1',
    'movieGenre2',
    'movieGenre3'
]

NUMERIC_FEATURES = [
    'releaseYear',
    'movieRatingCount',
    'movieAvgRating',
    'movieRatingStddev',
    'userRatingCount',
    'userAvgRating',
    'userRatingStddev'
]

ALL_MODEL_FEATURES = GENRE_FEATURES + ['movieId', 'userId'] + NUMERIC_FEATURES


def select_model_features(features, label):
    selected = {name: features[name] for name in ALL_MODEL_FEATURES}
    return selected, tf.cast(label, tf.float32)


train_dataset = train_dataset.map(select_model_features)
test_dataset = test_dataset.map(select_model_features)

# Keras 3 compatible model architecture
inputs = {}
for feature in GENRE_FEATURES:
    inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.string)

inputs['movieId'] = tf.keras.Input(shape=(1,), name='movieId', dtype=tf.int64)
inputs['userId'] = tf.keras.Input(shape=(1,), name='userId', dtype=tf.int64)

for feature in NUMERIC_FEATURES:
    inputs[feature] = tf.keras.Input(shape=(1,), name=feature, dtype=tf.float32)

genre_lookup = tf.keras.layers.StringLookup(
    vocabulary=genre_vocab,
    mask_token=None,
    num_oov_indices=1
)
genre_embedding_layer = tf.keras.layers.Embedding(
    input_dim=genre_lookup.vocabulary_size(),
    output_dim=10
)

genre_embeddings = []
for feature in GENRE_FEATURES:
    genre_ids = genre_lookup(inputs[feature])
    genre_vec = tf.keras.layers.Flatten()(genre_embedding_layer(genre_ids))
    genre_embeddings.append(genre_vec)

movie_embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(1001, 10)(inputs['movieId']))
user_embedding = tf.keras.layers.Flatten()(tf.keras.layers.Embedding(30001, 10)(inputs['userId']))

all_feature_tensors = [inputs[name] for name in NUMERIC_FEATURES] + genre_embeddings + [movie_embedding, user_embedding]
concat_features = tf.keras.layers.Concatenate()(all_feature_tensors)

x = tf.keras.layers.Dense(128, activation='relu')(concat_features)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=output)

# compile the model, set loss function, optimizer and evaluation metrics
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy', tf.keras.metrics.AUC(curve='ROC'), tf.keras.metrics.AUC(curve='PR')])

# train the model
model.fit(train_dataset, epochs=5)

# evaluate the model
test_loss, test_accuracy, test_roc_auc, test_pr_auc = model.evaluate(test_dataset)
print('\n\nTest Loss {}, Test Accuracy {}, Test ROC AUC {}, Test PR AUC {}'.format(test_loss, test_accuracy,
                                                                                   test_roc_auc, test_pr_auc))

# print some predict results
predictions = model.predict(test_dataset)
for prediction, goodRating in zip(predictions[:12], list(test_dataset)[0][1][:12]):
    print("Predicted good rating: {:.2%}".format(prediction[0]),
          " | Actual rating label: ",
          ("Good Rating" if bool(goodRating) else "Bad Rating"))