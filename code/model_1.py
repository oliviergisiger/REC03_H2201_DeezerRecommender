import pandas as pd
import numpy as np
import tensorflow as tf
import tensorflow_recommenders as tfrs
import keras
from tensorflow.keras.optimizers.legacy import Adam

from datetime import datetime

np.set_printoptions(precision=3, suppress=True)


class DeezerData:

    # setting constant CUT: for code tests
    CUT = 100000

    def __init__(self, train_data_path, test_data_path):
        self._df_train = pd.read_csv(train_data_path)
        self._df_test = pd.read_csv(test_data_path)
        self._ratings = self._get_ratings(self._df_train)
        self.dataset = self._get_tf_dataset(self._ratings).map(self._rename)
        self.songs = self._ratings.song_id.values
        self.users = self._ratings.user_id.values
        self.unique_song_ids = np.unique(list(self.songs))
        self.unique_user_ids = np.unique(list(self.users))



    def _get_ratings(self, df):

        return pd.DataFrame(
            {
                'user_id': np.array(df['user_id'][:self.CUT].astype(np.str_)),
                'song_id': np.array(df['media_id'][:self.CUT].astype(np.str_)),
                'rating': np.array(df['is_listened'][:self.CUT].astype(np.float32)),

            }
        )


    def _get_tf_dataset(self, ratings):

        return tf.data.Dataset.from_tensor_slices(
            (
                tf.cast(
                    ratings['user_id'].values.reshape(
                        -1, 1), tf.string), tf.cast(ratings['song_id'].values.reshape(
                    -1, 1), tf.string),
                tf.cast(ratings['rating'].values.reshape(-1, 1), tf.float32)
            )
        )


    @staticmethod
    @tf.function
    def _rename(x0, x1, x2):
        y = {}
        y["user_id"] = x0
        y['song_id'] = x1
        y['rating'] = x2
        return y



class RankingModel(tf.keras.Model):

    def __init__(self, unique_user_ids, unique_song_ids):
        super().__init__()
        embedding_dimension = 32
        unique_user_ids = unique_user_ids
        unique_song_ids = unique_song_ids

        # Compute embeddings for users.
        self.user_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

        # Compute embeddings for songs.
        self.song_embeddings = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_song_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_song_ids) + 1, embedding_dimension)
        ])

        # Compute predictions.
        self.ratings = tf.keras.Sequential([
            # Learn multiple dense layers.
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(64, activation="relu"),
            # Make rating predictions in the final layer.
            tf.keras.layers.Dense(1)
        ])

    def __call__(self, x):
        user_id, song_id = x
        user_embedding = self.user_embeddings(user_id)
        song_embedding = self.song_embeddings(song_id)

        return self.ratings(tf.concat([user_embedding, song_embedding], axis=1))


class DeezerModel(tfrs.models.Model):

    def __init__(self, unique_user_ids, unique_song_ids):
        super().__init__()
        self.ranking_model: tf.keras.Model = RankingModel(unique_user_ids=unique_user_ids,
                                                          unique_song_ids=unique_song_ids)
        self.task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss = tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()]
        )

    def compute_loss(self, features, training=False) -> tf.Tensor:
        print(features)
        rating_predictions = self.ranking_model((features['user_id'], features["song_id"]))

        # The task computes the loss and the metrics.
        return self.task(labels=features["rating"], predictions=rating_predictions)



if __name__ == '__main__':

    data = DeezerData(
        train_data_path='../data/train.csv',
        test_data_path='../data/test.csv'
    )

    model = DeezerModel(
        unique_user_ids=data.unique_user_ids,
        unique_song_ids=data.unique_song_ids
    )

    model.compile(optimizer=Adam(learning_rate=0.5),
                  metrics=['accuracy'])
    cache_dataset = data.dataset.cache()

    # setup tensorboard
    logdir = "../logs/tf_model_1/"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir,
                                                       histogram_freq=1)

    # train
    model.fit(cache_dataset, epochs=1,
              verbose=1, callbacks=[tensorboard_callback])

