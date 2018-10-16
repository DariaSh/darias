import random
import math
import numpy as np
from sklearn import metrics


class ModelEvaluator:

    def __init__(self, user_id_col, item_id_col, rating_col, eval_sample_size):
        self.user_id_col = user_id_col
        self.item_id_col = item_id_col
        self.rating_col = rating_col
        self.eval_sample_size = eval_sample_size

    @staticmethod
    def user_apk(actual, predicted, k=0):
        if k == 0:
            k = len(actual)
        else:
            actual = actual[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        return score / min(len(actual), k)

    @staticmethod
    def dcg_at_k(scores, k):
        return scores[0] + sum(sc / math.log(ind, 2) for sc, ind in zip(scores[1:], range(2, k + 1)))

    def get_user_ndcg_k(self, actual, predicted, k=0):
        if k == 0:
            k = len(actual)

        k = min(len(actual), k)

        p_dcg = self.dcg_at_k(predicted, k)
        a_dcg = self.dcg_at_k(actual, k)

        return p_dcg / a_dcg

    def evaluate(self, model, test_data):
        return self.generate_global_metrics(model, test_data)

    def generate_user_metrics(self, actual_user_df, predicted_user_df):

        actual_ids = list(actual_user_df[self.item_id_col].values)
        predicted_ids = list(predicted_user_df[self.item_id_col].values)

        user_map_k_5 = self.user_apk(actual_ids, predicted_ids, 5)
        user_map_k_20 = self.user_apk(actual_ids, predicted_ids, 20)

        actual_ratings = list(actual_user_df[self.rating_col].values)
        avg_user_rating = np.array(actual_ratings).mean()

        predicted_ratings = []
        for a_item in actual_ids:
            if a_item in predicted_user_df[self.item_id_col].values:
                user_rating = \
                    predicted_user_df.loc[predicted_user_df[self.item_id_col] == a_item][self.rating_col].values[0]
                predicted_ratings.append(user_rating)
            else:
                predicted_ratings.append(avg_user_rating)

        user_ndcg_k_20 = self.get_user_ndcg_k(actual_ratings, predicted_ratings, 20)

        user_msa = metrics.mean_absolute_error(actual_ratings, predicted_ratings)
        user_rmse = math.sqrt(metrics.mean_squared_error(actual_ratings, predicted_ratings))

        return user_map_k_5, user_map_k_20, user_ndcg_k_20, user_msa, user_rmse

    def generate_global_metrics(self, model, test_data):
        map_5 = []
        map_20 = []
        ndcg_20 = []
        msa = []
        rmse = []

        user_id = set(test_data[self.user_id_col].unique())
        # select random subsample of users of a given subsample size
        user_id = random.sample(user_id, self.eval_sample_size)

        for uid in user_id:
            actual_user_df = test_data[test_data[self.user_id_col] == uid]
            actual_user_df = actual_user_df.sort_values(by=self.rating_col, ascending=False)

            predicted_user_df = model.predict(test_data, uid)

            user_map_k_5, user_map_k_20, ndcg_k_20, user_msa, user_rmse = self.generate_user_metrics(actual_user_df,
                                                                                                     predicted_user_df)

            map_5.append(user_map_k_5)
            map_20.append(user_map_k_20)
            ndcg_20.append(ndcg_k_20)
            msa.append(user_msa)
            rmse.append(user_rmse)

        global_metrics = {'MAP@5': np.array(map_5).mean(),
                          'MAP@20': np.array(map_20).mean(),
                          'NDCG@20': np.array(ndcg_20).mean(),
                          'MSA': np.array(msa).mean(),
                          'RMSE': np.array(rmse).mean()}

        return global_metrics
