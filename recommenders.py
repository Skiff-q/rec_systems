import pandas as pd
import numpy as np

# Для работы с матрицами
from scipy.sparse import csr_matrix

# Матричная факторизация
from implicit.als import AlternatingLeastSquares
from implicit.nearest_neighbours import ItemItemRecommender  # нужен для одного трюка
from implicit.nearest_neighbours import bm25_weight, tfidf_weight


class MainRecommender:
    '''
    Рекоммендации, которые можно получить из ALS

    Input
    -----
    user_item_matrix: pd.DataFrame
        Матрица взаимодействий user-item
    '''

    def __init__(self, data, weighting=None):

        # Топ покупок по всему датасету
        self.overall_top_purchases = data.groupby('item_id')['quantity'].count().reset_index()
        self.overall_top_purchases.sort_value('quantity', ascending=False, inplace=True)
        self.overall_top_purchases = self.overall_top_purchases.item_id.tolist()

        self.user_item_matrix = self.prepare_matrix(data)
        self.id_to_itemid, self.id_to_userid, \
            self.itemid_to_id, self.userid_to_id = self.prepare_dicts(self.user_item_matrix)

        if weighting == 'bm25':
            self.user_item_matrix = bm25_weight(self.user_item_matrix.T)
        elif weighting == 'tfidf':
            self.user_item_matrix = tfidf_weight(self.user_item_matrix.T)

        self.model = self.fit(self.user_item_matrix)
        self.own_recommender = self.fit_own_recommender(self.user_item_matrix)

    @staticmethod
    def prepare_matrix(data: pd.DataFrame):
        user_item_matrix = pd.pivot_table(data,
                                          index='user_id', columns='item_id',
                                          values='quantity',
                                          aggfunc='count',
                                          fill_value=0)
        user_item_matrix = user_item_matrix.astype(float)
        return user_item_matrix

    @staticmethod
    def prepare_dicts(user_item_matrix):
        '''Подготавливает вспомогательные словари'''

        # перенумеруем пользователей и товары,
        # чтобы они задавались числами от 0 до номера количества товаров.

        # userids array содержит индексы user_id, какие-то значения могут быть пропущены
        userids = user_item_matrix.index.values

        # itemids array содержит индексы item_id, какие-то значения могут быть пропущены
        itemids = user_item_matrix.columns.values

        # Формируем матрицы с последовательными числами, т.е. без пропусков
        # номеров, перенумеруем для задания последовательных номеров userids, itemids
        # числами от 0 до фактического количества userids, itemids

        matrix_userids = np.arange(len(userids))
        matrix_itemids = np.arange(len(itemids))

        # Формируем словари, где ключ это перенумерованный порядковый номер userids, itemids
        # значение ключа это фактическое значение из данных userids, itemids

        id_to_itemid = dict(zip(matrix_itemids, itemids))
        id_to_userid = dict(zip(matrix_userids, userids))

        itemid_to_id = dict(zip(itemids, matrix_itemids))
        userid_to_id = dict(zip(userids, matrix_userids))

        # Другими словами, ставим фактическому номеру его порядковое значение
        #  от 0 до n, подставляем в модель и снова переходим от порядковог номера к фактическому

        return id_to_itemid, id_to_userid, itemid_to_id, userid_to_id

    @staticmethod
    def fit_own_recommender(user_item_matrix):
        '''Обучаем модель, которая рекомендует товары, среди товаров, купленных юзером'''

        own_recommender = ItemItemRecommender(K=1, num_threads=4)
        own_recommender.fit(user_item_matrix)

        return own_recommender

    @staticmethod
    def fit(user_item_matrix, n_factors=20, regularization=0.001, iterations=15, calculate_training_loss=True, num_threads=4):
        '''Обучаем ALS'''

        model = AlternatingLeastSquares(factors=n_factors,
                                        regularization=regularization,
                                        iterations=iterations,
                                        calculate_training_loss=calculate_training_loss,
                                        num_threads=num_threads)

        model.fit(csr_matrix(user_item_matrix).T.tocsr())

        return model

    def _update_dict(self, user_id):
        '''Если появится новый user / item, то нужно обновить словари'''

        if user_id not in self.userid_to_id.keys():

            max_id = max(list(self.userid_to_id.values()))
            max_id += 1

            self.userid_to_id.update({user_id: max_id})
            self.id_to_userid.update({max_id: user_id})

    def _get_similar_item(self, item_id):
        '''Находится товар, похожий на item_id'''
        recs = self.model.similar_items(self.itemid_to_id[item_id], N=2) # Товар похожий на себя -> рекомендуем 2 товара
        top_rec = recs[1][0]  # Берем второй (не товар из аргумента метода)
        return self.id_to_itemid[top_rec]

    def _extend_with_top_popular(self, recommendations, N=5):
        '''Если кол-во рекоммендаций < N, то дополняем их топ-популярными'''

        if len(recommendations) < N:
            top_popular = [rec for rec in self.overall_top_purchases[:N] if rec not in recommendations]
            recommendations.extend(top_popular)
            recommendations = recommendations[:N]

        return recommendations

    def _get_recommendations(self, user, model, N=5):
        '''Рекомендации через стандартные библиотеки implicit'''

        self._update_dict(user_id=user)
        res = model.recommend(userid=self.userid_to_id[user],
                              user_items=self.user_item_matrix[self.userid_to_id[user]],
                              N=N,
                              filter_already_liked_items=False,
                              recalculate_user=True)
        mask = res[1].argsort()[::-1]
        res = [self.id_to_itemid[rec] for rec in res[0][mask]]
        res = self._extend_with_top_popular(res, N=N)

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_als_recommendations(self, user, N=5):
        '''Рекомендации через стандартные библиотеки implicit'''

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.model, N=N)

    def get_own_recommendations(self, user, N=5):
        '''Рекомендуем товары среди тех, которые юзер уже купил'''

        self._update_dict(user_id=user)
        return self._get_recommendations(user, model=self.own_recommender, N=N)

    def get_similar_users_recommendation(self, user, N=5):
        '''Рекомендуем топ-N товаров, среди купленных похожими юзерами'''

        # Получаем индентификатор пользователя из его имени
        user_id = self.userid_to_id[user]

        # Получаем список похожих пользователей
        similar_users = self.own_recommender.similar_users(user_id, N+1)

        # Удаляем исходного пользователя из списка похожих пользователей
        similar_users = similar_users[1:]

        # Получаем список товаров, купленных похожими пользователями
        items = []
        for similar_user_id in similar_users:
            recs = self.own_recommender.recommend(similar_user_id, self.user_item_matrix.T.tocsr(), N=1)
            items.append(self.id_to_itemid[recs[0][0]])

        # удаляем дубликаты и возвращаем список рукомендаций
        res = list(set(items))

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res

    def get_similar_items_recommendation(self, user, N=5):
        '''Рекомендуем товары, похожие на топ-N купленных юзером товаров'''

        # Получаем идентификатор пользователя из его имени
        user_id = self.userid_to_id[user]

        # Получаем топ-N товаров, купленных пользователем
        top_items = self.user_item_matrix.loc[user_id].sort_value(ascending=False).head(N)

        # Получаем похожие товары для каждого из топ-N товаров
        similar_items = []
        for item_id, score in top_items.iteritems():
            recs = self.model.similar_items(self.itemid_to_id[item_id], N=2)
            similar_items.extend([self.id_to_itemid[rec[0]] for rec in recs[1:]])

        # Удаляем дубликаты и возвращаем список рукомендаций
        res = list(set(similar_items))

        assert len(res) == N, 'Количество рекомендаций != {}'.format(N)
        return res
