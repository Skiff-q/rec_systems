import pandas as pd
import numpy as np


def prefilter_items(data_train, item_features, cut_length=3, sales_range=(10, 100)):
    '''

    :param data_train:
    :param item_features:
    :param cut_length:
    :param sales_range:
    :return:
    Функция prefilter_items выполняет предварительную фильтрацию данных с целью подготовки данных для рекомендаций.
    Она выполняет следующие действия:

    - Удаляет неинтересные для рекомендаций категории товаров на основе информации о категориях (item_features).
    - Редкие категории, в которых количество уникальных товаров меньше 150, считаются неинтересными,
    и все товары из этих категорий удаляются из данных.
    - Удаляет слишком дешевые и дорогие товары, так как на них не заработаем.
    - Удаляет самые популярные товары
    - Удаляет самые не популярные товары
    - Удаляет товары, которые не продавались за последние 12 месяцев

    '''
    data_train = pd.merge(data_train, item_features, on='item_id')

    # Уберем самые популярные товары (их и так купят)
    popularity = data_train.groupby('item_id')['user_id'].nunique().reset_index() / data_train['user_id'].nunique()
    popularity.rename(columns={'user_id': 'share_unique_users'}, inplace=True)

    top_popular = popularity[popularity['share_unique_users'] > 0.5].item_id.tolist()
    data_train = data_train[~data_train['item_id'].isin(top_popular)]

    # Уберем самые НЕ популярные товары (их и так НЕ купят)
    top_notpopular = popularity[popularity['share_unique_users'] < 0.01].item_id.tolist()
    data_train = data_train[~data_train['item_id'].isin(top_notpopular)]

    # Уберем товары, которые не продавались за последние 12 месяцев
    data_train = data_train[
        data_train['week_no'] > data_train['week_no'].max() - 52]

    # Уберем не интересные для рекоммендаций категории (department)
    to_del = ['GROCERY', 'MISC. TRANS.', 'PASTRY',
              'DRUG GM', 'MEAT-PCKGD',
              'SEAFOOD-PCKGD', 'PRODUCE',
              'NUTRITION', 'DELI', 'COSMETICS']
    data_train = data_train[~(data_train['department'].isin(to_del))]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб.
    data_train = data_train[~(data_train['sales_value'] < sales_range[0])]

    # Уберем слишком дорогие товары
    data = data_train[~(data_train['sales_value'] > sales_range[1])]

    return data


def prefilter_items_top_5000(data, take_n_popular=5000, item_features=None):
    '''

    :param data:
    :param take_n_popular:
    :param item_features:
    :return:
    Функция prefilter_items выполняет предварительную фильтрацию данных с целью подготовки данных для рекомендаций.
    Она выполняет следующие действия:

    - Удаляет неинтересные для рекомендаций категории товаров на основе информации о категориях (item_features).
    - Редкие категории, в которых количество уникальных товаров меньше 150, считаются неинтересными, и все товары из этих категорий удаляются из данных.
    - Удаляет слишком дешевые товары, так как на них не заработаем.
    -Товары, у которых стоимость (sales_value) деленная на количество продаж (quantity) меньше или равна 2, удаляются из данных.
    - Удаляет слишком дорогие товары. Товары, у которых стоимость (sales_value) больше 50, удаляются из данных.

    Оставляет только топ-N популярных товаров на основе суммарного количества продаж (quantity).
    Топ-N товаров с наибольшим количеством продаж остаются в данных, остальные удаляются. Значение N задается параметром take_n_popular.
    Заменяет item_id для всех товаров, которые не входят в топ-N популярных товаров, на фиктивный item_id 999999.
    Это позволяет учесть факт покупки любого товара, который не попал в топ-N, как покупку фиктивного товара.
    '''
    # Уберем не интересные для рекоммендаций категории (department):
    if item_features is not None:
        department_size = pd.DataFrame(item_features. \
                                       groupby('department')['item_id'].nunique(). \
                                       sort_values(ascending=False)).reset_index()

        department_size.columns = ['department', 'n_items']
        rare_departments = department_size[department_size['n_items'] < 150].department.tolist()
        items_in_rare_departments = item_features[
            item_features['department'].isin(rare_departments)].item_id.unique().tolist()

        data = data[~data['item_id'].isin(items_in_rare_departments)]

    # Уберем слишком дешевые товары (на них не заработаем). 1 покупка из рассылок стоит 60 руб:
    data['price'] = data['sales_value'] / (np.maximum(data['quantity'], 1))
    data = data[data['price'] > 2]

    # Уберем слишком дорогие товары:
    data = data[data['price'] < 50]

    # Возьмём топ по популярности:
    popularity = data.groupby('item_id')['quantity'].sum().reset_index()
    popularity.rename(columns={'quantity': 'n_sold'}, inplace=True)

    top = popularity.sort_values('n_sold', ascending=False).head(take_n_popular).item_id.tolist()

    # Заведем фиктивный item_id (если юзер покупал товары из топ-5000, то он "купил" такой товар)
    data.loc[~data['item_id'].isin(top), 'item_id'] = 999999

    return data