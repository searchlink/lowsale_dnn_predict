# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 11:12
# @Author  : skydm
# @Email   : wzwei1636@163.com
# @File    : predict.py
# @Software: PyCharm

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf

desired_width = 320
pd.set_option('display.width', desired_width)   # 控制工作台显示区域的宽度
pd.set_option('display.max_columns', None)  # 控制工作太显示多少列
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 警告信息


def read_from_pickle(file):
    """
    读取pickle文件
    """
    f = open(file, "rb")
    df = pickle.load(f)
    return df


def areaid_dummy(df):
    """将区域id转化为哑变量"""
    areaid_list = [1, 7, 84, 5, 12, 6, 46, 51, 16]
    area_dict = {value: key for key, value in enumerate(areaid_list)}
    df["areaid"] = df["areaid"].map(area_dict)
    areaid_df = pd.get_dummies(df['areaid'], prefix='areaid')
    areaid_df = areaid_df.rename(lambda col: col.replace('.0', ''), axis='columns')

    # 拼接数据
    df = df.join(areaid_df)

    return df


def cate_dummy(df):
    """将类目转化为哑变量"""
    cate = ["Antiques", "Jewellery&Watches", "Art", "SportingGoods", "CarsMotorcycles&Vehicles",
            "Holidays&Travel", "PetSupplies", "MobilePhones&Communication", "EverythingElse",
            "BusinessOffice&Industrial",
            "VehicleParts&Accessories", "VideoGames&Consoles", "Collectables", "Crafts", "MusicalInstruments",
            "PotteryPorcelain&Glass", "Garden&Patio", "PackageMaterial", "Health&Beauty", "SportsMemorabilia",
            "Computers/Tablets&Networking", "Sound&Vision", "Toys&Games", "ClothesShoes&Accessories", "EventsTickets",
            "ConsumptiveMaterial", "HomeFurniture&DIY", "Coins", "Cameras&Photography", "Stamps",
            "eBayMotors", "Wholesale&JobLots", "Baby", "Dolls&Bears", "Music", "DVDsFilms&TV", "BooksComics&Magazines"]

    cate_dict = dict(zip(cate, range(len(cate))))
    df["categoryid"] = df["categoryid"].map(cate_dict)
    cate_df = pd.get_dummies(df['categoryid'], prefix='categoryid')
    cate_df = cate_df.rename(lambda col: col.replace('.0', ''), axis='columns')

    # 拼接数据
    df = df.join(cate_df)

    return df


def get_nonzero_week(x):
    """
    获取过去一年中第一个非零的周的位置
    """
    n = 52
    l = ['last_{}week'.format(i) for i in range(52, 0, -1)]
    for week in l:
        if x[week] > 0:
            break
        else:
            n -= 1
    return n


def get_nonzero_month(x):
    """
    获取过去一年中第一个非零的月的位置
    """
    n = 12
    l = ['last_{}month'.format(i) for i in range(12, 0, -1)]
    for month in l:
        if x[month] > 0:
            break
        else:
            n -= 1
    return n


def get_mean_week(x):
    """
    获取有效周的均值
    """
    l = ['last_{}week'.format(i) for i in range(52, 0, -1)]
    sum_ = 0
    for week in l[::-1][:int(x["num_week"])]:
        sum_ += x[week]

    try:
        mean_week = sum_ / int(x["num_week"])
    except Exception as e:
        mean_week = -99

    return mean_week


def get_mean_month(x):
    """
    获取有效月的均值
    """
    l = ['last_{}month'.format(i) for i in range(12, 0, -1)]
    sum_ = 0
    for month in l[::-1][:int(x["num_month"])]:
        sum_ += x[month]

    try:
        mean_month = sum_ / int(x["num_month"])
    except Exception as e:
        mean_month = -999

    return mean_month


def get_week_var(x):
    """
    获取有效周的方差
    """
    l = ['last_{}week'.format(i) for i in range(52, 0, -1)]
    save_cols = []
    for week in l[::-1][:int(x["num_week"])]:
        save_cols.append(x[week])

    return np.var(save_cols)


def get_month_var(x):
    """
    获取有效月的方差
    """
    l = ['last_{}month'.format(i) for i in range(12, 0, -1)]
    save_cols = []
    for month in l[::-1][:int(x["num_month"])]:
        save_cols.append(x[month])

    return np.var(save_cols)


def week_month_derive_feat(df):
    '''
    :return: 返回关于周或者月的衍生特征
    '''
    pro_week_cols = ['last_{}week'.format(i) for i in range(2, 53)]
    for_week_cols = ['last_{}week'.format(i) for i in range(1, 52)]

    # 周之间的差
    for index, _ in enumerate(for_week_cols):
        df["diff_week_" + str(index + 2) + "_" + str(index + 1)] = df[pro_week_cols[index]] - df[for_week_cols[index]]

    pro_month_cols = ['last_{}month'.format(i) for i in range(2, 13)]
    for_month_cols = ['last_{}month'.format(i) for i in range(1, 12)]

    # 月之间的差
    for index, _ in enumerate(for_month_cols):
        df["diff_month_" + str(index + 2) + "_" + str(index + 1)] = df[pro_month_cols[index]] - df[
            for_month_cols[index]]

    # 过去三个月的总销量
    df["last_3month_sum"] = df["last_1month"] + df["last_2month"] + df["last_3month"]

    return df


def data_process(df):
    # 获取周和月的衍生特征
    df = week_month_derive_feat(df)

    # 获取波动特征
    df["num_week"] = df.apply(lambda x: get_nonzero_week(x), axis=1)
    df["num_month"] = df.apply(lambda x: get_nonzero_month(x), axis=1)
    df["mean_week"] = df.apply(lambda x: get_mean_week(x), axis=1)
    df["mean_month"] = df.apply(lambda x: get_mean_month(x), axis=1)
    df["var_week"] = df.apply(lambda x: get_week_var(x), axis=1)
    df["var_month"] = df.apply(lambda x: get_month_var(x), axis=1)

    # 区域
    df = areaid_dummy(df)

    # 类目
    df = cate_dummy(df)

    X_cols = ["productid", "propertyid", "areaid", "last_1week", "last_2week", "last_3week", "last_4week", "last_5week",
              "last_6week", "last_7week", "last_8week", "last_9week", "last_10week", "last_11week", "last_12week",
              "last_13week", "last_14week", "last_15week", "last_16week", "last_17week", "last_18week", "last_19week",
              "last_20week", "last_21week", "last_22week", "last_23week", "last_24week", "last_25week", "last_26week",
              "last_27week", "last_28week", "last_29week", "last_30week", "last_31week", "last_32week", "last_33week",
              "last_34week", "last_35week", "last_36week", "last_37week", "last_38week", "last_39week", "last_40week",
              "last_41week", "last_42week", "last_43week", "last_44week", "last_45week", "last_46week", "last_47week",
              "last_48week", "last_49week", "last_50week", "last_51week", "last_52week", "last_1month", "last_2month",
              "last_3month", "last_4month", "last_5month", "last_6month", "last_7month", "last_8month", "last_9month",
              "last_10month", "last_11month", "last_12month", "pred_1month", "instock", "stock_div_last_1week",
              "stock_div_last_2weeks", "stock_div_last_1month", 'diff_week_2_1', 'diff_week_3_2', 'diff_week_4_3',
              'diff_week_5_4', 'diff_week_6_5', 'diff_week_7_6', 'diff_week_8_7', 'diff_week_9_8', 'diff_week_10_9',
              'diff_week_11_10', 'diff_week_12_11', 'diff_week_13_12', 'diff_week_14_13', 'diff_week_15_14',
              'diff_week_16_15', 'diff_week_17_16', 'diff_week_18_17', 'diff_week_19_18', 'diff_week_20_19',
              'diff_week_21_20',
              'diff_week_22_21', 'diff_week_23_22', 'diff_week_24_23', 'diff_week_25_24', 'diff_week_26_25',
              'diff_week_27_26',
              'diff_week_28_27', 'diff_week_29_28', 'diff_week_30_29', 'diff_week_31_30', 'diff_week_32_31',
              'diff_week_33_32',
              'diff_week_34_33', 'diff_week_35_34', 'diff_week_36_35', 'diff_week_37_36', 'diff_week_38_37',
              'diff_week_39_38',
              'diff_week_40_39', 'diff_week_41_40', 'diff_week_42_41', 'diff_week_43_42', 'diff_week_44_43',
              'diff_week_45_44',
              'diff_week_46_45', 'diff_week_47_46', 'diff_week_48_47', 'diff_week_49_48', 'diff_week_50_49',
              'diff_week_51_50',
              'diff_week_52_51', 'diff_month_2_1', 'diff_month_3_2', 'diff_month_4_3', 'diff_month_5_4',
              'diff_month_6_5',
              'diff_month_7_6', 'diff_month_8_7', 'diff_month_9_8', 'diff_month_10_9', 'diff_month_11_10',
              'diff_month_12_11',
              'last_3month_sum', 'num_week', 'num_month', 'mean_week', 'mean_month', 'var_week', 'var_month',
              'categoryid_0',
              'categoryid_1', 'categoryid_2', 'categoryid_3', 'categoryid_4', 'categoryid_6', 'categoryid_7',
              'categoryid_8',
              'categoryid_9', 'categoryid_10', 'categoryid_11', 'categoryid_12', 'categoryid_13', 'categoryid_14',
              'categoryid_15',
              'categoryid_16', 'categoryid_18', 'categoryid_19', 'categoryid_20', 'categoryid_21', 'categoryid_22',
              'categoryid_23',
              'categoryid_26', 'categoryid_27', 'categoryid_28', 'categoryid_29', 'categoryid_30', 'categoryid_31',
              'categoryid_32',
              'categoryid_33', 'categoryid_34', 'categoryid_35', 'categoryid_36', "lev", "costprice", "spring",
              "summer", "autumn",
              "winter", "iscloth", 'areaid_0', 'areaid_1', 'areaid_3', 'areaid_5', 'areaid_6', 'areaid_7', 'arriveday']
    Y_cols = ["future_1month"]

    df["costprice"] = df["costprice"].astype("float")

    # 排除伏羲的预测结果
    X_cols_X = [col for col in X_cols if col not in ["productid", "propertyid", "areaid", "pred_1month"]]

    df.fillna(-1, inplace=True)

    X = df[X_cols]
    Y = df[Y_cols]

    X_feature = X[X_cols_X]

    return X, Y, X_feature


def main():
    data_file = os.path.join(os.getcwd(), "pandas_df_0404.pickle")
    df = read_from_pickle(data_file)

    meta_path = './data/model.ckpt.meta'
    ckpt_path = "./data/model.ckpt"

    X, Y, X_feature = data_process(df)

    batch_size = 512

    with tf.Session() as sess:

        saver = tf.train.import_meta_graph(meta_path)
        saver.restore(sess, ckpt_path)

        graph = tf.get_default_graph()

        x = graph.get_operation_by_name('placeholders/input_x').outputs[0]
        y = graph.get_operation_by_name('placeholders/input_y').outputs[0]
        # x = graph.get_operation_by_name('placeholders/input_x:0')
        # y = graph.get_operation_by_name('placeholders/input_y:0')
        logits = tf.get_collection('dnn_predict')[0]

        # 进行预测
        output_predict = np.zeros((X_feature.shape[0], 1))
        index = (X_feature.shape[0] // batch_size) * batch_size
        for i in range(0, index, batch_size):
            X_ = X_feature.iloc[i:(i + batch_size), :].values
            prediction = sess.run([logits], feed_dict={x: X_})
            output_predict[i:(i + batch_size)] = prediction[0]

        if index != X_feature.shape[0]:
            prediction = sess.run([logits], feed_dict={x: X_feature.iloc[index:X_feature.shape[0], :].values
                               })
            output_predict[index:] = prediction[0]

        mae = mean_absolute_error(output_predict, Y.values)
        rmse = np.sqrt(mean_squared_error(output_predict, Y.values))

        print("在0404数据集上进行预测，mae：{mae}, rmse: {rmse}".format(mae=mae, rmse=rmse))


if __name__ == "__main__":
    main()
