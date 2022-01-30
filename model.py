import pandas as pd
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext

from pyspark.sql.types import *
import pyspark.sql.functions as F
from pyspark.sql.functions import udf, col

from pyspark.ml.regression import LinearRegression
from pyspark.mllib.evaluation import RegressionMetrics

from pyspark.ml.tuning import ParamGridBuilder, CrossValidator, CrossValidatorModel
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors

import os
import seaborn as sns
import numpy as np
import matplotlib as plt

# import joblib

spark_session = SparkSession.builder.master("local[2]").appName("HousingRegression").getOrCreate()
spark_context = spark_session.sparkContext
spark_sql_context = SQLContext(spark_context)


def getData():
    # for dirname, _, filenames in os.walk('/data'):
    #     for filename in filenames:
    #         print(os.path.join(dirname, filename))
    # train_df = spark_session.read.option('header', 'true').csv(os.path.join('/data', 'train.csv'), inferSchema=True)
    # test_df = spark_session.read.option('header', 'true').csv(os.path.join('/data', 'test.csv'), inferSchema=True)
    # train_df = spark_session.read.csv("train.csv")
    # #train_df.rdd.saveAsPickleFile("train_df.pkl")
    # train_df.rdd.saveAsPickleFile("/")
    train_df = pd.read_csv("train.csv")


# Read the datasets


def predict():
    train_df = spark_session.read.option('header', 'true').csv(os.path.join('', 'train.csv'), inferSchema=True)

    # pickleRdd = sc.pickleFile(filename).collect()
    # df2 = spark.createDataFrame(pickleRdd)
    # train_df = joblib.load('train_df.sav')
    # return train_df
    train_df.show(5)
    # identifying the columns having less meaningful data on the basis of datatypes
    l_int = []
    for item in train_df.dtypes:
        if item[1] == 'int':
            l_int.append(item[0])
    print("Int", l_int)

    l_str = []
    for item in train_df.dtypes:
        if item[1] == 'string':
            l_str.append(item[0])
    print("String", l_str)

    # This is how fillna is done in PySpark

    from pyspark.sql.functions import col, sum

    # train_df.select(*(sum(col(c).isNull().cast("int")).alias(c) for c in train_df.columns)).show()
    #
    # for i in train_df.columns:
    #     if i in l_int:
    #         a = 'train_df' + '.' + i
    #         ct_total = train_df.select(i).count()
    #         ct_zeros = train_df.filter((col(i) == 0)).count()
    #     per_zeros = (ct_zeros / ct_total) * 100
    #     print('total count / zeros count '
    #           + i + ' ' + str(ct_total) + ' / ' + str(ct_zeros) + ' / ' + str(per_zeros))

    # above calculation gives us an insight about the useful features
    # now drop the columns having zeros or NA % more than 75 %

    df_new = train_df.drop(*['BsmtFinSF2', 'LowQualFinSF', 'BsmtHalfBath',
                             'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
                             'PoolArea', 'PoolQC', 'Fence', 'MiscFeature',
                             'MiscVal', 'Alley'])
    df_new = df_new.drop(*['Id'])
    # now we have the clean data to work

    # Housing prices greater than 500,000 (expensive houses)
    # print("No of houses: %i" % train_df.select('SalePrice').count())
    # print("No of houses greater than $500000 are: %i" % train_df.filter(train_df["SalePrice"] > 500000).count())

    # # Distribution of prices
    # sns.set_style("darkgrid")
    # sns.histplot(train_df.select('SalePrice').toPandas(), bins=10)

    # converting string to numeric feature

    feat_list = ['MSZoning', 'LotFrontage', 'Street', 'LotShape', 'LandContour',
                 'Utilities', 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1',
                 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation',
                 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
                 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual',
                 'Functional', 'FireplaceQu', 'GarageType',
                 'GarageYrBlt', 'GarageFinish', 'GarageQual', 'GarageCond',
                 'PavedDrive', 'SaleType', 'SaleCondition']
    print('indexed list created')

    # there are multiple features to work
    # using pipeline we can convert multiple features to indexers
    indexers = [StringIndexer(inputCol=column, outputCol=column + "_index").fit(df_new) for column in feat_list]
    type(indexers)
    # Combines a given list of columns into a single vector column.
    # input_cols: Columns to be assembled.
    # returns Dataframe with assembled column.

    pipeline = Pipeline(stages=indexers)
    df_feat = pipeline.fit(df_new).transform(df_new)
    # print(df_feat.show(10))

    # using above code we have converted list of features into indexes

    # we will convert below columns into features to work with
    assembler = VectorAssembler(inputCols=['MSSubClass', 'LotArea', 'OverallQual',
                                           'OverallCond', 'YearBuilt', 'YearRemodAdd',
                                           'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF',
                                           '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                                           'BsmtFullBath', 'FullBath', 'HalfBath',
                                           'GarageArea', 'MoSold', 'YrSold',
                                           'MSZoning_index', 'LotFrontage_index',
                                           'Street_index', 'LotShape_index',
                                           'LandContour_index', 'Utilities_index',
                                           'LotConfig_index', 'LandSlope_index',
                                           'Neighborhood_index', 'Condition1_index',
                                           'Condition2_index', 'BldgType_index',
                                           'HouseStyle_index', 'RoofStyle_index',
                                           'RoofMatl_index', 'Exterior1st_index',
                                           'Exterior2nd_index', 'MasVnrType_index',
                                           'MasVnrArea_index', 'ExterQual_index',
                                           'ExterCond_index', 'Foundation_index',
                                           'BsmtQual_index', 'BsmtCond_index',
                                           'BsmtExposure_index', 'BsmtFinType1_index',
                                           'BsmtFinType2_index', 'Heating_index',
                                           'HeatingQC_index', 'CentralAir_index',
                                           'Electrical_index', 'KitchenQual_index',
                                           'Functional_index', 'FireplaceQu_index',
                                           'GarageType_index', 'GarageYrBlt_index',
                                           'GarageFinish_index', 'GarageQual_index',
                                           'GarageCond_index', 'PavedDrive_index',
                                           'SaleType_index', 'SaleCondition_index'],
                                outputCol='features')
    output = assembler.transform(df_feat)

    feature_data_dict = {row['MSZoning']: row['MSZoning_index'] for row in df_feat.collect()}
    feature_data_dict.update({row['LotFrontage']: row['LotFrontage_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Street']: row['Street_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['LotShape']: row['LotShape_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['LandContour']: row['LandContour_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Utilities']: row['Utilities_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['LotConfig']: row['LotConfig_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['LandSlope']: row['LandSlope_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Neighborhood']: row['Neighborhood_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Condition1']: row['Condition1_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Condition2']: row['Condition2_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['BldgType']: row['BldgType_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['HouseStyle']: row['HouseStyle_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['RoofStyle']: row['RoofStyle_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['RoofMatl']: row['RoofMatl_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Exterior1st']: row['Exterior1st_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Exterior2nd']: row['Exterior2nd_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['MasVnrType']: row['MasVnrType_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['MasVnrArea']: row['MasVnrArea_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['ExterQual']: row['ExterQual_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['ExterCond']: row['ExterCond_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Foundation']: row['Foundation_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['BsmtQual']: row['BsmtQual_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['BsmtCond']: row['BsmtCond_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['BsmtExposure']: row['BsmtExposure_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['BsmtFinType1']: row['BsmtFinType1_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['BsmtFinType2']: row['BsmtFinType2_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Heating']: row['Heating_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['HeatingQC']: row['HeatingQC_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['CentralAir']: row['CentralAir_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Electrical']: row['Electrical_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['KitchenQual']: row['KitchenQual_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['Functional']: row['Functional_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['FireplaceQu']: row['FireplaceQu_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['GarageType']: row['GarageType_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['GarageYrBlt']: row['GarageYrBlt_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['GarageFinish']: row['GarageFinish_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['GarageQual']: row['GarageQual_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['GarageCond']: row['GarageCond_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['PavedDrive']: row['PavedDrive_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['SaleType']: row['SaleType_index'] for row in df_feat.collect()})
    feature_data_dict.update({row['SaleCondition']: row['SaleCondition_index'] for row in df_feat.collect()})
    return feature_data_dict
    final_data = output.select('features', 'SalePrice')

    # splitting data for test and validation
    train_data, test_data = final_data.randomSplit([0.7, 0.3])

    house_lr = LinearRegression(featuresCol='features', labelCol='SalePrice')
    trained_house_model = house_lr.fit(train_data)
    house_results = trained_house_model.evaluate(train_data)
    print('Rsquared Error :', house_results.r2)

    # Rsquared Error : 0.8279155904297449
    # model accuracy is 82 % with train data

    # evaluate model on test_data
    test_results = trained_house_model.evaluate(test_data)
    print('Rsquared error :', test_results.r2)

    # Rsquared error : 0.8431420382408793
    # result is quiet better with 84 % accuracy

    # create unlabelled data from test_data
    # test_data.show()
    unlabeled_data = test_data.select('features')
    unlabeled_data.show()

    # predictions = trained_house_model.transform(unlabeled_data)
    # predictions.show()
