import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from flask import Flask, request, jsonify, render_template
import model


app = Flask(__name__)


@app.route('/')
def home():
    model.getData()
    return render_template('HousePricePredictor.html')


@app.route('/predict', methods=['POST'])
def predict():
    # model.predict()
    feature_dict = model.predict()
    print(feature_dict)
    data = request.form.to_dict()
    # print(data)
    new_dict = {}
    # CreditHistory_dict = {"critical account": '1'}
    for k, v in data.items():
        if v.strip():
            new_dict[k] = v
    print(new_dict)
    test_data = pd.DataFrame(
        data=[[new_dict["MSSubClass"], new_dict["MSZoning"], new_dict["LotFrontage"], new_dict["LotArea"] ,new_dict["Street"], new_dict["LotShape"], new_dict["LandContour"], new_dict["Utilities"],new_dict["LotConfig"], new_dict["LandSlope"], new_dict["Neighborhood"], new_dict["Condition1"],new_dict["Condition2"], new_dict["BldgType"], new_dict["HouseStyle"], new_dict["OverallQual"],new_dict["OverallCond"], new_dict["YearBuilt"], new_dict["YearRemodAdd"], new_dict["RoofStyle"],new_dict["RoofMatl"], new_dict["Exterior1st"], new_dict["Exterior2nd"], new_dict["MasVnrType"], new_dict["MasVnrArea"], new_dict["ExterQual"], new_dict["ExterCond"], new_dict["Foundation"],
               new_dict["BsmtQual"], new_dict["BsmtCond"], new_dict["BsmtExposure"], new_dict["BsmtFinType1"],new_dict["BsmtFinSF1"], new_dict["BsmtFinType2"], new_dict["BsmtUnfSF"], new_dict["TotalBsmtSF"],
               new_dict["Heating"], new_dict["HeatingQC"], new_dict["CentralAir"], new_dict["Electrical"],new_dict["1stFlrSF"], new_dict["2ndFlrSF"], new_dict["GrLivArea"], new_dict["BsmtFullBath"],
               new_dict["FullBath"], new_dict["HalfBath"], new_dict["BedroomAbvGr"], new_dict["KitchenAbvGr"],new_dict["KitchenQual"], new_dict["TotRmsAbvGrd"], new_dict["Functional"], new_dict["Fireplaces"],
               new_dict["FireplaceQu"], new_dict["GarageType"], new_dict["GarageYrBlt"], new_dict["GarageFinish"],new_dict["GarageCars"], new_dict["GarageArea"], new_dict["GarageQual"], new_dict["GarageCond"],
               new_dict["PavedDrive"], new_dict["WoodDeckSF"], new_dict["OpenPorchSF"], new_dict["MoSold"],
               new_dict["YrSold"], new_dict["SaleType"], new_dict["SaleCondition"], feature_dict[new_dict["MSZoning"]],
               feature_dict[new_dict["LotFrontage"]], feature_dict[new_dict["Street"]],
               feature_dict[new_dict["LotShape"]], feature_dict[new_dict["LandContour"]],
               feature_dict[new_dict["Utilities"]], feature_dict[new_dict["LotConfig"]],
               feature_dict[new_dict["LandSlope"]], feature_dict[new_dict["Neighborhood"]],
               feature_dict[new_dict["Condition1"]], feature_dict[new_dict["Condition2"]],
               feature_dict[new_dict["BldgType"]], feature_dict[new_dict["HouseStyle"]],
               feature_dict[new_dict["RoofStyle"]], feature_dict[new_dict["RoofMatl"]],
               feature_dict[new_dict["Exterior1st"]], feature_dict[new_dict["Exterior2nd"]],
               feature_dict[new_dict["MasVnrType"]], feature_dict[new_dict["MasVnrArea"]],
               feature_dict[new_dict["ExterQual"]], feature_dict[new_dict["ExterCond"]],
               feature_dict[new_dict["Foundation"]], feature_dict[new_dict["BsmtQual"]],
               feature_dict[new_dict["BsmtCond"]], feature_dict[new_dict["BsmtExposure"]],
               feature_dict[new_dict["BsmtFinType1"]], feature_dict[new_dict["BsmtFinType2"]],
               feature_dict[new_dict["Heating"]], feature_dict[new_dict["HeatingQC"]],
               feature_dict[new_dict["CentralAir"]], feature_dict[new_dict["Electrical"]],
               feature_dict[new_dict["KitchenQual"]], feature_dict[new_dict["Functional"]],
               feature_dict[new_dict["FireplaceQu"]], feature_dict[new_dict["GarageType"]],
               feature_dict[new_dict["GarageYrBlt"]], feature_dict[new_dict["GarageFinish"]],
               feature_dict[new_dict["GarageQual"]], feature_dict[new_dict["GarageCond"]],
               feature_dict[new_dict["PavedDrive"]], feature_dict[new_dict["SaleType"]],
               feature_dict[new_dict["SaleCondition"]]]],
        columns=['MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'LotShape', 'LandContour', 'Utilities',
                 'LotConfig', 'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle',
                 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                 'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinSF1', 'BsmtFinType2', 'BsmtUnfSF', 'TotalBsmtSF',
                 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                 'BsmtFullBath', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'KitchenQual', 'TotRmsAbvGrd',
                 'Functional', 'Fireplaces', 'FireplaceQu', 'GarageType', 'GarageYrBlt', 'GarageFinish', 'GarageCars',
                 'GarageArea', 'GarageQual', 'GarageCond', 'PavedDrive', 'WoodDeckSF', 'OpenPorchSF', 'MoSold',
                 'YrSold', 'SaleType', 'SaleCondition', 'MSZoning_index', 'LotFrontage_index', 'Street_index',
                 'LotShape_index', 'LandContour_index', 'Utilities_index', 'LotConfig_index', 'LandSlope_index',
                 'Neighborhood_index', 'Condition1_index', 'Condition2_index', 'BldgType_index', 'HouseStyle_index',
                 'RoofStyle_index', 'RoofMatl_index', 'Exterior1st_index', 'Exterior2nd_index', 'MasVnrType_index',
                 'MasVnrArea_index', 'ExterQual_index', 'ExterCond_index', 'Foundation_index', 'BsmtQual_index',
                 'BsmtCond_index', 'BsmtExposure_index', 'BsmtFinType1_index', 'BsmtFinType2_index', 'Heating_index',
                 'HeatingQC_index', 'CentralAir_index', 'Electrical_index', 'KitchenQual_index', 'Functional_index',
                 'FireplaceQu_index', 'GarageType_index', 'GarageYrBlt_index', 'GarageFinish_index', 'GarageQual_index',
                 'GarageCond_index', 'PavedDrive_index', 'SaleType_index', 'SaleCondition_index'])
    print(test_data)
    return "OK"


if __name__ == "__main__":
    app.run('0.0.0.0', '9000', debug=True)
