"""
Testing module written to check the churn_library.py.
"""

import os
import logging
import pytest
import churn_library as cls

os.environ['QT_QPA_PLATFORM']='offscreen'
logging.basicConfig(
    filename='./logs/churn_library.log',
    level = logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

def test_import(import_data):
	'''
	test data import - this example is completed for you to assist with the other test functions
	'''
	try:
		df = import_data("./data/bank_data.csv")
		logging.info("Testing import_data: SUCCESS")
	except FileNotFoundError as err:
		logging.error("Testing import_eda: The file wasn't found")
		raise err

	try:
		assert df.shape[0] > 0
		assert df.shape[1] > 0
	except AssertionError as err:
		logging.error("Testing import_data: The file doesn't appear to have rows and columns")
		raise err


def test_eda(perform_eda):  
    '''
   	test perform eda function
    testing by checking the image directories where the output from the function is saved
   	''' 
    try:
        assert os.path.isfile('Images/eda/Churn.jpg')
        assert os.path.isfile('Images/eda/Customer_Age.jpg')
        assert os.path.isfile('Images/eda/Marital_Status.jpg')
        assert os.path.isfile('Images/eda/Total_Trans.jpg')
        assert os.path.isfile('Images/eda/Heat_map.jpg')
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.error("Testing import_data: The file wasn't found")
        raise err

def test_encoder_helper(encoder_helper):    
    '''
	test encoder helper
	'''
    try:
        encoder_helper(df,encoder_category_list)
        logging.info("Testing test_encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: function did not execute succesfully")
        raise err

    try:
        df_new = encoder_helper(df, encoder_category_list)
        for i in df_new.columns:
            assert isinstance(i,str)
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The encoder did not add new columns")
        raise err

def test_perform_feature_engineering(perform_feature_engineering):    
    '''
	test perform_feature_engineering
	'''
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(df_encoded)
        assert X_train.shape[0] > X_test.shape[0]
        assert y_train.shape[0] > y_test.shape[0]
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Data did not split as per requirement")
        raise err

def test_train_models(train_models):
    '''
	test train_models
	'''
    try:
        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df_encoded)
        train_models(X_train, X_test, y_train, y_test)
        assert os.path.isfile('./models/rfc_model.pkl')
        assert os.path.isfile('./models/logistic_model.pkl')
        assert os.path.isfile('Images/Results/Logistic_Regression.jpg')
        assert os.path.isfile('Images/Results/Random Forest.jpg')
        assert os.path.isfile('./Images/Results/feature_importance.jpg')
        logging.info("Testing test_train_models: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing test_train_models: model results not found")
        raise err	
  
df = cls.import_data("./data/bank_data.csv")
encoder_category_list = ["Gender",
                          "Education_Level",
                          "Marital_Status",
                          "Income_Category",
                          "Card_Category"]
test_import(cls.import_data)
test_eda(cls.perform_eda)
df_encoded = cls.encoder_helper(df, encoder_category_list)
test_encoder_helper(cls.encoder_helper)
test_perform_feature_engineering(cls.perform_feature_engineering)
test_train_models(cls.train_models)
