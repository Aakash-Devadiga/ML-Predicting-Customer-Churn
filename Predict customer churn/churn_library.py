"""
Python script to find out the customer churn
"""


# import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import pylint

def import_data(file_path):  
    '''
    returns dataframe for the csv found at path

    input:
            path: a path to the csv
    output:
            df : pandas dataframe
    '''	 
    df = pd.read_csv(file_path)
    df["Churn"] = df.Attrition_Flag.apply(lambda val: 0 if val == "Existing Customer" else 1)
    return df

def perform_eda(df,eda_output_path):   
    '''
    perform exploratory data analysis on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    column_names = ["Churn", "Customer_Age", "Marital_Status", "Total_Trans"]    
    for column_name in column_names:
        plt.figure(figsize=(20, 10))
        if column_name == "Churn":
            plt.title("Churn")
            df.Churn.hist()
        elif column_name == "Customer_Age":
            plt.title("customer age")
            df.Customer_Age.hist()
        elif column_name == "Marital_Status":
            plt.title("Marital Staus")
            df.Marital_Status.value_counts("normalize").plot(kind="bar")
        elif column_name == "Total_Trans":
            plt.title("Total_Trans_count")
            sns.displot(df.Total_Trans_Ct)
        plt.savefig(os.path.join(eda_output_path, str(column_name)+".jpg"))
        plt.close()
    plt.title("Heat Map")
    sns.heatmap(df.corr(), annot=False, cmap="Dark2_r", linewidths=2)
    plt.savefig(os.path.join(eda_output_path, "Heat_map.jpg"))
    plt.close()
        
def encoder_helper(df, category_list):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            df: pandas dataframe with new columns for categorical columns
    '''
    for category in category_list:
        category_list = []
        category_groups = df.groupby(category).mean()["Churn"]
        
        for val in df[category]:
            category_list.append(category_groups.loc[val])
            
        df["%s_%s" % (category, "Churn")] = category_list  
    return df

def perform_feature_engineering(df):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    
    target = df["Churn"] 
    feature_columns = [
        "Customer_Age",
        "Dependent_count",
        "Months_on_book",
        "Total_Relationship_Count",
        "Months_Inactive_12_mon",
        "Contacts_Count_12_mon",
        "Credit_Limit",
        "Total_Revolving_Bal",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Trans_Ct",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
        "Gender_Churn",
        "Education_Level_Churn",
        "Marital_Status_Churn",
        "Income_Category_Churn",
        "Card_Category_Churn"]
    
    features = df[feature_columns]

    # train test split
    x_train, x_test, y_train, y_test = train_test_split(features,
                                                        target,
                                                        test_size=0.3,
                                                        random_state=42)

    return x_train, x_test, y_train, y_test

def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    ''' 
    classification_report_dict = {'Random Forest': ('Random Forest Train',
                                                   y_train,
                                                   y_train_preds_rf,
                                                   'Random Forest Test',
                                                   y_test,
                                                   y_test_preds_rf ),
                                  
                                  'Logistic_Regression': ("Logistic Regression Train",
                                                            y_train,
                                                            y_train_preds_lr,
                                                            "Logistic Regression Test",
                                                            y_test,
                                                            y_test_preds_lr)}
   
    
    for model_name, y_data  in classification_report_dict.items():
        
        plt.rc("figure", figsize=(8, 8))
        plt.text(0.01, 1.25, str(y_data[0]), 
                 { "fontsize": 10}, 
                 fontproperties="monospace")
        plt.text( 0.01, 0.05,
                         str(classification_report(y_data[1], y_data[2])), 
                         {"fontsize": 10}, 
                         fontproperties="monospace")
        plt.text(0.01, 0.6, str(y_data[3]), 
                 {"fontsize": 10}, 
                 fontproperties="monospace")
        plt.text(0.01, 0.7,
                 str(classification_report(y_data[4],y_data[5])),
                 {"fontsize": 10},
                 fontproperties="monospace")
                        
        plt.axis("off")
        plt.savefig(r"Images\Results\%s.jpg" % model_name)
        plt.close()

def feature_importance_plot(model, x_data, output_path):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    '''
    importances = model.best_estimator_.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [x_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(x_data.shape[1]), importances[indices])
    plt.xticks(range(x_data.shape[1]), names, rotation=90)
    plt.savefig("Images/Results/%s.jpg" % output_path)
    plt.close()

def train_models(x_train, x_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(max_iter=1000)

    param_grid = {
        "n_estimators": [200, 500],
        "max_features": ["auto", "sqrt"],
        "max_depth": [4, 5, 100],
        "criterion": ["gini", "entropy"]
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(x_train, y_train)
    
    lrc.fit(x_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    feature_importance_plot(cv_rfc, x_test, "feature_importance")

    joblib.dump(cv_rfc.best_estimator_, r"models\rfc_model.pkl")
    joblib.dump(lrc, "models/logistic_model.pkl")

if __name__ == "__main__":
    
    os.environ['QT_QPA_PLATFORM']='offscreen'
    dataset_path = r"data\bank_data.csv"
    image_output_path = r"Images\eda"
    df = import_data(dataset_path)
    perform_eda(df, image_output_path)
    encoder_category_list = ["Gender",
                              "Education_Level",
                              "Marital_Status",
                              "Income_Category",
                              "Card_Category"]
    
    encoded_data_df = encoder_helper(df, encoder_category_list)
    
    x_train_, x_test_, y_train_, y_test_ = perform_feature_engineering(encoded_data_df)
    train_models(x_train_, x_test_, y_train_, y_test_)
    