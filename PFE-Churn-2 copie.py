# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 17:00:59 2021

@author: Matt
"""


###  INTRODUCTION : LECTURE DES DONNÉES  ###


# Importation des librairies
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, roc_curve, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier, AdaBoostClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE, ADASYN
from xgboost import XGBClassifier
import statsmodels.api as sm
from mlxtend.evaluate import lift_score
from sklearn.metrics import make_scorer
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.preprocessing import LabelEncoder
RANDOM_STATE = 42





    
    
    
def dataframe_reading():
    """
    Lecture des données et mise sous forme d'une dataframe
    
    Returns
    -------
    dataframe : dataframe
        Dataframe contenant les données du fichier telecom_churn_data.

    """
    #Charger et afficher la DataFrame
    dataframe = pd.read_csv("telecom_churn_data.csv", header = 0, sep=",", encoding = "ISO-8859-1", low_memory=False)
    #indices = pd.Series(dataframe.index, index=dataframe['mobile_number'])
    #print(dataframe.head())
    #print(dataframe.shape)
    #print(indices.head())
    return dataframe

df = dataframe_reading()

def missing_values_columns(df):
    """
    Affiche le nombre de colonnes de la dataframe et combien de colonnes possèdent des valeurs manquantes

    Parameters
    ----------
    df : dataframe
        Dataframe contenant les données du fichier csv telecom_churn_data.csv

    Returns
    -------
    mis_val_table : dataframe
        Une Dataframe contenant le nom des colonnes de df ainsi que le pourcentage
        de valeurs manquantes pour chaque colonne

    """
    mis_val = df.isnull().sum() 
    mis_val_percent = 100 * df.isnull().sum()/len(df) 
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) 
    mis_val_table = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'}) 
    mis_val_table_sorted = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1) 
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"  
      "There are " + str(mis_val_table_sorted.shape[0]) + 
       " columns that have missing values.") 
    return mis_val_table

def missing_values_rows(df):
    """
    Affiche le nombre de lignes de la dataframe et combien de lignes possèdent des valeurs manquantes

    Parameters
    ----------
    df : dataframe
        Dataframe contenant les données du fichier csv telecom_churn_data.csv

    Returns
    -------
    mis_val_table : dataframe
        Dataframe contenant les lignes de df ainsi que le pourcentage 
        de valeurs manquantes pour chaque ligne

    """
    df = df.transpose()
    mis_val = df.isnull().sum() 
    mis_val_percent = 100 * df.isnull().sum()/len(df) 
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1) 
    mis_val_table = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'}) 
    mis_val_table_sorted = mis_val_table[mis_val_table.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1) 
    print ("Your selected dataframe has " + str(df.shape[1]) + " rows.\n"  
     "There are " + str(mis_val_table_sorted.shape[0]) + 
      " rows that have missing values.") 
    return mis_val_table

vm_colonnes = missing_values_columns(df)
vm_lignes = missing_values_rows(df)

def double_detection(df):
    """
    Détecte et supprime les doublons dans la dataframe df

    Parameters
    ----------
    df : dataframe
        Dataframe contenant les données du fichier csv telecom_churn_data.csv

    Returns
    -------
    churn : dataframe
        df prélevée de toutes ses données en doubles (inchangée)

    """
    churn = df.copy()
    churn.drop_duplicates(keep = 'first', inplace=True)
    return churn       

def nb_uni_val_per_col(df):
    """
    Affiche une dataframe affichant le nombre de valeurs uniques par colonne
    
    Parameters
    ----------
    df : dataframe
        Dataframe contenant les données du fichier csv telecom_churn_data.csv

    Returns
    -------
    None.
    
    """
    print("\nUnique values:\n",df.nunique())

def uni_val_per_col(df):
    """
    Affiche le nom et toutes les valeurs uniques de chaque colonne de df
    
    Parameters
    ----------
    df : dataframe
        Dataframe contenant les données du fichier csv telecom_churn_data.csv

    Returns
    -------
    None.

    """
    print('\n\n\nAffichage des noms de colonnes et leurs valeurs uniques propres :\n\n')
    for columns in df:
        print(columns)
        print(df[columns].unique())


def col_name(df):
    """
    Affiche le nom de toutes les colonnes (et le nombre de colonnes) de df

    Parameters
    ----------
    df : dataframe
        Dataframe contenant les données du fichier csv telecom_churn_data.csv

    Returns
    -------
    None.

    """
    print('\n\n\nAffichage du nombre de colonnes et les noms de chacune d\'entre elles :\n\n')
    print('\nVoici les noms de toutes les colonnes :\n')
    print(list(df))
    print('\nNombre total de colonnes :', len(list(df)))

        
def uni_val_col_search(df): 
    """
    Recherche une colonne rentrée par l'utilisateur pour afficher son nombre de valeurs uniques dans df

    Parameters
    ----------
    df : dataframe
        Dataframe contenant les données du fichier csv telecom_churn_data.csv

    Returns
    -------
    None.

    """
    print('\n\n\nRecherche des valeurs uniques d\'une colonne spécifique :\n\n')
    print('\nNom de la colonne recherchée :')
    Column_name = input()
    if Column_name in df:
        print('\n\nValeur(s) unique(s) de la colonne recherchée :')
        uni_val_col = list(df[Column_name].unique())
        print(uni_val_col)
        print('\n\nNombre de valeur(s) unique(s) de la colonne recherchée :')
        print(len(uni_val_col))
    else :
        print('\n\nAucune colonne de nom ' + Column_name + ' n\'existe\n\n')
        uni_val_col_search(df)
        
def churn_rate_tot(df):
    """
    Affecte à une nouvelle colonne "churn" de la dataframe la valeur 1 ou 0 pour les clients 
    selon s'ils ont churné

    Parameters
    ----------
    df : dataframe
        Dataframe contenant les données du fichier csv telecom_churn_data.csv

    Returns
    -------
    churn_rate : float
        Churn rate sur la totalité de l'échantillon sous forme de pourcentage
    churn_tot_graph : graph
        Graphique montrant la proportion de churn au sein de l'échantillon

    """
    churn_col = ['total_ic_mou_9', 'total_og_mou_9', 'vol_2g_mb_9', 'vol_3g_mb_9']
    df['churn'] = np.where(df[churn_col].sum(axis=1) == 0, 1, 0 )
    churn_rate = round(((sum(df['churn'])/len(df['churn']))*100),2)
    sub_df_1 = df[df['churn'] == 0]
    sub_df_2 = df[df['churn'] == 1]
    print('Churn Rate : {0}%'.format(churn_rate))
    print('\nNombre de client non churnés : ', len(sub_df_1))
    print('\nNombre de client churnés : ', len(sub_df_2))
    churn_tot_graph = sns.countplot(x='churn', data=df)
    return churn_rate, churn_tot_graph

churn_rate_total, churn_tot_graph = churn_rate_tot(df)
data = df.copy()

def imputeNan(data,missingColList=False):
    """
    Remplace les valeurs Nan par 0 sur la dataframe data afin de pouvoir de réaliser 
    des opérations sur certaines colonnes

    Parameters
    ----------
    data : dataframe
        Dataframe étant une copie de df, dataframe contenant les valeurs 
        du fichier csv telecom_churn_data.csv
        
    missingColList : list
        Liste contenant les colonnes ou des valeurs manquantes sont encore présentes 
        The default is False

    Returns
    -------
    data : dataframe
        Dataframe data modifiée où les valeurs manquantes sont remplacées par 0

    """   
    for col in missingColList:
        data[col].fillna(0, inplace=True)
    return data

def new_data_col(data):
    """
    Création d'une nouvelle colonne qui servira au calcul du client rentable par la suite

    Parameters
    ----------
    data : dataframe
        Dataframe data modifiée où les valeurs manquantes sont remplacées par 0

    Returns
    -------
    data['total_rech_amt_'+str(i)] : data
        Dataframe contenant la nouvelle colonne que l'on a rajouter à data

    """
    for i in range(6,8):
        data['total_rech_amt_'+str(i)] = data['total_rech_amt_'+str(i)]+(data['total_rech_data_'+str(i)]*data['av_rech_amt_data_'+str(i)])
    return data['total_rech_amt_'+str(i)]
   
data_col_6_7 = new_data_col(data)

def imputeNan_new_col(data):
    """
    Remplace les NaN par des 0 dans la nouvelle colonne de la dataframe data

    Parameters
    ----------
    data : dataframe
        Dataframe dont tous les NaN sont remplacés par des 0 sauf la nouvelle colonne data_col_6_7

    Returns
    -------
    data : dataframe
        Dataframe copie de df, dont tous les NaN sont remplacés par des 0

    """
    data = imputeNan(data,missingColList=['total_rech_amt_6','total_rech_amt_7'])
    return data

data = imputeNan_new_col(data)

def CR_def(data):
    """
    Determine le nombre de clients rentables dans le dataset et supprime les clients non rentables
    du dataset

    Parameters
    ----------
    data : dataframe
        Dataframe copie de df, dont tous les NaN sont remplacés par des 0

    Returns
    -------
    data_CR_def : dataframe
        Dataframe ne contenant plus que les clients rentables

    """
    # Get the average recharge amount for 6 and 7 month
    data['avg_rech_amt_6_7'] = ( data['total_rech_amt_6'] + data['total_rech_amt_7'] ) / 2
    # Get the data greater than 70th percentile of average recharge amount
    data_CR_def = data.loc[(data['avg_rech_amt_6_7'] > np.percentile(data['avg_rech_amt_6_7'], 70))]
    #print(data.info())
    #print(data.shape)
    print("Number of High-Value Customers in the Dataset: %d\n"% data_CR_def.shape[0])
    #data.drop(data['avg_rech_amt_6_7'], axis=1, inplace=True)
    #data.drop(data['total_rech_data_6'], axis=1, inplace=True)
    #data.drop(data['total_rech_data_7'], axis=1, inplace=True)
    #data.drop(data['av_rech_amt_data_6'], axis=1, inplace=True)
    #data.drop(data['av_rech_amt_data_7'], axis=1, inplace=True)
    return data_CR_def

data_CR = CR_def(data)

def churn_rate_CR(data_CR):
    """
    Determine le nombre de clients rentables churnés et le pourcentage, le nombre de clients
    rentables non churnés et le graphe de proportion de population churnée

    Parameters
    ----------
    data_CR : dataframe
        Dataframe copie de df, dont tous les NaN sont remplacés par des 0

    Returns
    -------
     data_CR : dataframe
        Dataframe copie de data mais ne contenant plus que les clients rentables
        
    churn_rate_cr : float
        Churn rate sur les clients rentables de l'échantillon sous forme de pourcentage
        
    graph_churn_CR : graph 
        Graphique de la proportion de churn au sein de l'échantillon de clients rentables

    """
    pd.options.mode.chained_assignment = None  # default='warn'
    # Même technique que précédemment, mais sur la copie 'data_CR' de la dataframe 'df', ou ne sont plus répertoriés que les clients rentables
    churn_col_data = ['total_ic_mou_9', 'total_og_mou_9', 'vol_2g_mb_9', 'vol_3g_mb_9']
    data_CR['churn'] = np.where(data_CR[churn_col_data].sum(axis=1) == 0, 1, 0 )
    sub_data_1 = data_CR[data_CR['churn'] == 0]
    sub_data_2 = data_CR[data_CR['churn'] == 1]
    print('\n\nNombre de client non churnés : ', len(sub_data_1))
    print('Nombre de client churnés : ', len(sub_data_2))
    churn_rate_cr = round(((sum(data_CR['churn'])/len(data_CR['churn']))*100),2)
    print('Churn Rate CR : {0}%'.format(churn_rate_cr))
    graph_churn_CR = sns.countplot(x='churn', data=data_CR)
    return data_CR, churn_rate_cr, graph_churn_CR

data_CR, churn_rate_cr, graph_churn_cr = churn_rate_CR(data_CR)

def rows_and_col_supp(data_CR):
    """
    Supprime les lignes et colonnes ayant un pourcentage de valeurs manquantes supérieur à nos
    critères

    Parameters
    ----------
    data_CR : dataframe
        Dataframe copie de df, dont tous les NaN sont remplacés par des 0,
        ne contenant plus que les clients rentables

    Returns
    -------
    df_without_missing_values : dataframe
        Dataframe copie de df, dont tous les NaN sont remplacés par des 0, ne contenant plus
        que les clients rentables, nettoyée certaines de lignes et colonnes

    """
    df_without_missing_values = data_CR.copy()
    seuil_ligne = 50
    df_without_missing_values = df_without_missing_values.reset_index(drop=True)
    vm_lignes = missing_values_rows(df_without_missing_values)
    print(vm_lignes)
    perc_miss_val_row_list = vm_lignes["% of Total Values"].tolist()
    for i in range(0,len(perc_miss_val_row_list)):
        if perc_miss_val_row_list[i] > seuil_ligne:
            df_without_missing_values.drop(i,0,inplace=True)
    print(df_without_missing_values)
    

    """
    seuil_colonne = 30
    df_without_missing_values = df_without_missing_values.reset_index(drop=True)
    df_without_missing_values = df_without_missing_values.transpose()
    vm_colonnes = missing_values_rows(df_without_missing_values)
    
    print(vm_colonnes)
    perc_miss_val_row_list = vm_colonnes["% of Total Values"].tolist()
    for i in range(0,len(perc_miss_val_row_list)):
        if perc_miss_val_row_list[i] > seuil_colonne:
            df_without_missing_values.drop(i,0,inplace=True)
    print(df_without_missing_values)

    print(df_without_missing_values)
    """


    df_without_missing_values = df_without_missing_values.fillna(df_without_missing_values.mean())
    return df_without_missing_values

data_CR_cleaned = rows_and_col_supp(data_CR)

def data_cleaning(data_CR_cleaned):
    """
    Supprime les colonnes liées au mois de Septembre, les colonnes contenant des dates, et les colonnes 
    ayant 2 ou moins de 2 valeurs uniques

    Parameters
    ----------
    data_CR_cleaned : dataframe
        Dataframe copie de df, dont tous les NaN sont remplacés par des 0, ne contenant plus
        que les clients rentables, nettoyée certaines de lignes et colonnes

    Returns
    -------
    new_df : dataframe
        Dataframe copie de df, dont tous les NaN sont remplacés par des 0, ne contenant plus
        que les clients rentables, nettoyée de plus de lignes et colonnes

    """
    # Supprime toutes les colonnes contenant les chaînes de caratères '_9' et 'sep' et les dates 
    new_df = data_CR_cleaned[data_CR_cleaned.columns.drop(list(data_CR_cleaned.filter(regex='_9')))
                                       .drop(list(data_CR_cleaned.filter(regex='sep')))
                                       .drop(["last_date_of_month_6","last_date_of_month_7","last_date_of_month_8",
                                              "date_of_last_rech_6","date_of_last_rech_7","date_of_last_rech_8",
                                              "date_of_last_rech_data_6","date_of_last_rech_data_7","date_of_last_rech_data_8",
                                              "circle_id", "last_date_of_month_6", "last_date_of_month_7", "last_date_of_month_8",
                                              "loc_og_t2o_mou", "std_og_t2o_mou", "loc_ic_t2o_mou", "std_og_t2c_mou_6", 
                                              "std_og_t2c_mou_7", "std_og_t2c_mou_8", "std_ic_t2o_mou_6", "std_ic_t2o_mou_7", 
                                              "std_ic_t2o_mou_8"])
                                       .drop(data_CR_cleaned.columns[[145]])]
    return new_df

Final_Data = data_cleaning(data_CR_cleaned)

Final_Data = Final_Data.reset_index(drop=True)
        
def graph_churn_col(Final_Data, churn_rate_cr) : 
    """
    Retourne un graphe montrant les valeurs discriminantes d'une colonne sélectionnée par l'utilisateur

    Parameters
    ----------
    Final_Data : dataframe
        Dataframe nettoyée et préparée pour la modélisation
    churn_rate_cr : float
        Churn rate sur les clients rentables de l'échantillon sous forme de pourcentage

    Returns
    -------
    graph_churn : graph
        Graphique des valeurs discriminantes d'une colonne sélectionnée par l'utilisateur
        au sein de l'échantillon de clients rentables

    """
    print('\n\n\nRecherche des valeurs discriminantes d\'une colonne :')
    print('\nNom de la colonne recherchée :')
    Column_name = input()
    g = sns.catplot(x=Column_name, col="churn", data=Final_Data, kind="count", height=4, aspect=.7);
    graph_churn = g.map(plt.axhline, y = (churn_rate_cr/100*100000), ls = '--', c='red')
    return graph_churn 
    
graph_churn_column = graph_churn_col(Final_Data, churn_rate_cr)

def train_norm_and_pca(Final_Data):
    """
    Entrainement du jeu de données de Final_Data, normalisation et application d'une ACP pour réduire
    les données

    Parameters
    ----------
    Final_Data : dataframe
        Dataframe nettoyée et préparée pour la modélisation

    Returns
    -------
    X_test : dataframe
        Jeu de données entraîné
        
    y_test : dataframe
        Colonne churn entraînée
        
    X_acp : array
        Liste ou l'ACP est appliquée
        
    y_train_res : dataframe
        Colonne churn entraînée et normalisée
        
    X_test_acp : array
        ACP appliquée sur la dataframe normalisée

    """
    #master_df = Final_Data.copy()
    churn = Final_Data["churn"]
    df_test = Final_Data.drop('churn', axis=1)
    X = df_test
    y = churn
    adasyn = ADASYN(random_state=42)
    #sm = SMOTE(random_state=12)
    X, y = adasyn.fit_sample(X, y)
    # Create train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    X_train = X_train.reset_index(drop=True)
    mobile_number = X_train['mobile_number']
    X = X.drop(['mobile_number'], axis=1)
    X_train = X_train.drop(['mobile_number'], axis=1)
    X_test = X_test.reset_index(drop=True)
    X_test = X_test.drop(['mobile_number'], axis=1)
    y_test = y_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    scal = StandardScaler()
    X_train_scal = scal.fit_transform(X_train)
    X_test_scal = scal.transform(X_test)
    
    from sklearn.decomposition import PCA
    acp = PCA(svd_solver='randomized')
    acp.fit(X_train)
    #Screeplot for the PCA components
    fig = plt.figure(figsize = (8,5))
    plt.plot(np.cumsum(acp.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
    # Initialize pca with 70 components
    acp = PCA(n_components=30)
    X_train_acp = acp.fit_transform(X_train)
    X_acp = acp.fit_transform(X)
    #creating correlation matrix for the principal components
    corrmat = np.corrcoef(X_acp.transpose())
    # 1s -> 0s in diagonals
    corrmat_nodiag = corrmat - np.diagflat(corrmat.diagonal())
    #print("max corr:",corrmat_nodiag.max(), ", min corr: ", corrmat_nodiag.min(),)
    # we see that correlations are indeed very close to 0
    #Applying selected components to the test data - 70 components
    X_test_acp = acp.transform(X_test)
    
    
    return X_train_acp, X_test, y_test, X_acp, y_train, X_test_acp, y, X_train, mobile_number

X_train_acp, X_test, y_test, X_acp, y_train, X_test_acp, y, X_train, mobile_number= train_norm_and_pca(Final_Data)
pd.options.mode.chained_assignment = None  # default='warn'

def q_function():
    """
    Définit une variable rentrée par l'utilisateur qui servira de taux de réponse prédit
    pour le calcul du lift qui suit

    Returns
    -------
    r : integer
        Valeur sous forme de pourcentage compris entre 0 et 1 pour la définition du taux de
        réponse prédit pour le lift
        
    model_scores : list
        Liste vide ou seront stocké les résultats des modèles appliqués.

    """
    x = input("\nEntrez le pourcentage Lift : ")
    if x.isdigit():
        c = int(x)
        if c >= 0 and c <= 100 :
            r = c/100
            c = str(c)
            print("\nLift à " + c + "% choisi")
            return r
        elif c < 0 or c > 100 :
            print("\nLe pourcentage doit être compris entre 0 et 100")
            print("Veuillez rentrer une valeur valide")
            q_function()
    else :
        print("\nLe pourcentage doit être une valeur numérique")
        print("Veuillez rentrer une valeur valide")        
        q_function()

q = 1 - q_function()
model_scores = []

def models_scores(q, X_train_acp, X_test, y_test, X_acp, X_test_acp, models, models_name):
    """
    Définit une variable rentrée par l'utilisateur qui servira de taux de réponse prédit
    pour le calcul du lift qui suit

    Parameters
    ----------
    q : float
        Valeur sous forme de pourcentage compris entre 0 et 1 pour la définition du taux de
        réponse prédit pour le lift
        
    X_test : dataframe
        Jeu de données entraîné
        
    y_test : dataframe
        Colonne churn entraînée
        
    X_acp : array
        Liste ou l'ACP est appliquée
        
        
    X_test_acp : array
        ACP appliquée sur la dataframe normalisée
        
    Returns
    -------
    None.

    """
    for k in range(0, len(models)):
        models[k].fit(X_train_acp, y_train)
        y_pred = models[k].predict(X_test_acp)
        acc_score = round(accuracy_score(y_test, y_pred)*100,2)
        roc_score = round(roc_auc_score(y_test, y_pred)*100,2)
        prec_score = round(precision_score(y_test, y_pred)*100,2)
        rec_score = round(recall_score(y_test, y_pred)*100,2)
        churn_rate_prob = sum(y_test) / len(y_test)
        X_test["y_pred"]=pd.Series(models[k].predict_proba(X_test_acp)[:,1])
        X_test["y"] = pd.Series(y_test)
        top_xp = X_test[X_test["y_pred"]>X_test["y_pred"].quantile(q=q)]
        lift_xp = (top_xp["y"].sum() / top_xp.shape[0]) / churn_rate_prob
        model_scores.append({'model_name':models_name[k], 'acc_score':acc_score,
                             'roc_score':roc_score,'prec_score':prec_score,
                             'rec_score':rec_score, 'lift_xp':lift_xp})
    


def models_comparison():
    """
    Compare les différents modèles et stocke les résultats dans une dataframe

    Returns
    -------
    models : list
        Liste contenant les différents modèles à tester
        
    models_name : list
        Liste contenant lesnoms des différents modèles à tester
        
    model_score_df : dataframe
        Dataframe contenant l'analyse des modèles testés

    """
    models = [DecisionTreeClassifier(max_depth=6, random_state=10), 
              LogisticRegression(random_state=10),
              KNeighborsClassifier(),
              SVC(kernel='rbf', class_weight='balanced', random_state=10, probability=True),
              RandomForestClassifier(class_weight='balanced', random_state=10),
              ExtraTreesClassifier(class_weight='balanced', random_state=10),
              GradientBoostingClassifier(random_state=10),
              AdaBoostClassifier(random_state=10),
              XGBClassifier(random_state=10)]
    models_name = ["Decision Tree Classifier",
                   "Logistic Regression",
                   "K-Neighbors Classifier",
                   "SVC rbf kernel",
                   "Random Forest Classifier",
                   "Extra Trees Classifier",
                   "Gradient Boosting Classifier",
                   "ADA Boost Classifier",
                   "XGB Classifier"]
    models_scores(q, X_train_acp, X_test, y_test, X_acp, X_test_acp, models, models_name)
    model_score_df = pd.DataFrame(model_scores,columns=['model_name', 'acc_score',
                                                        'roc_score','prec_score',
                                                        'rec_score', 'lift_xp'])
    model_score_df = model_score_df.drop_duplicates(subset=None, keep='first', inplace=False)
    model_score_df = model_score_df.sort_values(["lift_xp","rec_score","roc_score"], ascending=False)
    return models, models_name, model_score_df

models, models_name, model_score_df = models_comparison()



def cross_val(X_acp, y):
            
    RANDOM_STATE = 42
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = RANDOM_STATE)

    # Cross validation for KNeighborsClassifier
    model = ExtraTreesClassifier(random_state=RANDOM_STATE)
    etc_cv_scores = cross_val_score(model, X_acp, y, scoring='recall', cv = skf, n_jobs=-1)
    print('KNeighborsClassifier cv_score_mean : ', round(etc_cv_scores.mean() * 100, 2))
    print('KNeighborsClassifier cv_score_std : ', round(etc_cv_scores.std() * 100, 2))

    # Cross validation for SVC_Kernel_rbf
    model = RandomForestClassifier(random_state=RANDOM_STATE)
    rfc_cv_scores = cross_val_score(model, X_acp, y, scoring='recall', cv = skf, n_jobs=-1)
    print('RandomForestClassifier cv_score_mean : ', round(rfc_cv_scores.mean() * 100, 2))
    print('RandomForestClassifier cv_score_std : ', round(rfc_cv_scores.std() * 100, 2))

    # Cross validation for ExtraTreesClassifier
    model = XGBClassifier(random_state=RANDOM_STATE)
    xgb_cv_scores = cross_val_score(model, X_acp, y, scoring='recall', cv = skf, n_jobs=-1)
    print('ExtraTreesClassifier cv_score_mean : ', round(xgb_cv_scores.mean() * 100, 2))
    print('ExtraTreesClassifier cv_score_std : ', round(xgb_cv_scores.std() * 100, 2))


    cv_scores_df = pd.DataFrame({'index':np.linspace(1,n_splits,n_splits),'etc':etc_cv_scores,'rfc':rfc_cv_scores,'xgb':xgb_cv_scores})
    return cv_scores_df
    
cv_scores_df = cross_val(X_acp, y)
cv_scores_df.plot.line(x='index', y=['etc','rfc','xgb'])





def hyperparameters_opti(X_acp, y, X_train_acp, y_train):
    n_splits = 5
    n_estimators =  [int(x) for x in np.linspace(start = 1, stop = 5000, num = 5000)]
    min_samples_split = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]

    params = {
            'n_estimators': n_estimators,
            'min_samples_split': min_samples_split
         }

    # initialize the KNN model
    model = ExtraTreesClassifier(n_jobs=-1)

    # initialize the StratifiedKFold 
    skf = StratifiedKFold(n_splits=n_splits, shuffle = True, random_state = 42)

    # initialize the RandomSearchCV
    #grid = GridSearchCV(estimator=model, param_grid=params, scoring='recall', n_jobs=-1, cv=skf.split(X,Y), verbose=3 )
    grid = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter = 1, scoring='recall', n_jobs=-1, cv=skf.split(X_acp,y), verbose=3, random_state=42 )
    
    # fit the whole pca dataset
    grid.fit(X_acp, y)

    # print the grid results
    print('\n Best estimator:')
    print(grid.best_estimator_)
    print('\n Best score:')
    print(grid.best_score_ * 2 - 1)
    print('\n Best parameters:')
    print(grid.best_params_)


    model = grid.best_estimator_
    model.fit(X_train_acp, y_train)
    Y_pred = model.predict(X_test_acp)
    Y_pred_proba = model.predict_proba(X_test_acp)

    return Y_pred, Y_pred_proba

Y_pred, Y_pred_proba = hyperparameters_opti(X_acp, y, X_train_acp, y_train)




def draw_roc(actual, probs):
    fpr, tpr, thresholds = roc_curve( actual, probs, drop_intermediate = False )
    auc_score = roc_auc_score( actual, probs )
    plt.figure(figsize=(6, 6))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()





def model_eval(y_test, Y_pred):
    
    # Classification Report
    print('\nClassification Report : \n\n', classification_report(y_test, Y_pred))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, Y_pred).ravel()
    print('\nTN = {0}, FP = {1}, FN = {2}, TP = {3}\n\n'.format(tn, fp, fn, tp))

    # Model evaluation
    acc_score = round(accuracy_score(y_test, Y_pred)*100,2)
    roc_score = round(roc_auc_score(y_test, Y_pred)*100,2)
    prec_score = round(precision_score(y_test, Y_pred)*100,2)
    rec_score = round(recall_score(y_test, Y_pred)*100,2)
    print("acc_score", acc_score, "\nroc_score", roc_score, "\nprec_score", prec_score, "\nrec_score", rec_score)

    return draw_roc(y_test, Y_pred)
    
model_eval_kn = model_eval(y_test, Y_pred)



def cutoff_and_matrix():
    Y_pred_final = pd.DataFrame({'actual':y_test,'pred_nonchurn_prob':Y_pred_proba[:,0],'pred_churn_prob':Y_pred_proba[:,1],'predicted':Y_pred})
    Y_pred_final.head(5)

    numbers = [float(x)/10 for x in range(10)]
    for i in numbers:
        Y_pred_final[i]= Y_pred_final['pred_churn_prob'].map( lambda x: 1 if x > i else 0)
    Y_pred_final.head()

    Y_pred_final.to_csv(r"C:\Users\Matt\Documents\ESME_inge5\PFE\Y_pred_final.csv")

    # calculate accuracy sensitivity and specificity for various probability cutoffs.
    cutoff_df = pd.DataFrame( columns = ['prob','accuracy','sensi','speci'])

    num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
    for i in num:
        cm1 = confusion_matrix( Y_pred_final['actual'], Y_pred_final[i] )
        total1=sum(sum(cm1))
        accuracy = (cm1[0,0]+cm1[1,1])/total1
        sensi = cm1[0,0]/(cm1[0,0]+cm1[0,1])
        speci = cm1[1,1]/(cm1[1,0]+cm1[1,1])
        cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]
        print(cutoff_df)
        cutoff_df.to_csv(r"C:\Users\Matt\Documents\ESME_inge5\PFE\cutoff_df.csv")

    # plot accuracy sensitivity and specificity for various probabilities.
    cutoff_df.plot.line(x='prob', y=['accuracy','sensi','speci'])

    Y_pred_final['final_predicted'] = Y_pred_final['pred_churn_prob'].map( lambda x: 1 if x > 0.4 else 0)


    # Classification Report
    print('\nClassification Report : \n\n', classification_report(y_test, Y_pred_final['final_predicted']))

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, Y_pred_final['final_predicted']).ravel()
    print('\nTN = {0}, FP = {1}, FN = {2}, TP = {3}\n\n'.format(tn, fp, fn, tp))

    # Model evaluation
    acc_score = round(accuracy_score(y_test, Y_pred)*100,2)
    roc_score = round(roc_auc_score(y_test, Y_pred)*100,2)
    prec_score = round(precision_score(y_test, Y_pred)*100,2)
    rec_score = round(recall_score(y_test, Y_pred)*100,2)
    print("acc_score", acc_score, "\nroc_score", roc_score, "\nprec_score", prec_score, "\nrec_score", rec_score)


    # ROC-AUC curve
    draw_roc(y_test, Y_pred_final['final_predicted'])

cutoff_and_matrix()








def features_imp(Final_Data):
    
    SP_data = Final_Data.copy()
    churn = SP_data['churn']
    SP_data = SP_data.drop('churn', axis=1)

    X = SP_data.reset_index(drop=True)
    Y = churn.reset_index(drop=True)

    model = ExtraTreesClassifier(min_samples_split=4, n_estimators=796, n_jobs=-1, random_state=42)
    model.fit(X, Y)

    # Check the feature importance score for each feature
    feature_imp_df = pd.DataFrame({'Feature':SP_data.columns, 'Score':model.feature_importances_})
    # Order the features by max score
    feature_imp_df = feature_imp_df.sort_values('Score', ascending=False).reset_index()
    feature_imp_df.head(50)

    (pd.Series(model.feature_importances_, index = SP_data.columns)
     .nlargest(50)
     .plot(kind='barh', figsize=(18,12)))


    top_25_features = feature_imp_df[:50]['Feature']
    plt.figure(figsize=(18,12))
    sns.heatmap(data[top_25_features].corr(), annot=True)

    return X, Y, churn, SP_data

X, Y, churn, SP_data = features_imp(Final_Data)











def feat_graphs_and_matrix(SP_data, churn, Final_Data):

    cols = ['total_og_mou_8','total_rech_num_8','fb_user_8','max_rech_amt_8',
        'aon','roam_og_mou_8','arpu_8','last_day_rch_amt_8','loc_og_t2m_mou_8']

    X = SP_data[cols].reset_index(drop=True)
    Y = churn.reset_index(drop=True)

    logm = sm.GLM(Y,(sm.add_constant(X)), family = sm.families.Binomial())
    modres = logm.fit()
    logm.fit().summary()
    SP_data = SP_data[cols]
    SP_data = (( SP_data - SP_data.mean()) / SP_data.std())

    X = SP_data
    Y = churn
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.3, random_state=42)
    print(X_train.shape, Y_train.shape)
    print(X_test.shape, Y_test.shape)
    
    print('Y_train :', Counter(Y_train))
    print('Y_test :', Counter(Y_test))
    
    
    adasyn = ADASYN(random_state=42)
    X_train, Y_train = adasyn.fit_sample(X_train, Y_train)
    
    print('Class Balance count : ', Counter(Y))
    
    
    
    
    model = ExtraTreesClassifier(bootstrap=False, class_weight='balanced', criterion='gini', max_depth=60, max_features='sqrt',
                max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, min_samples_leaf=1, min_samples_split=4, 
                min_weight_fraction_leaf=0.0, n_estimators=796, n_jobs=-1, oob_score=False, random_state=42, verbose=0, warm_start=False)
    model.fit(X_train, Y_train)
    Y_pred_1 = model.predict(X_test)

    print('\nClassification Report : \n\n', classification_report(Y_test, Y_pred_1))
    
    tn, fp, fn, tp = confusion_matrix(Y_test, Y_pred_1).ravel()
    cm = confusion_matrix(Y_test, Y_pred_1) 
    
    df_cm = pd.DataFrame(cm, index = (0, 1), columns = (0, 1))
    plt.figure(figsize = (28,20))
    fig, ax = plt.subplots()
    sns.set(font_scale=1.4)
    sns.heatmap(df_cm, annot=True, fmt='g'#,cmap="YlGnBu" 
               )
    class_names=[0,1]
    tick_marks = np.arange(len(class_names))
    plt.tight_layout()
    plt.title('Confusion matrix\n', y=1.1)
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)
    ax.xaxis.set_label_position("top")
    plt.ylabel('Actual label\n')
    plt.xlabel('Predicted label\n')
    
    print('\nTN = {0}, FP = {1}, FN = {2}, TP = {3}\n\n'.format(tn, fp, fn, tp))
    
    acc_score = round(accuracy_score(Y_test, Y_pred_1)*100,2)
    roc_score = round(roc_auc_score(Y_test, Y_pred_1)*100,2)
    prec_score = round(precision_score(Y_test, Y_pred_1)*100,2)
    rec_score = round(recall_score(Y_test, Y_pred_1)*100,2)
    print("acc_score", acc_score, "\nroc_score", roc_score, "\nprec_score", prec_score, "\nrec_score", rec_score)
    
    draw_roc(Y_test, Y_pred_1)
    
    cmap = plt.get_cmap('Spectral')
    colors = [cmap(i) for i in np.linspace(0, 1, 9)]
    
    plt.figure(figsize=(10,10))
    plt.pie(model.feature_importances_, labels=SP_data.columns, autopct='%1.1f%%', shadow=False, colors=colors)
    plt.axis('equal')
    plt.show()
    
    
    Final_Data = Final_Data.reset_index(drop=True)
    
    SP_data_fin = Final_Data.copy()
    
    #selected final list of features
    cols = ['total_og_mou_8','total_rech_num_8','fb_user_8','max_rech_amt_8',
            'aon','roam_og_mou_8','arpu_8','last_day_rch_amt_8','loc_og_t2m_mou_8']
    
    SP_data_fin = SP_data_fin[cols]
    SP_data_fin['churn'] = Final_Data['churn']
    
    plt.figure(figsize=(25,25))
    
    for i, col in enumerate(cols):
        plt.subplot(3,3,i+1)
        ax = sns.distplot(SP_data_fin.loc[SP_data_fin['churn']==0, [col]], color='b', label='Non-Churn')
        ax = sns.distplot(SP_data_fin.loc[SP_data_fin['churn']==1, [col]], color='r', label='Churn')
        ax.legend()
        plt.xlabel(col)
        plt.title('{0} dist plot'.format(col))
        
    return SP_data_fin
        
SP_data_fin = feat_graphs_and_matrix(SP_data, churn, Final_Data)
    
    
    
    
    
    
    
    
    
    
    


# Binary Classification with Sonar Dataset: Baseline



def Neural_network(Final_Data, X_train_acp, y_train, X_test_acp, y_test):

    dataset = Final_Data.drop(['mobile_number'], axis=1)
    dataset = dataset.values
    # split into input (X) and output (Y) variables
    X = dataset[:,0:151].astype(float)
    Y = dataset[:,151]
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    encoded_Y = encoder.transform(Y)
    # baseline model
    def create_baseline():
    	# create model
    	model = Sequential()
    	model.add(Dense(70, input_dim=30, activation='relu'))
    	model.add(Dense(1, activation='sigmoid'))
    	# Compile model
    	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    	return model
    
    neural_network = create_baseline()
    
    neural_network.fit(X_train_acp, y_train)
    y_pred = neural_network.predict(X_test_acp)
    acc_score = round(accuracy_score(y_test, y_pred.round())*100,2)
    roc_score = round(roc_auc_score(y_test, y_pred.round())*100,2)
    prec_score = round(precision_score(y_test, y_pred.round())*100,2)
    rec_score = round(recall_score(y_test, y_pred.round())*100,2)
    churn_rate_prob = sum(y_test) / len(y_test)
    X_test["y_pred"]=pd.Series(neural_network.predict_proba(X_test_acp)[:,0])
    X_test["y"] = pd.Series(y_test)
    top_xp = X_test[X_test["y_pred"]>X_test["y_pred"].quantile(q=q)]
    lift_xp = (top_xp["y"].sum() / top_xp.shape[0]) / churn_rate_prob
    model_scores.append({'model_name':"Neural Network", 'acc_score':acc_score,
                                 'roc_score':roc_score,'prec_score':prec_score,
                                 'rec_score':rec_score, 'lift_xp':lift_xp}) 

    model_score_df = pd.DataFrame(model_scores,columns=['model_name', 'acc_score',
                                                        'roc_score','prec_score',
                                                        'rec_score', 'lift_xp'])
    model_score_df = model_score_df.drop_duplicates(subset=None, keep='first', inplace=False)
    model_score_df = model_score_df.sort_values(["lift_xp","rec_score","roc_score"], ascending=False)
    
    
Neural_network(Final_Data, X_train_acp, y_train, X_test_acp, y_test)










#model_score_df.to_csv(r"C:\Users\Matt\Documents\ESME_inge5\PFE\model_scores.csv")
def Propension(mobile_number, X_train_acp, y_train, X_test_acp, y_test): 
    
    mobile_number = mobile_number.reset_index(drop=True)
    
    model_final = ExtraTreesClassifier(min_samples_split=4, n_estimators=796, random_state=42)
    
    
    model_final.fit(X_train_acp, y_train)
    
    y_pred = model_final.predict(X_test_acp)
    
    y_pred_proba = model_final.predict_proba(X_test_acp)
    y_pred_proba = y_pred_proba [:, 1]
    
    
    final_results = pd.concat([mobile_number, y_test], axis = 1).dropna()
    final_results['Predictions'] = y_pred
    final_results["Propension de Churn(%)"] = y_pred_proba
    final_results["Propension de Churn(%)"] = final_results["Propension de Churn(%)"]*100
    final_results["Propension de Churn(%)"] = final_results["Propension de Churn(%)"].round(2)
    final_results = final_results[['mobile_number', 'churn', 'Predictions', 'Propension de Churn(%)']]
    final_results ['Classement'] = pd.qcut(final_results['Propension de Churn(%)'].rank(method = 'first'),10,labels=range(10,0,-1))
    
    final_results = final_results.sort_values(by=['Propension de Churn(%)'], ascending=False).reset_index(drop=True)
    print (final_results)
    
    return final_results

#final_results.to_csv(r"C:\Users\Matt\Documents\ESME_inge5\PFE\final_results.csv")


final_results = Propension(mobile_number, X_train_acp, y_train, X_test_acp, y_test)