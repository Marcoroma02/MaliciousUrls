# Import the libraries
import pandas as pd                  # data processing, CSV file I/O (e.g. pd.read_csv)
import re                            # regular expressions
from urllib.parse import urlparse, urlunparse # url parsing
import numpy as np                    # linear algebra
import matplotlib.pyplot as plt       # data visualization
import seaborn as sns                 # statistical data visualization
from sklearn.model_selection import train_test_split # data split


"""
    Funzione che stampa i dati
"""
def printData():
    print("SONO IN PRINT DATA")
    # Read the data
    data = pd.read_csv('malicious_phish.csv')

    # Print the first 5 rows
    print(data.head())
    print("--------------------")
    print(data.info())
    print("--------------------")
    print(data.sample(10))
    print("--------------------")
    print(data.shape)
    print("--------------------")
    print(data.describe())
    print("--------------------")
    cleanData(data)


"""
    Funzione che pulisce i dati
"""
def cleanData(data):
    print("SONO IN CLEAN DATA")
    data['type'] = data['type'].values.astype(str)
    # data['type'] = data['type'].map({'benign': 1, 'phishing': 2, 'defacement': 3})
    print(data['type'])
    print("--------------------")
    dataDiscovery(data)


"""
    Funzione che esegue la data discovery
"""
def dataDiscovery(data):
    print("SONO IN DATA DISCOVERY")

    # Percentual of each type
    def perc(values):
        pct = float(values / data['type'].count()) * 100
        return round(pct, 2)

    # Group by type
    grouped_data = data.groupby('type').size()
    print(grouped_data)

    d_benign = grouped_data['benign']
    d_phishing = grouped_data['phishing']
    d_defacement = grouped_data['defacement']

    print('Benign:', d_benign, '%', perc(d_benign))
    print('Phishing:', d_phishing, '%', perc(d_phishing))
    print('Defacement:', d_defacement, '%', perc(d_defacement))

    # Plot the data
    p_bar = data['type'].value_counts().plot(kind='bar', title='Type', figsize=(10, 8));
    p_bar.set_ylabel('Count');
    plt.show()
    print("--------------------")
    dataPreprocessing(data)


"""
    Funzione che pre-elabora i dati
"""
def dataPreprocessing(data):
    print("SONO IN DATA PREPROCESSING")
    """
    data.isnull().sum()
    print("Numeri dati prima dell'eliminazione di quelli nulli: " + data.shape)
    data = data.dropna()
    print("Numeri dati dopo l'eliminazione di quelli nulli: " + data.shape)
    """
    
    # Divide il dataset in dati di addestramento e dati di test
    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['type'], random_state=42)

    # Salva i dati di addestramento e dati di test in file CSV separati
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    train_raw_num = len(train_df)
    test_raw_num = len(test_df)
    print("Train raw number: " + str(train_raw_num))
    print("Test raw number: " + str(test_raw_num))


