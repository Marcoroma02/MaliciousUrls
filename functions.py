# Import the libraries
import pandas as pd                  # data processing, CSV file I/O (e.g. pd.read_csv)
import re                            # regular expressions
from urllib.parse import urlparse, urlunparse # url parsing
import numpy as np                    # linear algebra
import matplotlib.pyplot as plt       # data visualization
import seaborn as sns                 # statistical data visualization
from sklearn.model_selection import train_test_split # data split
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import SGDClassifier
import validators
from sklearn.model_selection import cross_val_score
from tabulate import tabulate

# RIGHE DA ELIMINARE: 521009

"""
    Funzione che stampa i dati
"""
def printData():
    print("SONO IN PRINT DATA")
    # Read the data
    data = pd.read_csv('malicious_phish.csv', nrows=4000)

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
    print(data['type'])
    print("Drop duplicates and NA values")
    #data = data.drop_duplicates()
    #print("Dropna")
    #data = data.dropna()
    #print("Validators")
    #data = data[data['url'].apply(lambda x: validators.url(x))]
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
    d_malware = grouped_data['malware']

    print('Benign:', d_benign, '%', perc(d_benign))
    print('Phishing:', d_phishing, '%', perc(d_phishing))
    print('Defacement:', d_defacement, '%', perc(d_defacement))
    print('Malware:', d_malware, '%', perc(d_malware))

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
    print("--------------------")
    trainingAndTesting(data)


def trainingAndTestingSplittingCSV(data):
    print("SONO IN TRAINING AND TESTING SPLITTING CSV")
    # Divide il dataset in dati di addestramento e dati di test
    train_df, test_df = train_test_split(data, test_size=0.2, stratify=data['type'], random_state=42)

    # Salva i dati di addestramento e dati di test in file CSV separati
    train_df.to_csv('train.csv', index=False)
    test_df.to_csv('test.csv', index=False)
    train_raw_num = len(train_df)
    test_raw_num = len(test_df)
    print("Train raw number: " + str(train_raw_num))
    print("Test raw number: " + str(test_raw_num))


def trainingAndTesting(data):
    print("SONO IN TRAINING AND TESTING")

    # Dimensione del batch
    batch_size = 1000

    total_raws = len(data)

    models = [
        DecisionTreeClassifier(),
        RandomForestClassifier(),
        LogisticRegression(),
        SGDClassifier()
    ]

    for model in models:
        total_accuracy = 0
        total_precision = 0
        total_recall = 0
        total_f1 = 0
        total_predictions = []
        total_labels = []
        i = 0

        for start in range(0, total_raws, batch_size):
            i += 1
            end = min(start + batch_size, total_raws)
            #print(data[start:end])

            # Seleziona il batch corrente
            batch_data = data[start:end]

            # Preparazione delle caratteristiche e delle etichette
            x = batch_data['url']
            y = batch_data['type']

            # Conversione delle etichette in formato numerico
            label_mapping = {'benign': 1, 'phishing': 2, 'defacement': 3, 'malware': 4}
            y = y.map(label_mapping)

            # Divisione dei dati in dati di addestramento e dati di test
            train_feature, test_feature, train_labels, test_labels = train_test_split(x, y, test_size=0.2, stratify=y, random_state=42)

            # Creazione di una rappresentazione numerica utilizzando TF-IDF
            tfidf = TfidfVectorizer()
            train_vectors = tfidf.fit_transform(train_feature).toarray()
            test_vectors = tfidf.transform(test_feature).toarray()

            # Trasformazione delle etichette in array 1D
            train_labels = train_labels.values.reshape(-1)
            test_labels = test_labels.values.reshape(-1)

            # Addestramento del modello
            print("MODELLO: " + model.__class__.__name__)
            model.fit(train_vectors, train_labels)
            predictions = model.predict(test_vectors)
            total_predictions.extend(predictions)
            total_labels.extend(test_labels)

            accuracy = accuracy_score(test_labels, predictions)
            precision = precision_score(test_labels, predictions, average='weighted', zero_division=1)
            recall = recall_score(test_labels, predictions, average='weighted')
            f1 = f1_score(test_labels, predictions, average='weighted')
            #roc_auc = roc_auc_score(test_labels, predictions, average='weighted', multi_class='ovr')

            # Aggiorna le metriche totali
            total_accuracy += accuracy
            total_precision += precision
            total_recall += recall
            total_f1 += f1

            print(f"Elaborazione n {i} completata per il batch {start + 1}-{end} su {total_raws} righe.")

        num_models = i
        print("Numero di modelli: ", num_models)

        avarage_accuracy = total_accuracy / num_models
        avarage_precision = total_precision / num_models
        avarage_recall = total_recall / num_models
        avarage_f1 = total_f1 / num_models

        print("Accuracy: ", avarage_accuracy)
        print("Precision: ", avarage_precision)
        print("Recall: ", avarage_recall)
        print("F1: ", avarage_f1)

        creaTabella(model, avarage_accuracy, avarage_precision, avarage_recall, avarage_f1)
        creaConfusionMatrix(model, total_labels, total_predictions)

        print("\n--------------------\n\n")


def creaTabella(model, avarage_accuracy, avarage_precision, avarage_recall, avarage_f1):
    print("SONO IN CREA TABELLA")
    # Creazione della tabella
    df = pd.DataFrame({'Name: ': [model.__class__.__name__],
                       'Accuracy: ': [avarage_accuracy],
                       'Precision: ': [avarage_precision],
                       'Recall: ': [avarage_recall],
                       'F1: ': [avarage_f1]})

    # Creazione della tabella utilizzando Tabulate
    table = tabulate(df, headers='keys', tablefmt='fancy_grid')

    # Configurazione del grafico 8 larghezza x 2 altezza
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.axis('off')

    # Aggiunta della tabella come testo all'asse
    ax.text(0, 0, table, fontsize=10, family='monospace')

    # Centrare l'immagine nel grafico
    fig.tight_layout(pad=3)

    # Salvataggio della tabella come immagine PNG
    plt.savefig('tabella.png', bbox_inches='tight', pad_inches=0.5, dpi=300)

    # Mostra la tabella
    plt.show()


def creaConfusionMatrix(model, total_labels, total_predictions):
    confusion = confusion_matrix(total_labels, total_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion / np.sum(confusion), annot=True, fmt='0.2%', cmap='Blues')
    plt.title(model.__class__.__name__)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()