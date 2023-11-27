# Import the libraries
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np  # linear algebra
import seaborn as sns  # statistical data visualization
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from tabulate import tabulate
import matplotlib.pyplot as plt
import os
#from tensorflow.keras.models import Sequential
#from tensorflow.keras.layers import LSTM, Dense


# RIGHE DA ELIMINARE: 521009

"""
    Funzione che stampa i dati
"""
def print_data():

    print("SONO IN PRINT DATA")
    # Read the data
    data = pd.read_csv('malicious_phish.csv') #nrows=50000
    data.drop(index=521009)
    # Print the first 5 rows
    print("Data head:")
    print(data.head())
    print("--------------------\n\n")
    # Print the data info
    print("Data info:")
    print(data.info())
    print("--------------------\n\n")
    # Print 10 data sample
    print("Data sample:")
    print(data.sample(10))
    print("--------------------\n\n")
    # Print the data shape
    print("Data shape:")
    print(data.shape)
    print("--------------------\n\n")
    # Print the data describe
    print("Data describe:")
    print(data.describe())
    print("--------------------\n\n")
    # Print the url length
    print("Url length:")
    data['url_length'] = data['url'].apply(len)
    print(data[['url', 'url_length']])
    print("--------------------\n\n")
    #clean_data(data)


"""
    Funzione che pulisce i dati
"""
def clean_data(data):
    print("SONO IN CLEAN DATA")
    #print("Data types: ")
    #data['type'] = data['type'].values.astype(str)
    #print(data['type'])
    #print("--------------------\n\n")

    # Delete the duplicates and NA valuesd
    print("Drop duplicates and NA values: ")
    data = data.drop_duplicates()
    print("--------------------\n\n")
    # Delete the rows with NA values
    print("Dropna: ")
    data = data.dropna()
    print("--------------------\n\n")
    # Print the data shape
    print("Data shape:")
    print(data.shape)
    print("--------------------\n\n")
    # Print the data describe
    print("Data describe:")
    print(data.describe())
    print("--------------------\n\n")
    data_discovery(data)


"""
    Funzione che esegue la data discovery
"""
def data_discovery(data):
    print("SONO IN DATA DISCOVERY")

    # Percentual of each type
    def perc(values):
        pct = float(values / data['type'].count()) * 100
        return round(pct, 2)

    # Group by type
    print("Group by type: ")
    grouped_data = data.groupby('type').size()
    print(grouped_data)
    print("--------------------\n\n")

    d_benign = grouped_data['benign']
    d_phishing = grouped_data['phishing']
    d_defacement = grouped_data['defacement']
    d_malware = grouped_data['malware']

    # Print the percentual of each type
    print("Percentual of each type: ")
    print('Benign:', d_benign, '%', perc(d_benign))
    print('Phishing:', d_phishing, '%', perc(d_phishing))
    print('Defacement:', d_defacement, '%', perc(d_defacement))
    print('Malware:', d_malware, '%', perc(d_malware))
    print("--------------------\n\n")

    # Plot the data
    p_bar = data['type'].value_counts().plot(kind='bar', title='Type', figsize=(10, 8))
    p_bar.set_ylabel('Count')

    # Save the plot
    folder_path_info = '/Users/marcoromanella/Desktop/MaliciousUrls/img/info'
    file_name_info = 'type.png'
    file_path_info = os.path.join(folder_path_info, file_name_info)
    plt.savefig(file_path_info, dpi=300, bbox_inches='tight')
    plt.show()


    # Grafico a torta
    data['type'].value_counts().plot.pie(autopct='%1.1f%%')
    folder_path_info = '/Users/marcoromanella/Desktop/MaliciousUrls/img/info'
    file_name_info = 'pie.png'
    file_path_info = os.path.join(folder_path_info, file_name_info)
    plt.savefig(file_path_info, dpi=300, bbox_inches='tight')
    plt.show()

    print("--------------------")
    #data_preprocessing(data)


"""
    Funzione che pre-elabora i dati
"""
def data_preprocessing(data):
    print("SONO IN DATA PREPROCESSING")
    """
    data.isnull().sum()
    print("Numeri dati prima dell'eliminazione di quelli nulli: " + data.shape)
    data = data.dropna()
    print("Numeri dati dopo l'eliminazione di quelli nulli: " + data.shape)
    """
    print("--------------------")
    training_and_testing(data)


def training_and_testing_splitting_csv(data):
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


def training_and_testing(data):
    print("SONO IN TRAINING AND TESTING")

    # Dimensione del batch
    batch_size = 10000
    total_raws = len(data)

    models = [
        MultinomialNB(),           # FUNZIONA
        GaussianNB(),              # FUNZIONA
        DecisionTreeClassifier(),  # FUNZIONA
        RandomForestClassifier(),  # FUNZIONA
    ]

    results = []

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
            # print(data[start:end])

            # Seleziona il batch corrente
            batch_data = data[start:end]

            # Preparazione delle caratteristiche e delle etichette
            x = batch_data['url']
            y = batch_data['type']

            # Conversione delle etichette in formato numerico
            label_mapping = {'benign': 1, 'phishing': 2, 'defacement': 3, 'malware': 4}
            y = y.map(label_mapping)

            # Divisione dei dati in dati di addestramento e dati di test
            train_feature, test_feature, train_labels, test_labels = train_test_split(x, y, test_size=0.2, stratify=y,
                                                                                      random_state=42)

            # Creazione di una rappresentazione numerica utilizzando TF-IDF
            tfidf = TfidfVectorizer()
            train_vectors = tfidf.fit_transform(train_feature).toarray()
            test_vectors = tfidf.transform(test_feature).toarray()

            # Trasformazione delle etichette in array 1D
            # train_labels = train_labels.values.reshape(-1)
            # test_labels = test_labels.values.reshape(-1)

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
            # roc_auc = roc_auc_score(test_labels, predictions, average='weighted', multi_class='ovr')

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

        salva_risultati(results, model, avarage_accuracy, avarage_precision, avarage_recall, avarage_f1)

        """
        print("Accuracy: ", avarage_accuracy)
        print("Precision: ", avarage_precision)
        print("Recall: ", avarage_recall)
        print("F1: ", avarage_f1)
        """
        crea_tabella(model, avarage_accuracy, avarage_precision, avarage_recall, avarage_f1)
        crea_confusion_matrix(model, total_labels, total_predictions)

    mostra_risultati(results)
    print("\n--------------------\n\n")


# Creazione tabella in formato png dei vari risultati
def crea_tabella(model, avarage_accuracy, avarage_precision, avarage_recall, avarage_f1):
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

    folder_path_table = '/Users/marcoromanella/Desktop/MaliciousUrls/img/table'
    file_name_table = f'{model.__class__.__name__}Table.png'
    file_path_table = os.path.join(folder_path_table, file_name_table)
    plt.savefig(file_path_table, bbox_inches='tight', dpi=300)

    # Mostra la tabella
    plt.show()


# Creazione della matrice di confusione
def crea_confusion_matrix(model, total_labels, total_predictions):
    confusion = confusion_matrix(total_labels, total_predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion / np.sum(confusion), annot=True, fmt='0.2%', cmap='Blues')
    plt.title(model.__class__.__name__)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Salvataggio della matrice di confusione
    folder_path_confusion_matrix = '/Users/marcoromanella/Desktop/MaliciousUrls/img/confusion_matrix'
    file_name_confusion_matrix = f'{model.__class__.__name__}ConfusionMatrix.png'
    file_path_confusion_matrix = os.path.join(folder_path_confusion_matrix, file_name_confusion_matrix)
    plt.savefig(file_path_confusion_matrix, bbox_inches='tight', dpi=300)
    plt.show()


# Salvataggio dei risultati in un dizionario
def salva_risultati(results, model, avarage_accuracy, avarage_precision, avarage_recall, avarage_f1):
    results.append({'Model': model.__class__.__name__,
                    'Accuracy': avarage_accuracy,
                    'Precision': avarage_precision,
                    'Recall': avarage_recall,
                    'F1': avarage_f1})


def mostra_risultati(results):
    print("SONO IN MOSTRA RISULTATI")
    table_results = pd.DataFrame(results)
    print(table_results.head())


    table_results.plot.bar(x='Model', y='Accuracy', legend=False)
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.title('Accuracy of Different Models')
    #plt.xticks(rotation=45)
    folder_path_results = '/Users/marcoromanella/Desktop/MaliciousUrls/img/results'
    file_name_results = 'accuracy.png'
    file_path_results = os.path.join(folder_path_results, file_name_results)
    plt.savefig(file_path_results, bbox_inches='tight', dpi=300)
    plt.show()


    table_results.plot.bar(x='Model', y='Precision', legend=False)
    plt.xlabel('Model')
    plt.ylabel('Precision')
    plt.title('Precision of Different Models')
    #plt.xticks(rotation=45)
    folder_path_results = '/Users/marcoromanella/Desktop/MaliciousUrls/img/results'
    file_name_results = 'precision.png'
    file_path_results = os.path.join(folder_path_results, file_name_results)
    plt.savefig(file_path_results, bbox_inches='tight', dpi=300)
    plt.show()

    table_results.plot.bar(x='Model', y='Recall', legend=False)
    plt.xlabel('Model')
    plt.ylabel('Recall')
    plt.title('Recall of Different Models')
    #plt.xticks(rotation=45)
    folder_path_results = '/Users/marcoromanella/Desktop/MaliciousUrls/img/results'
    file_name_results = 'recall.png'
    file_path_results = os.path.join(folder_path_results, file_name_results)
    plt.savefig(file_path_results, bbox_inches='tight', dpi=300)
    plt.show()

    table_results.plot.bar(x='Model', y='F1', legend=False)
    plt.xlabel('Model')
    plt.ylabel('F1')
    plt.title('F1 of Different Models')
    #plt.xticks(rotation=45)
    folder_path_results = '/Users/marcoromanella/Desktop/MaliciousUrls/img/results'
    file_name_results = 'f1.png'
    file_path_results = os.path.join(folder_path_results, file_name_results)
    plt.savefig(file_path_results, bbox_inches='tight', dpi=300)
    plt.show()

    table_results.plot.bar(x='Model', y=['Accuracy', 'Precision', 'Recall', 'F1'], legend=False)
    plt.xlabel('Model')
    plt.ylabel('Accuracy, Precision, Recall, F1')
    plt.title('Accuracy, Precision, Recall, F1 of Different Models')
    #plt.xticks(rotation=45)
    folder_path_results = '/Users/marcoromanella/Desktop/MaliciousUrls/img/results'
    file_name_results = 'accPreRecF1.png'
    file_path_results = os.path.join(folder_path_results, file_name_results)
    plt.savefig(file_path_results, bbox_inches='tight', dpi=300)
    plt.show()

