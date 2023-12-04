import numpy as np

import matplotlib.pyplot as plt

import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.tree import DecisionTreeClassifier # for surogate

# TODO : add meaningful comments

def model_evaluation(model, encoder, X, y, label = None):

    '''
    Function to generate a report on the performance of a pytorch model. 
    Reports Accuracy, Precision, Recall, F1-Score and Matthews Correlation Coefficent. 
    Displays a confusion matrix as well.
    
        Parameters:
            model   : Model to be evaluated
            encoder : OrdinalEncoder used to encode play types
            X       : Input features
            y       : True Labels
            label   : Header to use in graphic
        Returns:
            None
            
    '''

    model.eval()

    if torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    device = torch.device(device)

    with torch.no_grad():
        y_pred = model.forward(torch.Tensor(X).to(device)).argmax(dim = 1).cpu().numpy()


    accuracy = accuracy_score(y,y_pred)
    precision = precision_score(y,y_pred)
    recall = recall_score(y,y_pred)
    f1 = f1_score(y,y_pred)

    mcc = matthews_corrcoef(y, y_pred)

    print(F'''--- Metrics - {label} ---
    Accuracy: \t {100*accuracy:.4f}%
    Precision: \t {100*precision:.4f}%
    Recall: \t {100*recall:.4f}%
    F1-Score: \t {100*f1:.4f}%
    MCC: \t \t {mcc:.4f}
    ''')

    cf_matrix = confusion_matrix(y,y_pred, normalize = 'all')

    disp = ConfusionMatrixDisplay(confusion_matrix = cf_matrix,
                                  display_labels = encoder.categories_[0])

    disp.plot()
    plt.title(f'Confusion Matrix - {label}')
    plt.show()


def surrogate(data, pred):
    '''
    Function to train a surrogate model on the outputs of a model.
    Creates a bar graph for feature importance of the model.
    
        Parameters:
            data    : Data used to train the original model
            pred    : Predictions of the original model
            
    '''

    surrogate_model = DecisionTreeClassifier()
    surrogate_model.fit(data, pred)

    importances = surrogate_model.feature_importances_
    indices = np.argsort(importances)

    plt.title('Feature Importances')
    plt.barh([data[i] for i in indices], importances[indices])
    plt.yticks(fontsize = 7)

    plt.xlabel('Relative Importance')
    plt.show()
