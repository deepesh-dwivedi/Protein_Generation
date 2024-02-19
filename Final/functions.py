

# Standard Imports

import os
import copy
import numpy as np
from math import ceil
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


import torch
import torch.nn.functional as F
from torch.utils.data import random_split


import dgl
from dgl.dataloading import GraphDataLoader
from dgl.nn import GraphConv
from IPython.display import Latex


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


import torch_geometric
from torch_geometric.data import DataLoader, DenseDataLoader
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T
from torch_geometric.transforms import AddSelfLoops
from torch_geometric.nn import GCNConv, dense_diff_pool
from torch_geometric.utils import add_self_loops
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score, roc_curve, auc


import warnings
warnings.filterwarnings("ignore", category=UserWarning)





########################################
## UTILS
########################################


def set_reproducible():
    # The below is necessary to have reproducible behavior.
    import random as rn
    import os
    os.environ['PYTHONHASHSEED'] = '0'
    # The below is necessary for starting Numpy generated random numbers
    # in a well-defined initial state.
    np.random.seed(17)
    # The below is necessary for starting core Python generated random numbers
    # in a well-defined state.
    rn.seed(12345)
    
    
    
def compute_accuracy(preds, labels):
  # Converting the predictions and labels to tensors
  #preds = torch.tensor(preds)
  #labels = torch.tensor(labels)
  
  preds, labels = preds.detach(), labels

  # Getting the predicted classes
  preds = torch.argmax(preds, dim=1)

  # Computing the number of correct predictions
  correct = torch.sum(preds.view(-1) == labels.view(-1))

  # Computing the accuracy
  accuracy = correct.item() / len(labels)

  return accuracy

    

########################################
## LOADERS
########################################

    
def get_dgl_loaders(dataset, train_ratio, val_ratio, train_batch_size, val_batch_size, test_batch_size, generator):
    """Get Loaders and preprocessing with one hot encoder, since 60 types of node labels

    Args:
        dataset (_type_): _description_
        train_ratio (_type_): _description_
        val_ratio (_type_): _description_
        train_batch_size (_type_): _description_
        val_batch_size (_type_): _description_
        test_batch_size (_type_): _description_

    Returns:
        _type_: _description_
    """

    st = Counter()

    for graph in dataset.graph_lists:
        st.update(graph.ndata["node_attr"].numpy().astype(int).ravel().tolist())

    one_hot = OneHotEncoder(sparse_output=False)
    one_hot.fit(np.array(list(st.keys())).reshape(-1, 1))

    for graph in dataset.graph_lists:
        graph.ndata["node_attr"] = torch.from_numpy(one_hot.transform(graph.ndata["node_attr"].numpy()))
        

    num_train = int(len(dataset) * train_ratio)
    num_val = int(len(dataset) * val_ratio)
    num_test = len(dataset) - (num_train + num_val)

    training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test], generator=generator)

    print(f"\nLength training set {len(training_set)}, Length validation set {len(validation_set)}, Length test set {len(test_set)}")

    train_loader = GraphDataLoader(training_set, batch_size=train_batch_size, shuffle=True,  ddp_seed=generator.seed, drop_last=False)
    val_loader = GraphDataLoader(validation_set, batch_size=val_batch_size, shuffle=True,  ddp_seed=generator.seed, drop_last=False)
    test_loader = GraphDataLoader(test_set, batch_size=test_batch_size, shuffle=False,  ddp_seed=generator.seed, drop_last=False)
    
    return train_loader, val_loader, test_loader



def get_torch_loaders(dataset, train_ratio, val_ratio, train_batch_size, val_batch_size, test_batch_size, generator, dense=True):

    num_train = int(len(dataset) * train_ratio)
    num_val = int(len(dataset) * val_ratio)
    num_test = len(dataset) - (num_train + num_val)

    training_set, validation_set, test_set = random_split(dataset, [num_train, num_val, num_test], generator=generator)

    print(f"\nLength training set {len(training_set)}, Length validation set {len(validation_set)}, Length test set {len(test_set)}")

    if dense:
        train_loader = DenseDataLoader(training_set, batch_size=train_batch_size, shuffle=True)
        val_loader = DenseDataLoader(validation_set, batch_size=val_batch_size, shuffle=False)
        test_loader = DenseDataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    else:
        train_loader = DataLoader(training_set, batch_size=train_batch_size, shuffle=True)
        val_loader = DataLoader(validation_set, batch_size=val_batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader




########################################
## EARLY STOPPING
########################################


class EarlyStopping:
    """ Class used to store the best model checkpoint during training """

    def __init__(self, patience=5, delta=0, path="drive/MyDrive/MLNS/models"):
        """
        patience (int): How long to wait after last time validation loss improved. 
        delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model = None
        self.best_params = {}

        self.path = path
        if not os.path.exists(self.path):
          os.makedirs(self.path)

        
    def __call__(self, val_metric, model, save=False):
        """ 
        The validation metric is passed at each iteration, the class keeps track of the best checkpoint
        """
        score = val_metric

        if self.best_score is None:
            self.best_score = score

        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_metric
            self.counter = 0
 
            self.best_model = copy.deepcopy(model).cpu()
            self.best_params = self.best_model.state_dict()

            if type(save)==str:
              torch.save(self.best_params, os.path.join(self.path, f"{save}.pt"))
              



########################################
## PLOTTING
########################################


def plot_roc(fpr, tpr, roc_auc, title, c, path, name_save):
    fig = plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, color=c, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='lightgray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=14)
    plt.ylabel('True Positive Rate', fontsize=14)
    plt.title('ROC Curve (' + title + ')', fontsize=16)
    plt.legend(loc="lower right", fontsize=11)

    plt.tight_layout()

    if not os.path.exists(path):
        os.makedirs(path)
    fig.savefig(os.path.join(path, name_save + ".jpg"), bbox_inches='tight')
        
    plt.show()
  

def plot_results(history, name_vis, yticks, path, name_save=None):
    
    
    c = ["blue", "red"]
    lab = ["Train", "Validation"]
    y_label = ["Loss", "Accuracy"]

    title_lst = [x + " evolution" for x in y_label] 


    nb_epochs = len(history[0][0])
    # epochs = range(1, nb_epochs+1, 5)
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))

    for id, (ax, data) in enumerate(zip(axs, history)):
        for idx, st in enumerate(data): 
            ax.plot(list(range(1, nb_epochs+1)), st, linewidth=1.4, color=c[idx], label=lab[idx])
            ax.legend()
            
        ax.set_xlabel("Epochs", fontsize=13)
        ax.set_title(title_lst[id], fontsize=15)
        ax.set_ylabel(y_label[id], rotation=90, fontsize=13)
        if id==1:
            ax.yaxis.set_major_formatter(yticks)

    plt.suptitle(name_vis, fontsize=17)

    plt.tight_layout()

    if type(name_save)==str:
        if not os.path.exists(path):
            os.makedirs(path)
        fig.savefig(os.path.join(path, name_save + ".jpg"), bbox_inches='tight')

    plt.show()
    
    
    
def plot_confusion_matrix(cm, path, name_save=None):

  with sns.axes_style("white"):
      fig, ax = plt.subplots(figsize=(5, 5))
      ax = sns.heatmap(cm,square=False, annot=True, fmt=".2f", cmap="YlGnBu", cbar=False)
      ax.set_xlabel('Predicted', fontsize=12)
      ax.set_ylabel('Ground Truth', fontsize=12)

  
  plt.tight_layout()
  if type(name_save)==str:
    if not os.path.exists(path):
            os.makedirs(path)
    fig.savefig(os.path.join(path, name_save + ".jpg"), bbox_inches='tight')
    
  plt.show()
    
    
    
    

########################################
## TRAINING
########################################



def train(model, loss_fcn, optimizer, train_dataloader, val_dataloader, num_epochs, early_stopper, device, save=False, scheduler=None):
    
    model = model.double()
    model.to(device)
    model.train()

    tot_tr_loss, tot_tr_acc = [], []
    tot_val_loss, tot_val_acc = [], []

    for epoch in range(num_epochs):
        train_loss, train_acc = [], []
        for _, (batched_graph,labels)  in enumerate(train_dataloader):
            batched_graph, labels = batched_graph.to(device), labels.to(device)
            logits = model(batched_graph, batched_graph.ndata['node_attr'].double())
            loss = loss_fcn(logits, labels.T[0])
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(compute_accuracy(logits, labels.T[0]))

            if scheduler:
              scheduler.step()

        train_loss, train_acc = np.mean(train_loss), np.mean(train_acc)
        val_loss, val_acc = test(model, loss_fcn, val_dataloader, device)
        early_stopper(val_acc, model, save)

        tot_tr_loss.append(train_loss) ; tot_tr_acc.append(train_acc)
        tot_val_loss.append(val_loss) ; tot_val_acc.append(val_acc)

        if epoch % 10 == 0:         
            print("Epoch {}/{} | Train Loss: {:.4f} | Train Accuracy: {:.4%} | Val Loss: {:.4f} | Val Accuracy {:.4%}".format(
                epoch, num_epochs, train_loss, train_acc, val_loss, val_acc))
            
        if early_stopper.early_stop:
          print("Early Stopping!")
          break
    return tot_tr_loss, tot_tr_acc, tot_val_loss, tot_val_acc
    
            

def test(model, loss_fcn, dataloader, device, cm=False, title=False, c="darkred", path="", name_save=""):
    model = model.double() 
    model.eval()

    loss_scores, acc_scores = [], []
    y_true, y_pred, all_logits = [], [], []

    with torch.no_grad():
      for _, (batched_graph, labels) in enumerate(dataloader):
          test_loss, test_acc, logits = evaluate(model, batched_graph, labels, loss_fcn, device)
          loss_scores.append(test_loss)
          acc_scores.append(test_acc)

          if cm:
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(logits.detach().argmax(axis=1).cpu().numpy().tolist())
            all_logits.extend(logits.detach()[:,1].view(-1).cpu().numpy().tolist())

    loss_mean, acc_mean = np.mean(loss_scores), np.mean(acc_scores)

    if cm:
      confusion_matrix = np.zeros((2, 2), dtype=np.int32)
      for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1

      confusion_matrix = confusion_matrix/len(y_true)

      f_score = f1_score(y_true, y_pred)

      fpr, tpr, _ = roc_curve(y_true, all_logits)
      roc_auc = auc(fpr, tpr)

      print("\nTest Loss: {:.4f} | F1 score: {:.4f} | Test Accuracy {:.4%}\n".format(loss_mean, f_score, acc_mean))      

      plot_roc(fpr, tpr, roc_auc, title, c, path, name_save)
      print("\n")

      return loss_mean, acc_mean, confusion_matrix

    return loss_mean, acc_mean

    

def evaluate(model, batched_graph, labels, loss_fcn, device):
    
    batched_graph, labels = batched_graph.to(device), labels.to(device)
    with torch.no_grad():
        logits = model(batched_graph, batched_graph.ndata['node_attr'].double()).detach()

    loss = loss_fcn(logits, labels.T[0]).item()
    acc = compute_accuracy(logits, labels.T[0])
    return loss, acc, logits




## DENSE


def test_dense(model, loss_fcn, dataloader, device, cm=False, title=False, c="darkred", path="", name_save=""):
    model.to(device)
    model.eval()

    loss_scores, acc_scores = [], []
    y_true, y_pred, all_logits = [], [], []

    with torch.no_grad():
      for data in dataloader:
          data = data.to(device)
          test_loss, test_acc, logits = evaluate_dense(model, data, loss_fcn, device)
          loss_scores.append(test_loss)
          acc_scores.append(test_acc)

          if cm:
            y_true.extend(data.y.view(-1).cpu().numpy().tolist())
            y_pred.extend(logits.detach().argmax(axis=1).cpu().numpy().tolist())
            all_logits.extend(logits.detach()[:,1].view(-1).cpu().numpy().tolist())

    loss_mean, acc_mean = np.mean(loss_scores), np.mean(acc_scores)

    if cm:
        confusion_matrix = np.zeros((2, 2), dtype=np.int32)
        for i in range(len(y_true)):
            confusion_matrix[y_true[i], y_pred[i]] += 1
        
        confusion_matrix = confusion_matrix/len(y_true)

        f_score = f1_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, all_logits)
        roc_auc = auc(fpr, tpr)

        print("\nTest Loss: {:.4f} | F1 score: {:.4f} | Test Accuracy {:.4%}\n".format(loss_mean, f_score, acc_mean))      

        plot_roc(fpr, tpr, roc_auc, title, c, path, name_save)
        print("\n")

        return loss_mean, acc_mean, confusion_matrix

    return loss_mean, acc_mean




def evaluate_dense(model, data, loss_fcn, device):
    
    data.to(device)
    with torch.no_grad():
        logits, _, _ = model(data.x, data.adj, data.mask)

    loss = loss_fcn(logits, data.y.view(-1)).item()
    acc = compute_accuracy(logits, data.y.view(-1))
    return loss, acc, logits



def train_dense(model, loss_fcn, optim, train_loader, val_loader, num_epochs, early_stopper, device, save=False, scheduler=None):
    model.to(device)

    model.train()

    tot_tr_loss, tot_tr_acc = [], []
    tot_val_loss, tot_val_acc = [], []

    for epoch in range(num_epochs):
      train_loss, train_acc = [], []
      for data in train_loader:
          data = data.to(device)
          output, _, _ = model(data.x, data.adj, data.mask)

          loss = loss_fcn(output, data.y.view(-1))

          optim.zero_grad()
          loss.backward()
          optim.step()

          train_loss.append(loss.item())
          train_acc.append(compute_accuracy(output, data.y.view(-1)))

          if scheduler:
            scheduler.step()

      train_loss, train_acc = np.mean(train_loss), np.mean(train_acc)
      val_loss, val_acc = test_dense(model, loss_fcn, val_loader, device)
      early_stopper(val_acc, model, save)

      tot_tr_loss.append(train_loss) ; tot_tr_acc.append(train_acc)
      tot_val_loss.append(val_loss) ; tot_val_acc.append(val_acc)

      if epoch % 10 == 0:         
            print("Epoch {}/{} | Train Loss: {:.4f} | Train Accuracy: {:.4%} | Val Loss: {:.4f} | Val Accuracy {:.4%}".format(
                epoch, num_epochs, train_loss, train_acc, val_loss, val_acc))
            
      if early_stopper.early_stop:
          print("Early Stopping!")
          break

    return tot_tr_loss, tot_tr_acc, tot_val_loss, tot_val_acc



## TORCH


def evaluate_torch(model, data, loss_fcn, device):
    
    data.to(device)
    with torch.no_grad():
        logits = model(data)

    loss = loss_fcn(logits, data.y.view(-1)).item()
    acc = compute_accuracy(logits, data.y.view(-1))
    return loss, acc, logits



def test_torch(model, loss_fcn, dataloader, device, cm=False, title=False, c="darkred", path="", name_save=""):
    model.to(device)
    model.eval()

    loss_scores, acc_scores = [], []
    y_true, y_pred, all_logits = [], [], []

    with torch.no_grad():
      for data in dataloader:
          data = data.to(device)
          test_loss, test_acc, logits = evaluate_torch(model, data, loss_fcn, device)
          loss_scores.append(test_loss)
          acc_scores.append(test_acc)

          if cm:
            y_true.extend(data.y.view(-1).cpu().numpy().tolist())
            y_pred.extend(logits.detach().argmax(axis=1).cpu().numpy().tolist())
            all_logits.extend(logits.detach()[:,1].view(-1).cpu().numpy().tolist())

    loss_mean, acc_mean = np.mean(loss_scores), np.mean(acc_scores)

    if cm:
        confusion_matrix = np.zeros((2, 2), dtype=np.int32)
        for i in range(len(y_true)):
            confusion_matrix[y_true[i], y_pred[i]] += 1
        
        confusion_matrix = confusion_matrix/len(y_true)

        f_score = f1_score(y_true, y_pred)

        fpr, tpr, _ = roc_curve(y_true, all_logits)
        roc_auc = auc(fpr, tpr)

        print("\nTest Loss: {:.4f} | F1 score: {:.4f} | Test Accuracy {:.4%}\n".format(loss_mean, f_score, acc_mean))      

        plot_roc(fpr, tpr, roc_auc, title, c, path, name_save)
        print("\n")

        return loss_mean, acc_mean, confusion_matrix

    return loss_mean, acc_mean



def train_torch(model, loss_fcn, optim, train_loader, val_loader, num_epochs, early_stopper, device, save=False, scheduler=None):
    model.to(device)

    model.train()

    tot_tr_loss, tot_tr_acc = [], []
    tot_val_loss, tot_val_acc = [], []

    for epoch in range(num_epochs):
      train_loss, train_acc = [], []
      for data in train_loader:
          data = data.to(device)
          output = model(data)

          loss = loss_fcn(output, data.y.view(-1))

          optim.zero_grad()
          loss.backward()
          optim.step()

          train_loss.append(loss.item())
          train_acc.append(compute_accuracy(output, data.y.view(-1)))

          if scheduler:
            scheduler.step()

      train_loss, train_acc = np.mean(train_loss), np.mean(train_acc)
      val_loss, val_acc = test_torch(model, loss_fcn, val_loader, device)
      early_stopper(val_acc, model, save)

      tot_tr_loss.append(train_loss) ; tot_tr_acc.append(train_acc)
      tot_val_loss.append(val_loss) ; tot_val_acc.append(val_acc)

      if epoch % 10 == 0:         
            print("Epoch {}/{} | Train Loss: {:.4f} | Train Accuracy: {:.4%} | Val Loss: {:.4f} | Val Accuracy {:.4%}".format(
                epoch, num_epochs, train_loss, train_acc, val_loss, val_acc))
            
      if early_stopper.early_stop:
          print("Early Stopping!")
          break

    return tot_tr_loss, tot_tr_acc, tot_val_loss, tot_val_acc