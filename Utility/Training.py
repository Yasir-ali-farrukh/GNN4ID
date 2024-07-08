import seaborn as sns
import torch
from tqdm import tqdm

import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score, precision_score, recall_score, roc_auc_score
import numpy as np


def train(train_loader, model, args, device="cuda"):
    """
    Trains the model using the provided DataLoader, optimizer, and learning rate scheduler.

    Args:
        model (torch.nn.Module): The model to be trained.
        train_loader (DataLoader): DataLoader for the training data.
        args (dict): Dictionary containing training arguments like learning rate and epochs.
        device (str): The device to run the training on (default is "cuda").

    Returns:
        None
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr']) #weight_decay=5e-4)
    ## Dynamic LR (You can run it without this scheduler too)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.01, min_lr=0.00001)
    best_model = None
    max_val = -1
    for epoch in range(args['epochs']):
        total_loss = 0
        model.train()
        num_graphs = 0
        for batch in tqdm(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x_dict,batch.edge_index_dict,batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs
        train_acc = test(train_loader,model, device)
        scheduler.step(train_acc)
        current_lr = optimizer.param_groups[0]['lr']
        log = "Epoch {}: Train: {:.4f}, Loss: {:.4f}, Lr: {:.6f}"
        print(log.format(epoch + 1, train_acc, total_loss,current_lr))

def test(loader, model, device='cuda'):
    """
    Evaluates the model on the provided DataLoader and calculates accuracy.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (DataLoader): DataLoader for the evaluation data.
        device (str): The device to run the evaluation on (default is "cuda").

    Returns:
        float: The accuracy of the model on the provided data.
    """
    model.eval()
    correct = 0
    num_graphs = 0

    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict,batch.edge_index_dict,batch).max(dim=1)[1]
            label = batch.y
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
    return correct / num_graphs
    
 
def test_cm(loader,model, device='cuda'):
    """
    Evaluates the model on the provided DataLoader, calculates accuracy, and generates predictions and labels.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        loader (DataLoader): DataLoader for the evaluation data.
        device (str): The device to run the evaluation on (default is "cuda").

    Returns:
        tuple: The accuracy of the model, predicted labels, and true labels.
    """
    model.eval()
    correct = 0
    num_graphs = 0
    all_preds = []
    all_labels = []
    for batch in (tqdm(loader)):
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict,batch.edge_index_dict,batch).max(dim=1)[1]
            label = batch.y
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())
        
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
        
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels)

    return correct / num_graphs, all_preds, all_labels

def calculate_metrics(y_pred, y_true):
    """
    Calculates and prints the confusion matrix and accuracy for the given predictions and true labels.

    Args:
        y_pred (np.ndarray): The predicted labels.
        y_true (np.ndarray): The true labels.

    Returns:
        None
    """
    print(f"\n Confusion matrix: \n {confusion_matrix(y_pred, y_true)}")
    print(f"Accuracy: {accuracy_score(y_true, y_pred)}")
    
    
 ############################################################################################################################
 
def train_with_edge_Att(train_loader,model, args, device="cuda"):
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr']) #weight_decay=5e-4)
    ## Dynamic LR
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, threshold=0.01, min_lr=0.00001)
    
    best_model = None
    max_val = -1
    for epoch in range(args['epochs']):
        total_loss = 0
        model.train()
        num_graphs = 0
        for batch in tqdm(train_loader):
            batch.to(device)
            optimizer.zero_grad()
            pred = model(batch.x_dict,batch.edge_index_dict, batch.edge_attr_dict, batch)
            label = batch.y
            loss = model.loss(pred, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            num_graphs += batch.num_graphs
        total_loss /= num_graphs
        train_acc = test_edge(train_loader,model, device)
        # scheduler.step(train_acc)
        current_lr = optimizer.param_groups[0]['lr']
        log = "Epoch {}: Train: {:.4f}, Loss: {:.4f}, Lr: {:.6f}"
        print(log.format(epoch + 1, train_acc, total_loss,current_lr))

def test_edge(loader,model, device='cuda'):
    model.eval()
    correct = 0
    num_graphs = 0

    for batch in loader:
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict,batch.edge_index_dict, batch.edge_attr_dict, batch).max(dim=1)[1]
            label = batch.y
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
    return correct / num_graphs
    
 
def test_cm_with_edge_att(loader,model, device='cuda'):
    model.eval()
    correct = 0
    num_graphs = 0
    all_preds = []
    all_labels = []
    for batch in (tqdm(loader)):
        batch.to(device)
        with torch.no_grad():
            pred = model(batch.x_dict,batch.edge_index_dict, batch.edge_attr_dict, batch).max(dim=1)[1]
            label = batch.y
        all_preds.append(pred.cpu().detach().numpy())
        all_labels.append(label.cpu().detach().numpy())
        
        correct += pred.eq(label).sum().item()
        num_graphs += batch.num_graphs
        
    all_preds = np.concatenate(all_preds).ravel()
    all_labels = np.concatenate(all_labels).ravel()
    calculate_metrics(all_preds, all_labels)

    return correct / num_graphs, all_preds, all_labels

