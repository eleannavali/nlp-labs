import math
import sys

import torch
from sklearn.metrics import f1_score, recall_score 
#  ERROR on sklearn metrices: Classification metrics can't handle a mix of multilabel-indicator and continuous-multioutput targets

def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_dataset(_epoch, dataloader, model, loss_function, optimizer):
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0
    running_acc = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device
    for index, batch in enumerate(dataloader, 1):
        # get the inputs (batch)
        inputs, labels, lengths = batch

        # move the batch tensors to the right device
        inputs.to(device) # EX9
        labels.to(device)
        lengths.to(device)

        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        optimizer.zero_grad()  # EX9

        # Step 2 - forward pass: y' = model(x)
        pred = model(inputs, lengths) # EX9

        # Step 3 - compute loss: L = loss_function(y', y)
        loss = loss_function(pred,labels)  # EX9

        # Step 4 - backward pass: compute gradient wrt model parameters
        loss.backward()

        # Step 5 - update weights
        optimizer.step()

        running_loss += loss.data.item()
 

        # print statistics
        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))

    return running_loss / index


def eval_dataset(dataloader, model, loss_function, binary_classification=True):
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0
    num_of_corrects = 0
    num_of_samples = 0
    running_f1 = 0.0
    running_recall = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # get the inputs (batch)
            inputs, labels, lengths = batch

            # Step 1 - move the batch tensors to the right device
            inputs.to(device) # EX9
            labels.to(device)
            lengths.to(device)  

            # Step 2 - forward pass: y' = model(x)
            pred = model(inputs, lengths) # EX9  # EX9

            # Step 3 - compute loss.
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time
            loss = loss_function(pred,labels) # EX9

            # Step 4 - make predictions (class = argmax of posteriors)
            class_pred = torch.argmax(pred, dim=1)  # EX9

            # Step 5 - collect the predictions, gold labels and batch loss
            y_pred.append(class_pred)  # EX9
            y.append(labels)

            # compute batch accuracy
            running_loss += loss.data.item()
            num_of_samples += len(inputs)
            if binary_classification:
                num_of_corrects += torch.sum(class_pred == torch.argmax(labels,dim=1))
                running_f1 += f1_score(torch.argmax(labels,dim=1),class_pred,average='macro')
                running_recall += recall_score(torch.argmax(labels,dim=1),class_pred,average='macro')
            else:
                # print(labels.size(),class_pred.size())
                num_of_corrects += torch.sum(class_pred == labels)
                running_f1 += f1_score(labels,class_pred,average='macro')
                running_recall += recall_score(labels,class_pred,average='macro')
        # print("accuracy: ",num_of_corrects,num_of_samples,num_of_corrects.item()/num_of_samples)

    return running_loss / index, (y_pred, y), num_of_corrects.item() / num_of_samples, running_f1 / index , running_recall / index
