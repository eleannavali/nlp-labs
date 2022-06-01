import matplotlib.pyplot as plt 

def plot_training_curves(tr_loss, tr_acc, val_loss, val_acc):
    '''Plot training learning curves for both train and validation.'''

    #Range for the X axis.
    epochs = range(len(tr_loss))

    #Plotting Loss figures.
    fig = plt.figure(figsize=(12,10)) #figure size h,w in inches
    plt.rcParams.update({'font.size': 22}) #configuring font size.
    plt.plot(epochs,tr_loss,c="red",label="Training Loss") #plotting
    plt.plot(epochs,val_loss,c="blue",label="Validation Loss")
    plt.xlabel("Epochs") #title for x axis
    plt.ylabel("Loss")   #title for y axis
    plt.legend(fontsize=11)

    #Plotting Accuracy figures.
    fig = plt.figure(figsize=(12,10)) #figure size h,w in inches
    plt.plot(epochs,tr_acc,c="red",label="Training Acc") #plotting
    plt.plot(epochs,val_acc,c="blue",label="Validation Acc")
    plt.xlabel("Epochs")   #title for x axis
    plt.ylabel("Accuracy") #title for y axis
    plt.legend(fontsize=11)