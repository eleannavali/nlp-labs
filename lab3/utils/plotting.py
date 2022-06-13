import matplotlib.pyplot as plt 

def plot_training_curves(tr_loss, tr_acc, val_loss, val_acc,name,model,concat,bidirectional):
    '''Plot training learning curves for both train and validation.'''

    #Range for the X axis.
    epochs = range(len(tr_loss))

    #Plotting Loss figures.
    plt.figure(figsize=(12,10)) #figure size h,w in inches
    plt.rcParams.update({'font.size': 22}) #configuring font size.
    plt.plot(epochs,tr_loss,c="red",label="Training Loss") #plotting
    plt.plot(epochs,val_loss,c="blue",label="Validation Loss")
    plt.xlabel("Epochs") #title for x axis
    plt.ylabel("Loss")   #title for y axis
    plt.legend(fontsize=11)
    plt.savefig('diagrams/loss_'+name+'_'+model+"_concat="+str(concat)+"_bidirectional="+str(bidirectional)+'.png')

    #Plotting Accuracy figures.
    plt.figure(figsize=(12,10)) #figure size h,w in inches
    plt.plot(epochs,tr_acc,c="red",label="Training Acc") #plotting
    plt.plot(epochs,val_acc,c="blue",label="Validation Acc")
    plt.xlabel("Epochs")   #title for x axis
    plt.ylabel("Accuracy") #title for y axis
    plt.legend(fontsize=11)
    plt.savefig('diagrams/accuracy_'+name+'_'+model+"_concat="+str(concat)+"_bidirectional="+str(bidirectional)+'.png')

# if __name__ == "__main__":
#     a = [1,2,3,4]
#     b = [0.3,1.2,1.6,2.1]
#     c = [1,2,3,4]
#     d = [1.3,4.2,5.6,6.1]
#     plot_training_curves(a,c,b,c,d)