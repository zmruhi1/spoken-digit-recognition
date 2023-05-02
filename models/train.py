from utilities.utils import import EarlyStopping, shared_step
import tqdm 




def train_CL(model, train_dl, val_dl, criterion, optimizer, epochs=5, early_stop=True, patience=5, checkpoint_name="checkpoint.pt", rnn=False):
    """
    Train the model on the given data

    param model: the instance of Pytorch NN model
    param train_dl: the training dataloader to be trained on
    param val_dl: the validation dataloader for evaluating (in training and in early stopping)
    param criterion: the loss function
    param optimizer: the optimizer function
    param epochs: the number of epochs to train the model for
    param early_stop: If the early stopping should be used or not
    param patience: (for early stopping) the number of times to wait before stopping training
    param checkpoint_name: (for early stopping) the name of the model checkpoint
    """
    losses_to_plot = []
    validation_accuracy = []

    # initialize the earlystopping module
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=checkpoint_name)

    for epoch in range(epochs):
        epoch_loss = []
        #initialize pairs to calcuclate CL losses
        pair = {}
        for i, (inp_wave, inp_len, aug_wave, aug_len, label) in enumerate(pbar := tqdm(train_dl)):  
          
            inp_wave = inp_wave.squeeze(0).to(device)
            inp_len = inp_len.to(device)
            label = label.to(device)
            aug_wave = aug_wave.squeeze(0).to(device)
            aug_len = aug_len.to(device)
            
            # Forward pass
            # for conformer, send input wave and input length
            out1 = model(inp_wave, inp_len, return_logits=True)
            out2 = model(aug_wave, aug_len, return_logits=True)

            #storing output pairs 
            # if len(pair) == 0: 
            #   z1 = inp_wave
            #   out1 = model(inp_wave, inp_len)
            #   pair[z1] = [out1, label]

            # else:
            #   z2 = inp_wave
            #   out2 = model(inp_wave, inp_len)
            #   pair[z2] = [out2, label]

            #   assert len(pair)==2

            loss = shared_step(y_hat1=out1, y_hat2=out2, y=label)

            # if loss == -1: #skips propagating loss if labels are not same
            #   continue

            # Save the step loss for plotting
            losses_to_plot.append(loss.item())

            # Save the step loss for calculating mean error in epoch
            epoch_loss.append(loss.item())
            
            # Backpropagation and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # add loss value to the tqdm progress bar
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

            #reset for next pair 
            pair = {}


            # add loss value to the tqdm progress bar
            pbar.set_postfix_str(f"Loss: {loss.item():.4f}")

        epoch_loss_mean = sum(epoch_loss)/len(epoch_loss)

        # Evaluate on validation set
        preds, labs = evaluate(model, val_dl, rnn=rnn)
        accuracy = accuracy_score(preds, labs)
        validation_accuracy.append(accuracy)

        if early_stop:
          # if early stopping is enabled, store the validation scores
          early_stopping(accuracy, model)

          if early_stopping.early_stop:
            # if the patience has been crossed, the training is stopped and
            # the checkpoint file contains the best model according to the
            # validation accuracy
            print("Stopping early at epoch [{}/{}]".format(epoch+1, epochs))
            return losses_to_plot, validation_accuracy

        print("Epoch [{}/{}] Loss: {:.4f} Val. Acc: {:.4f}".format(epoch+1, epochs, epoch_loss_mean, accuracy))
          
    return losses_to_plot, validation_accuracy