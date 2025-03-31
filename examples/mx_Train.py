import random
import numpy as np
import mxnet as mx,autograd
from decode.FracTAL_ResUNet.nn.loss.mtsk_loss import mtsk_loss
from myModel import MyFractalResUNetcmtsk, ReduceLROnPlateau, LossModel
from tqdm import tqdm
from image_Mixup_CutMix import cutmix_data, cutmix_loss, mixup_data, mixup_loss


# Global Training Parameters
#epochs = 50
initial_learning_rate  = 0.001
weight_decay = 1e-4  # L2 regularization

num_classes = 2
depth = 6
nfilters_init = 32
early_stopping_patience = 10  # Early stopping patience

class myTrain:
    def __init__(self, train_loader, val_loader, config):
        """
         Handles the full training loop for FracTAL-ResUNet with multi-task loss.
            - Trains and validates the model over multiple epochs
            - Uses learning rate scheduler (ReduceLROnPlateau)
            - Applies early stopping based on validation loss
            - Tracks loss per epoch and saves model checkpoints
        Args:
            train_loader (DataLoader): Training DataLoader.
            val_loader (DataLoader): Validation DataLoader.

        Returns:
            loss_each_epoch (list): Epoch-wise training and validation loss
            model_list (list): Saved model objects with metadata
            final_epoch (int): Last completed epoch index
        """
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        
    def get_training_loss(self, epoch, ctx, netTrain, myMTSKL, trainer):
        # Training loop
        train_loss = 0.0  # Track training loss
        for batch_img, batch_mask in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
            batch_img = batch_img.as_in_context(ctx)
            batch_mask = batch_mask.as_in_context(ctx)
             
            if self.config.use_mixup_cutmix:
                loss = self.get_training_mixup_cutmix_loss(batch_img, batch_mask, netTrain, myMTSKL) # Apply Mixup or CutMix
            else:         
                with autograd.record():
                    ListOfPredictions = netTrain.net(batch_img)   # Forward pass
                    loss = myMTSKL.loss(ListOfPredictions, batch_mask)

                loss.backward()  # Backward pass
                trainer.step(batch_img.shape[0])  # Update parameters (batch size scaling)

                train_loss += loss.mean().asscalar()  # Compute batch loss
        return train_loss
    
    def get_training_mixup_cutmix_loss(self,  batch_img, batch_mask, netTrain, myMTSKL):
        if np.random.rand() < 0.5:
            imgs, labels_a, labels_b, lam = mixup_data(batch_img, batch_mask) # Apply Mixup: Mixup interpolates two images and their labels, improving generalization.
            with autograd.record():
                ListOfPredictions = netTrain.net(imgs) # Forward pass
                loss = mixup_loss(myMTSKL.loss, ListOfPredictions, labels_a, labels_b, lam)    
        else:
            imgs, labels_a, labels_b, lam = cutmix_data(batch_img, batch_mask) # Apply Mixup: CutMix replaces a portion of one image with another, forcing the model to focus on multiple parts.
            with autograd.record():
                ListOfPredictions = netTrain.net(imgs) # Forward pass
                loss = cutmix_loss(myMTSKL.loss, ListOfPredictions, labels_a, labels_b, lam) 
        return loss
        
    def get_val_loss(self, epoch, ctx, netTrain, myMTSKL):
        # Validation loop
        val_loss = 0.0
        for batch_img, batch_mask in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
            batch_img = batch_img.as_in_context(ctx)
            batch_mask = batch_mask.as_in_context(ctx)
            ListOfPredictions = netTrain.net(batch_img)
            loss = myMTSKL.loss(ListOfPredictions, batch_mask)
            val_loss += loss.mean().asscalar()
        return val_loss

    def train(self, ctx, epochs = 50):
        mx.nd.waitall()
        netTrain = MyFractalResUNetcmtsk(False, "", ctx, nfilters_init=nfilters_init, depth=depth, num_classes=num_classes) # Initialize the network, trainer, scheduler, and loss function
        myMTSKL = mtsk_loss(depth=0, NClasses=num_classes)  # Loss function for multitasking operation
         # Track losses and models
        train_losses = []
        val_losses = []
        loss_each_epoch = []
        model_list = []

        # Early stopping variables
        best_val_loss = float('inf')
        patience_counter = 0
        final_epoch = 0
        
        if self.config.use_hyperparameter_tuning:
            learning_rates = [0.001, 0.0005, 0.0001]
            weight_decays  = [1e-4, 1e-5, 1e-6]
        else:
            trainer = mx.gluon.Trainer(netTrain.net.collect_params(), 'adam', {'learning_rate': initial_learning_rate, 'wd': weight_decay})
            reduce_lr = ReduceLROnPlateau(trainer, patience=3, factor=0.5, min_lr=1e-6) # Learning rate scheduler
            
        print('Start training now...')
        for epoch in range(epochs):  # Loop for the number of epochs
            if self.config.use_hyperparameter_tuning:
                lr = random.choice(learning_rates)
                wd = random.choice(weight_decays)
                print(f"Testing: lr={lr}, wd={wd}")
                trainer = mx.gluon.Trainer(netTrain.net.collect_params(), 'adam', {'learning_rate': lr, 'wd': wd})
                reduce_lr = ReduceLROnPlateau(trainer, patience=3, factor=0.5, min_lr=1e-6) # Learning rate scheduler
            
            final_epoch = epoch
            current_lr = trainer.learning_rate
            print(f"Epoch {epoch}: Current Learning Rate = {current_lr}")
         
            train_loss = self.get_training_loss(epoch, ctx, netTrain, myMTSKL, trainer, train_loss)
            
            # Average training loss for the epoch
            current_epoch_loss = train_loss / len(self.train_loader)
            train_losses.append(current_epoch_loss)
            print(f"Training Loss: {current_epoch_loss}")

            val_loss = self.get_val_loss(epoch, ctx, netTrain, myMTSKL)
            # Average validation loss for the epoch
            current_epoch_val_loss = val_loss / len(self.val_loader)
            val_losses.append(current_epoch_val_loss)
            print(f"Validation Loss: {current_epoch_val_loss}")

            # Early stopping check
            if current_epoch_val_loss < best_val_loss- 1e-4:
                    best_val_loss = current_epoch_val_loss
                    patience_counter = 0  # Reset patience counter
            else:
                patience_counter += 1
                if patience_counter >= early_stopping_patience:
                    print("Early stopping triggered")
                    break
            # Adjust learning rate
            reduce_lr.step(current_epoch_val_loss)

            loss_each_epoch.append({"Current Epoch": epoch, "Training Loss": current_epoch_loss, "Validation Loss": current_epoch_val_loss})
            current_model = LossModel(epoch, current_epoch_loss, current_epoch_val_loss, netTrain.net)
            model_list.append(current_model)
            print('****************************************************************')
                   
        return loss_each_epoch, model_list, final_epoch

