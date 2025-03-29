import mxnet as mx
from mxnet import autograd
from decode.FracTAL_ResUNet.nn.loss.mtsk_loss import mtsk_loss
from myModel import MyFractalResUNetcmtsk, ReduceLROnPlateau, LossModel
from tqdm import tqdm
from mxnet.lr_scheduler import FactorScheduler

# Global Training Parameters
#epochs = 50
initial_learning_rate  = 0.001
weight_decay = 1e-4  # L2 regularization

num_classes = 2
depth = 6
nfilters_init = 32
early_stopping_patience = 10  # Early stopping patience

class myTrain:
    def __init__(self, train_loader, val_loader):
        """
        Args:
            train_loader (DataLoader): Training DataLoader.
            val_loader (DataLoader): Validation DataLoader.
        """
        self.train_loader = train_loader
        self.val_loader = val_loader

    def train(self, ctx, epochs = 50):
        mx.nd.waitall()
        # Initialize the network, trainer, scheduler, and loss function
        netTrain = MyFractalResUNetcmtsk(False, "", ctx, nfilters_init=nfilters_init, depth=depth, num_classes=num_classes)
        #Trainer with dynamic learning rate
        #lr_scheduler = FactorScheduler(step=10, factor=0.5)  # Reduce LR every 10 epochs
        #, 'lr_scheduler': lr_scheduler
        trainer = mx.gluon.Trainer(
            netTrain.net.collect_params(), 'adam', 
            {'learning_rate': initial_learning_rate, 'wd': weight_decay}
        )
        reduce_lr = ReduceLROnPlateau(trainer, patience=3, factor=0.5, min_lr=1e-6) # Learning rate scheduler
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

        print('Start training now...')
        for epoch in range(epochs):  # Loop for the number of epochs
            final_epoch = epoch

            current_lr = trainer.learning_rate
            print(f"Epoch {epoch}: Current Learning Rate = {current_lr}")
         
            train_loss = 0.0  # Track training loss

            # Training loop
            for batch_img, batch_mask in tqdm(self.train_loader, desc=f"Training Epoch {epoch}"):
                batch_img = batch_img.as_in_context(ctx)
                batch_mask = batch_mask.as_in_context(ctx)
                    
                with autograd.record():
                    # Forward pass
                    ListOfPredictions = netTrain.net(batch_img)
                    #[pred_segm, pred_bound,pred_dists]
                    loss = myMTSKL.loss(ListOfPredictions, batch_mask)

                loss.backward()  # Backward pass
                trainer.step(batch_img.shape[0])  # Update parameters (batch size scaling)

                # Compute batch loss
                train_loss += loss.mean().asscalar()

            # Average training loss for the epoch
            current_epoch_loss = train_loss / len(self.train_loader)
            train_losses.append(current_epoch_loss)
            print(f"Training Loss: {current_epoch_loss}")

            # Validate every 5 epochs
            #if epoch % 5 == 0 or epoch == epochs - 1:

            # Validation loop
            val_loss = 0.0
            for batch_img, batch_mask in tqdm(self.val_loader, desc=f"Validation Epoch {epoch}"):
                batch_img = batch_img.as_in_context(ctx)
                batch_mask = batch_mask.as_in_context(ctx)

                # Forward pass only
                ListOfPredictions = netTrain.net(batch_img)
                # Apply sigmoid activation to validation predictions
                pred_segm = mx.nd.sigmoid(ListOfPredictions[0])
                pred_bound = mx.nd.sigmoid(ListOfPredictions[1])
                pred_dists = mx.nd.sigmoid(ListOfPredictions[2])

                # Compute validation loss
                #[pred_segm, pred_bound,pred_dists]
                loss = myMTSKL.loss(ListOfPredictions, batch_mask)
                val_loss += loss.mean().asscalar()

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

