import os
import csv
import mxnet as mx
from mxnet import autograd, nd
from decode.FracTAL_ResUNet.nn.loss.mtsk_loss import mtsk_loss
from decode.postprocessing.instance_segmentation import InstSegm
from myModel import MyFractalResUNetcmtsk, ReduceLROnPlateau, LossModel


epochs = 50
learning_rate = 0.001
num_classes = 2
depth = 6
nfilters_init = 32

loss_path_plot = r"D:\Source\Output\Loss\loss_VNIR_plot.png"
loss_path_csv = r"D:\Source\Output\Loss\loss_VNIR.csv"
trained_model_path = r"D:\Source\Output\Models\Model_VNIR_params"

class myTrain:
    def __init__(self, images, masks, image_dict, mask_dict, train_ids, val_ids):
        self.images= images
        self.masks= masks
        self.image_dict = image_dict
        self.mask_dict = mask_dict
        self.train_ids = train_ids
        self.val_ids = val_ids
           
    def train(self, ctx): #Track epoch losses, backward with multitasking operation
        mx.nd.waitall()
        netTrain = MyFractalResUNetcmtsk(False, "", ctx, nfilters_init= nfilters_init, depth= depth, num_classes= num_classes)
        trainer = mx.gluon.trainer.Trainer(netTrain.net.collect_params(), 'adam', {'learning_rate': learning_rate})
        reduce_lr = ReduceLROnPlateau(trainer, patience=5, factor=0.1) # Example usage inside a training loop
        myMTSKL = mtsk_loss(depth=0, NClasses=num_classes) # get default Tanimoto loss (depth=0) function for multitasking operation
        
        train_losses = []
        val_losses = []
        loss_each_epoch = []
        model_list = []

        print('Start training now')
        for epoch in range(epochs): # Train for as many num_epochs you want
            print(f'current epoch: {epoch}')
            train_loss = 0.0     # compute training loss
            for id in self.train_ids: # NOTE: img/mask contain batches of images/lables each with the size (batch_size, H, W)
                print(f'image id:{id}')
                img = self.images.getImage(id, self.image_dict, ctx)
                mask = self.masks.getImage(id, self.mask_dict, ctx)
                if img is None or mask is None:
                    continue
                print(f"Image shape: {img.shape}, Mask shape: {mask.shape}")
                # forward + backward
                with autograd.record():
                    ListOfPredictions = netTrain.net(img)
                    print(f"ListOfPredictions shape: {[pred.shape for pred in ListOfPredictions]}")
                    loss = myMTSKL.loss(ListOfPredictions, mask)
                    print(f"Loss before backward: {loss.asscalar()}")
                loss.backward()
                trainer.step(1)
                #using one image so it does not need step trainer.step(1) # update parameters
                print(f'epoch {epoch} currentloss: {loss.asscalar()}')
                train_loss += loss.mean().asscalar() # calculate training metrics
                print(f'epoch {epoch} train_loss: {train_loss}')
            current_epoch_loss = train_loss / len(self.train_ids)  # compute overall loss
            print(f'epoch {epoch} current_epoch_loss: {current_epoch_loss}')
            train_losses.append(current_epoch_loss)    # now append the epoch loss
            print('--------------------------------------------------------------')
    
            # compute validation loss
            val_loss = 0.0
            for id in self.val_ids:           
                img = self.images.getImage(id, self.image_dict, ctx)
                mask = self.masks.getImage(id, self.mask_dict, ctx)
                if img is None or mask is None:
                    continue
                ListOfPredictions = netTrain.net(img)  # forward only
                loss = myMTSKL.loss(ListOfPredictions, mask)  # get loss
                print(f'epoch {epoch} validation currentloss: {loss.asscalar()}')
                val_loss += loss.mean().asscalar() # calculate validation metrics
                print(f'epoch {epoch} validation val_loss: {val_loss}')

            # compute overall loss
            current_epoch_val_loss = val_loss/ len(self.val_ids)
            print(f'epoch {epoch} current_epoch_val_loss: {current_epoch_val_loss}')
            val_losses.append(current_epoch_val_loss)

            
            reduce_lr.step(current_epoch_val_loss)  # Adjust learning rate if validation loss stagnates
    
            # Track loss and model for analysis    
            loss_each_epoch.append({"Current Epoch": epoch, "Traing Loss": current_epoch_loss, "Validation loss": current_epoch_val_loss})
            current_model = LossModel(epoch, current_epoch_loss, current_epoch_val_loss, netTrain.net)
            model_list.append(current_model)
            print('****************************************************************')
        
        return loss_each_epoch, model_list
    
    def saveLosses(self, loss_each_epoch): #Write each epoch loss in CSV
        headers = ["Current Epoch", "Traing Loss", "Validation loss"]
        with open(r"D:\Source\Output\Loss\loss_VNIR.csv", mode='w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=headers)
            writer.writeheader()
            writer.writerows(loss_each_epoch)
        print(f"Data saved to {loss_path_csv}")
    
    def SaveModels(self, model_list):
        for model in model_list:
            path =  os.path.join(trained_model_path, f"model_VNIR_{model.epoch}.params")
            model.net.save_parameters(path)
    
        