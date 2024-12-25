from mxnet.lr_scheduler import LRScheduler
from decode.FracTAL_ResUNet.models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk

#Default train params and initialize FracTAL_ResUNet_cmtsk
class MyFractalResUNetcmtsk():
    def __init__(self, isLoad, path, ctx, nfilters_init=32, depth =6, num_classes= 2):
        self.net = FracTAL_ResUNet_cmtsk(nfilters_init, depth, num_classes)
        self.net.initialize(ctx=ctx)
        if isLoad:
            self.net.load_parameters(path, ctx=ctx)
        self.net.hybridize()
        
# Example usage inside a training loop
class ReduceLROnPlateau(LRScheduler):
    def __init__(self, trainer, patience=5, factor=0.1, min_lr=1e-6, mode='min', threshold=1e-4, cooldown=0):
        super(ReduceLROnPlateau, self).__init__()
        self.trainer = trainer
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.mode = mode
        self.threshold = threshold
        self.cooldown = cooldown
        self.best = None
        self.num_bad_epochs = 0
        self.cooldown_counter = 0
    
    def step(self, metric):
        # Cooldown logic to delay LR changes
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            return

        # Set the best metric and determine if LR needs to be reduced
        if self.best is None or \
           (self.mode == 'min' and metric < self.best - self.threshold) or \
           (self.mode == 'max' and metric > self.best + self.threshold):
            self.best = metric
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        # Reduce LR if we exceed patience
        if self.num_bad_epochs >= self.patience:
            new_lr = max(self.trainer.learning_rate * self.factor, self.min_lr)
            self.trainer.set_learning_rate(new_lr)
            print(f'Reducing learning rate to {new_lr}')
            self.num_bad_epochs = 0
            self.cooldown_counter = self.cooldown


class LossModel():
    def __init__(self, epoch, loss, val_loss, net):
        self.epoch = epoch
        self.loss = loss
        self.val_loss = val_loss
        self.net = net