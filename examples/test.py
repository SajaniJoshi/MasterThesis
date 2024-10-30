import os
import rasterio
from mxnet.gluon import nn
os.environ["MXNET_CUDNN_LIB_CHECKING"] = "0"
os.environ["MXNET_CUDNN_AUTOTUNE_DEFAULT"] = "0"

import sys
sys.path.append(r'D:\ImageSeg')

import numpy as np
import mxnet as mx
from mxnet import autograd, gluon
#from resuneta.models.resunet_d7_causal_mtskcolor_ddist import ResUNet_d7
from decode.FracTAL_ResUNet.models.semanticsegmentation.FracTAL_ResUNet import FracTAL_ResUNet_cmtsk
from decode.FracTAL_ResUNet.nn.loss.mtsk_loss import mtsk_loss

from sklearn.model_selection import train_test_split


# Paths
image_dir = r"D:\Source\Test\TextMxnet\data\2022\BB\InputFeatures"
mask_dir = r"D:\Source\Test\TextMxnet\data\2022\BB\InputMask"

# Load satellite images and masks
def load_tif_files(directory):
    image_list = []
    for filename in os.listdir(directory):
        if filename.endswith(".tif"):
            with rasterio.open(os.path.join(directory, filename)) as src:
                image = src.read()  # Load as NumPy array
                image_list.append(image)
    return image_list

# Load the images and masks
images = load_tif_files(image_dir)
masks = load_tif_files(mask_dir)

# Normalize images and preprocess masks (assuming binary segmentation for agricultural fields)
def preprocess_data(images, masks):
    images_preprocessed = []
    masks_preprocessed = []
    for img, msk in zip(images, masks):
        img = img / 255.0  # Normalize image (assuming pixel values are 0-255)
        msk = np.where(msk > 0, 1, 0)  # Convert mask to binary (if needed)
        images_preprocessed.append(img)
        masks_preprocessed.append(msk)
    return np.array(images_preprocessed), np.array(masks_preprocessed)


# Preprocess the data
X, Y = preprocess_data(images, masks)

class ResUNetA(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ResUNetA, self).__init__(**kwargs)
        with self.name_scope():
            self.encoder = nn.HybridSequential()
            # Define your ResUNetA encoder layers
            self.decoder = nn.HybridSequential()
            # Define your ResUNetA decoder layers

            self.output = nn.Conv2D(1, kernel_size=1)  # For binary segmentation output

    def hybrid_forward(self, F, x):
        # Define your forward pass using the layers
        x_enc = self.encoder(x)
        x_dec = self.decoder(x_enc)
        return self.output(x_dec)

# Initialize the model
ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
model = ResUNetA()
model.initialize(ctx=ctx)


# Define loss function and trainer
loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})

# DataLoader to batch data
batch_size = 16
train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X, Y), batch_size=batch_size, shuffle=True)

# Training loop
epochs = 50
for epoch in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        
        with autograd.record():
            output = model(data)
            loss = loss_fn(output, label)
        loss.backward()
        trainer.step(batch_size)

    print(f"Epoch [{epoch+1}/{epochs}], Loss: {nd.mean(loss).asscalar():.4f}")
    
    
    
    
    # Load 2010 satellite images (in the same way as 2022 images)
image_dir_2010 = r"D:\Source\Test\TextMxnet\data\2010\BB\08X_Features_Multi"
images_2010 = load_tif_files(image_dir_2010)

# Preprocess 2010 images
X_2010, _ = preprocess_data(images_2010, images_2010)  # No masks for testing, so set the second param to images

# Predict agricultural fields in 2010
for image in X_2010:
    image = nd.array(image).as_in_context(ctx)
    pred_mask = model(image.expand_dims(0))  # Add batch dimension
    pred_mask = nd.sigmoid(pred_mask)  # Apply sigmoid for binary classification
    pred_mask = (pred_mask > 0.5).astype(np.uint8)  # Threshold for binary output
    
    # Visualize or save the predicted mask
    plt.imshow(pred_mask.squeeze().asnumpy(), cmap='gray')
    plt.show()



print(images)
print(masks)


