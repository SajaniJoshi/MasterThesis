import os
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import mxnet as mx
from mxnet import gluon, nd, autograd
from mxnet.gluon import nn
from sklearn.model_selection import train_test_split

# Paths
image_dir = r"D:\Source\Test\TextMxnet\data\2022\BB\08X_Features_Multi"
mask_dir = r"D:\Source\Test\TextMxnet\data\2022\BB\XX_Reference_Masks_ResUNetA"
loss_path = r"D:\Source\Test\TextMxnet\data\Loss"

# Load TIFF files
def load_tif_files(directory):
    image_list = []
    try:
        for filename in os.listdir(directory):
            if filename.endswith(".tif"):
                with rasterio.open(os.path.join(directory, filename)) as src:
                    image = src.read()  # Load as NumPy array
                    image_list.append(image)
    except Exception as e:
        print(f"Error loading TIFF files: {e}")
    return image_list

# Preprocessing
def preprocess_data(images, masks):
    images_preprocessed = []
    masks_preprocessed = []
    try:
        for img, msk in zip(images, masks):
            img = img / 255.0  # Normalize image (0-255 scale)
            if msk is None:
                print(f"msk is none.")
            else:
                print(f"msk is empty or None.{msk}")

            # Handle invalid values (NaN or inf) in the mask
            msk = np.nan_to_num(msk)  # Replace NaN, inf, -inf with 0
            #msk = np.where(msk > 0, 1, 0)  # Convert mask to binary
            # Assuming binary segmentation, reduce multi-channel label to a single channel
            if msk.shape[0] > 1:
                # Sum across the channels or select a specific channel based on your data
                msk = np.sum(msk, axis=0, keepdims=True)  # Sum across channels to create a binary mask
            # Convert sum to binary mask (0 or 1)
            msk = np.where(msk > 0, 1, 0)

            images_preprocessed.append(img.astype('float32'))  # Convert images to float32
            masks_preprocessed.append(msk.astype('float32'))  # Convert masks to float32
    except Exception as e:
        print(f"Error during preprocessing: {e}")
    return np.array(images_preprocessed), np.array(masks_preprocessed)

# Load and preprocess data
try:
    images = load_tif_files(image_dir)
    masks = load_tif_files(mask_dir)
    X, Y = preprocess_data(images, masks)
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {Y.shape}")
except Exception as e:
    print(f"Error loading or preprocessing data: {e}")

# Check for NaN or Inf values in the training and validation sets
print(f"X_train min: {np.min(X)}, max: {np.max(X)}")
print(f"Y_train min: {np.min(Y)}, max: {np.max(Y)}")

# Split the data (80% training, 20% validation)
try:
    if X.shape[0] > 0:
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_val)}")
        print(f"Training samples: {len(Y_train)}, Test samples: {len(Y_val)}")
        assert X_train is not None, "X_train is not defined!"
        assert Y_train is not None, "y_train is not defined!"
        assert len(X_train) > 0, "Training set is empty!"
        assert len(X_val) > 0, "Test set is empty!"
        print(f"X_train min: {np.min(X_train)}, max: {np.max(X_train)}, NaN count: {np.isnan(X_train).sum()}")
        print(f"Y_train min: {np.min(Y_train)}, max: {np.max(Y_train)}, NaN count: {np.isnan(Y_train).sum()}")

    else:
        print("Error: No samples available in X or y.")

except Exception as e:
    print(f"Error splitting the data: {e}")

# Define the ResUNetA model
class ResUNetA(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(ResUNetA, self).__init__(**kwargs)
        with self.name_scope():
            self.encoder = nn.HybridSequential()
            self.encoder.add(nn.Conv2D(64, kernel_size=3, padding=1))
            self.encoder.add(nn.BatchNorm())
            self.encoder.add(nn.Activation('relu'))
            self.encoder.add(nn.Dropout(0.5))  # Added dropout to stabilize training

            self.decoder = nn.HybridSequential()
            self.decoder.add(nn.Conv2D(64, kernel_size=3, padding=1))
            self.decoder.add(nn.BatchNorm())
            self.decoder.add(nn.Activation('relu'))
            self.decoder.add(nn.Dropout(0.5))  # Added dropout to stabilize training

            self.output = nn.Conv2D(1, kernel_size=1)  # Output for binary segmentation

    def hybrid_forward(self, F, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return self.output(x)

# Function to compute IoU
def iou_metric(pred, mask):
    try:
        pred = (pred > 0.5).astype(np.uint8)
        intersection = np.logical_and(mask, pred)
        union = np.logical_or(mask, pred)
        iou_score = np.sum(intersection) / np.sum(union)
    except Exception as e:
        print(f"Error computing IoU: {e}")
        iou_score = 0
    return iou_score

# Function for validation
def validate(model, val_data, ctx):
    iou_scores = []
    try:
        for data, label in val_data:
            data = data.as_in_context(ctx)
            label = label.as_in_context(ctx)

            pred = model(data)
            pred = nd.sigmoid(pred)
            print(f"Sigmoid output - min: {nd.min(pred).asscalar()}, max: {nd.max(pred).asscalar()}")
            pred = pred > 0.5

            iou = iou_metric(pred.asnumpy(), label.asnumpy())
            iou_scores.append(iou)
    except Exception as e:
        print(f"Error during validation: {e}")
    return np.mean(iou_scores)

# Training function with loss plotting and hyperparameter control
def train_model(epochs, batch_size, learning_rate):
    try:
        ctx = mx.gpu() if mx.context.num_gpus() > 0 else mx.cpu()
    except:
        ctx = mx.cpu()

    model = ResUNetA()
    try:
        model.initialize(ctx=ctx)
        model.hybridize()
    except Exception as e:
        print(f"Error initializing or hybridizing model: {e}")

    # Define loss function and trainer
    loss_fn = gluon.loss.SigmoidBinaryCrossEntropyLoss(from_sigmoid=True)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': learning_rate, 'clip_gradient': 0.1})

    # DataLoaders for training and validation
    try:
        train_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
        val_data = mx.gluon.data.DataLoader(mx.gluon.data.ArrayDataset(X_val, Y_val), batch_size=batch_size, shuffle=False)

    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return

    train_losses = []
    val_ious = []

    # Training loop
    for epoch in range(epochs):
        epoch_loss = 0
        try:
            for i, (data, label) in enumerate(train_data):
                data = data.as_in_context(ctx)
                label = label.as_in_context(ctx)

                with autograd.record(): # Start recording the operations for autograd
                    output = model(data) # Forward pass: compute the model output
                    output_sigmoid = nd.sigmoid(output)
                    # Print output stats before and after sigmoid
                    print(f"Output before sigmoid - min: {nd.min(output).asscalar()}, max: {nd.max(output).asscalar()}")
                    print(f"Output after sigmoid - min: {nd.min(output_sigmoid).asscalar()}, max: {nd.max(output_sigmoid).asscalar()}")

                    loss = loss_fn(output_sigmoid, label)
                    print(f'loss: {loss}')
                loss.backward()  # Backward pass: compute the gradients
                print(f'loss after backward: {loss}')
                trainer.step(batch_size) # Update model parameters

                # Check for NaNs in output and loss
                print(f"Model output stats - min: {nd.min(output).asscalar()}, max: {nd.max(output).asscalar()}")
                epoch_loss += nd.mean(loss).asscalar()
                print(f'Batch {i}, epoch_loss: {epoch_loss}')
                print(f'epoch_loss: {epoch_loss}')

            print(f'Epoch {epoch + 1}, total epoch_loss: {epoch_loss}')
            print(f'train data: {train_data}')

            avg_loss = epoch_loss / len(train_data)
            val_iou = validate(model, val_data, ctx)

            train_losses.append(avg_loss)
            val_ious.append(val_iou)

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}, Validation IoU: {val_iou:.4f}")
        except Exception as e:
            print(f"Error during training epoch {epoch + 1}: {e}")

    # Plot training loss
    try:
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Training Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(loss_path)
        plt.show()
    except Exception as e:
        print(f"Error plotting training losses: {e}")

    # Return model and metrics
    return model, train_losses, val_ious

# Run training with hyperparameters
try:
    epochs = 50
    batch_size = 8
    learning_rate = 0.000001
    model, train_losses, val_ious = train_model(epochs, batch_size, learning_rate)
except Exception as e:
    print(f"Error during model training: {e}")
