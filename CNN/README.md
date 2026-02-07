# CNN - CIFAR-10 Image Classification

This folder contains a complete, single-notebook workflow for training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and using the trained model to predict labels for custom images.

## What This Project Does
- Loads CIFAR-10 (60,000 32x32 color images across 10 classes).
- Normalizes pixel values to speed up training.
- Trains a simple CNN with convolution + pooling layers.
- Uses Early Stopping to prevent overfitting.
- Visualizes training vs validation accuracy.
- Evaluates test accuracy.
- Runs predictions on images stored in `predicted_images/`.

## Folder Structure
- `cnn.ipynb` - End-to-end notebook (data load, model, training, evaluation, prediction).
- `predicted_images/` - Sample images for prediction (PNG/JPG). Use your own images here.

## Step-by-Step Workflow (What the Notebook Does)
1. **Import libraries**
   - TensorFlow/Keras for the model.
   - Matplotlib for plots.
2. **Load CIFAR-10**
   - Uses `tf.keras.datasets.cifar10.load_data()`.
3. **Normalize data**
   - Scales pixel values to `[0, 1]` by dividing by 255.
4. **Preview samples**
   - Shows 25 training images with labels.
5. **Build the CNN**
   - Conv2D -> MaxPool -> Conv2D -> MaxPool -> Conv2D
6. **Add classifier head**
   - Flatten -> Dense(64, relu) -> Dense(10)
7. **Compile and train with Early Stopping**
   - Optimizer: Adam
   - Loss: Sparse Categorical Crossentropy (from logits)
   - EarlyStopping monitors `val_loss` with `patience=3` and restores best weights
8. **Plot accuracy**
   - Training vs validation accuracy per epoch.
9. **Evaluate**
   - Prints test accuracy on CIFAR-10 test set.
10. **Predict custom images**
   - Loads each image in `predicted_images/`, resizes to 32x32, predicts class name.

## How to Run

### Option 1: Jupyter Notebook (recommended)
1. Create and activate a virtual environment (optional but recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install tensorflow matplotlib numpy jupyter
```

3. Launch Jupyter and open the notebook:

```powershell
jupyter notebook
```

Then open `CNN/cnn.ipynb` and run all cells in order.

### Option 2: VS Code Notebook
1. Open the workspace in VS Code.
2. Open `CNN/cnn.ipynb`.
3. Select a Python kernel with TensorFlow installed.
4. Run cells top to bottom.

## Using Your Own Images
- Put your images into `CNN/predicted_images/`.
- Supported formats: `.png`, `.jpg`, `.jpeg`.
- The notebook automatically resizes images to 32x32 and prints the predicted class.

## Class Labels
The model predicts one of these CIFAR-10 classes:
`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

## Why Use Early Stopping
Early stopping is a simple regularization technique. During training, the model can start to memorize the training data and lose performance on new data (overfitting). Early stopping watches validation loss and stops training when it stops improving. This saves time, reduces overfitting, and keeps the best version of the model.

## CNN Revision Notes (Theory)
Use this section as a quick revision guide.

### Convolution
A convolution applies a small matrix (kernel/filter) across the image to detect patterns like edges or textures. It produces a feature map that highlights where those patterns appear.

### Kernel (Filter)
A kernel is the small matrix used in convolution (for example 3x3). It slides over the image and computes a dot product to produce each value in the feature map.

### Stride
Stride is the step size of the kernel as it moves across the image. Stride 1 moves one pixel at a time; stride 2 skips every other pixel, reducing output size.

### Padding
Padding adds extra pixels (usually zeros) around the image so the output size is controlled. With padding, we can keep the same spatial size or avoid losing edge information.

### Horizontal Edge Filter
A kernel designed to detect horizontal edges by responding strongly where intensity changes along the vertical direction. Example:
```
[[-1, -1, -1],
 [ 0,  0,  0],
 [ 1,  1,  1]]
```

### Vertical Edge Filter
A kernel designed to detect vertical edges by responding strongly where intensity changes along the horizontal direction. Example:
```
[[-1,  0,  1],
 [-1,  0,  1],
 [-1,  0,  1]]
```

### Pooling
Pooling reduces the spatial size of feature maps to make the model faster and more robust to small shifts in the image.

### Max Pooling
Takes the maximum value from each window (e.g., 2x2). Keeps the strongest feature in each region.

### Min Pooling
Takes the minimum value from each window. Rarely used in practice for CNNs, but conceptually the opposite of max pooling.

### Flattening
Converts the 2D feature maps into a 1D vector so it can be passed into fully connected (dense) layers.

### ANN (Artificial Neural Network)
The dense layers at the end of a CNN act like a traditional ANN. They combine extracted features to make the final classification decision.

### Full Pipeline Summary
Convolution -> Pooling -> Flatten -> ANN (Dense Layers) -> Output

### Output Layer
For CIFAR-10, the output layer has 10 neurons. Each neuron corresponds to a class. The highest value indicates the predicted class.

### Dropout
Dropout randomly turns off neurons during training to reduce overfitting.
But here in a CNN we dont need ir becuase CNN has a already built in mechnasim for it .

### How compilation Processs Happened
1-forward Pass
2-Loss Calculation
3-Backpropogation
4-Weight Updates
5-Repeat

## Notes
- First run will download the CIFAR-10 dataset automatically.
- Training for up to 10 epochs should finish quickly on CPU.
- For better accuracy, increase epochs or tune the model.
