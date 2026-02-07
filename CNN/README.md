# CNN - CIFAR-10 Image Classification

This folder contains a complete, single-notebook workflow for training a Convolutional Neural Network (CNN) on the CIFAR-10 dataset and using the trained model to predict labels for custom images.

## What This Project Does
- Loads CIFAR-10 (60,000 32x32 color images across 10 classes).
- Normalizes pixel values to speed up training.
- Trains a simple CNN with convolution + pooling layers.
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
7. **Compile and train**
   - Optimizer: Adam
   - Loss: Sparse Categorical Crossentropy (from logits)
   - Epochs: 10
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

## Notes
- First run will download the CIFAR-10 dataset automatically.
- Training for 10 epochs should finish quickly on CPU.
- For better accuracy, increase epochs or tune the model.
