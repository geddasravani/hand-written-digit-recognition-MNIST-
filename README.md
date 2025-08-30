# Handwritten Digit Recognition (MNIST) — CNN with TensorFlow/Keras

This having steps follows:
## 📌 Problem Statement
Handwritten text varies widely in style and neatness. Automatically recognizing digits from scanned or photographed handwriting is valuable for education tech (e.g., reading answers on OMR sheets, grading worksheets) and accessibility tools.
## 🎯 Objective
Build a **Convolutional Neural Network (CNN)** to recognize digits (0–9) using the **MNIST** dataset and evaluate performance to achieve **≥98% test accuracy**.
## 📦 Requirements (Conceptual)

* **Python** 3.9+
* **TensorFlow/Keras** for building and training the CNN
* **NumPy** for tensor manipulation
* **Matplotlib** for learning curves (optional but recommended)
* **scikit‑learn** for metrics (classification report & confusion matrix)

> *Installation (example):*
>> ```bash
> pip install tensorflow numpy matplotlib scikit-learn
## 🗂️ Dataset: MNIST (What & Why)

* **What it is:** 70,000 grayscale images of handwritten digits (28×28 pixels): **60,000** for training and **10,000** for testing.
* **Why it’s used:** Clean, standardized benchmark; small and fast to train; excellent for learning CNN fundamentals.
* **How we access it:** `tf.keras.datasets.mnist.load_data()` — downloads and caches the dataset automatically.

---

## 🔢 End‑to‑End Pipeline (Theory + What the Code Does)

Below are the 8 steps you requested, mapped to what the code performs and **why** each step matters.

### 1) Import Necessary Libraries

**What:** Bring in TensorFlow/Keras, NumPy, Matplotlib, and scikit‑learn.

**Why:**

* **TensorFlow/Keras**: define layers (Conv2D, MaxPooling2D, Dense), compile, train, and evaluate the model.
* **NumPy**: numerical arrays, reshaping, type casting.
* **Matplotlib**: visualize training curves to diagnose under/overfitting.
* **scikit‑learn**: create the classification report and confusion matrix.

---

### 2) Load the Dataset

**What:**

```python
(x_train, y_train), (x_test, y_test) = mnist.load_data()

This yields **60k** training images/labels and **10k** test images/labels. Each image is 28×28 with pixels in **\[0, 255]** (uint8).

**Why:** Provides a **train/test split** out of the box with representative distributions.
**
**### 3) Data Preprocessing (Cleaning, Missing Values, Noisy Data)**

**What the code does:**

* **Missing values check:** MNIST is clean; we verify with `np.isnan(...).sum()`.
* **Fallback cleaning:** Replace any accidental NaNs using `np.nan_to_num` (defensive programming).
* **Normalization:** Cast to float32 and divide by 255 → scale inputs to **\[0, 1]**.
* **Reshaping:** Expand channel dimension to **(N, 28, 28, 1)** for CNNs.
* **Label encoding:** One‑hot encode labels (0–9) for softmax classification.

**Why it matters:**

* **Normalization** stabilizes gradients and speeds up convergence.
* **Channel dimension** is required by Conv2D which expects H×W×C.
* **One‑hot labels** match the **categorical\_crossentropy** loss.
**Handling noisy data (theory & options):**
* MNIST is relatively clean. For *noisy* real‑world digits (e.g., from photos), consider:
  * **Denoising**: median filtering or Gaussian blur for salt‑and‑pepper noise.
  * **Thresholding**: binarize background vs foreground if lighting varies.
  * **Augmentation**: random shifts/rotations/zoom to improve robustness.
### 4) Split the Data into Training & Testing Sets
**What:** MNIST already provides a **training set (60k)** and a **held‑out test set (10k)**. Within training, we further reserve **validation** data (e.g., `validation_split=0.1`) to monitor generalization during training.
**Why:**
* **Training set**: learn model parameters.
* **Validation set**: tune hyperparameters & detect overfitting.
* **Test set**: unbiased estimate of final performance.
### 5) Train the Model (CNN to Recognize Digits)
**Model architecture (conceptual):**
* **Conv2D(32, 3×3, ReLU)** → extracts local features (strokes/edges)
* **MaxPooling2D(2×2)** → reduces spatial size, adds translation tolerance
* **Conv2D(64, 3×3, ReLU)** → learns more complex patterns
* **MaxPooling2D(2×2)** → further downsampling
* **Flatten** → converts feature maps to a 1D vector
* **Dense(128, ReLU)** → combines features into higher‑level concepts
* **Dropout(0.5)** → regularization to reduce overfitting
* **Dense(10, Softmax)** → outputs class probabilities for digits 0–9

**Compile settings:**

* **Optimizer:** `adam` — adaptive learning rate, fast convergence
* **Loss:** `categorical_crossentropy` — multi‑class classification
* **Metrics:** `accuracy`

**Training loop:**

* **Epochs:** commonly **8–15** for this architecture
* **Batch size:** **128**
* **Validation split:** **0.1** (10% of training set used for validation)

**Why this works:** CNNs exploit spatial locality, share weights across the image, and learn robust features (edges → parts → whole digits) with fewer parameters than fully‑connected nets.
### 6) Test the Model (Evaluate on Held‑Out Test Set)

**What:** Run `model.evaluate(x_test, y_test_cat)` to compute **test loss** and **test accuracy** on the 10,000 images not seen during training.

**Why:** Confirms generalization. Test accuracy close to validation accuracy indicates minimal overfitting.
### 7) Predict the Metrics (Qualitative & Quantitative)

**What the code outputs:**
* **Classification Report**: precision, recall, F1‑score for each class 0–9
* **Confusion Matrix**: counts of true vs predicted classes
* **Learning Curves**: training/validation accuracy and loss per epoch
**What these mean (theory):**
* **Precision (per class):** Of all images predicted as class *k*, how many were correct?
* **Recall (per class):** Of all true images of class *k*, how many did we catch?
* **F1 (per class):** Harmonic mean of precision & recall (balances the two).
* **Confusion Matrix:** Shows specific confusions (e.g., 4 vs 9). Useful for debugging and deciding augmentations.

**How accuracy is computed:**
$\text{Accuracy} = \frac{\text{# correct predictions}}{\text{# total samples}}$
### 8) Dataset File Links

Use any of the following reliable sources:

* **Keras .npz (direct, used by `mnist.load_data`)**: [https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)
* **Official MNIST page (Yann LeCun)**: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)
* **Kaggle MNIST dataset (for notebooks/competitions)**: [https://www.kaggle.com/competitions/digit-recognizer](https://www.kaggle.com/competitions/digit-recognizer)

> *Note:* Your script does **not** need manual downloads if you call `mnist.load_data()` — Keras fetches and caches automatically.
## ✅ Expected Results & Accuracy

* With the architecture & settings above (**\~8–15 epochs**, batch **128**, validation split **0.1**), typical **test accuracy** is **≈98–99%**.
* Small fluctuations come from random initialization and the number of epochs.
* To push reliably to **≥99%**:

  * Train a bit longer (10–15 epochs) or enable **EarlyStopping** on `val_loss`.
  * Add **BatchNormalization** after Conv layers.
  * Use light **augmentations** (random shifts/rotations) to improve robustness.

> **Interpretation:** Anything ≥98% on the MNIST test set indicates the model learned general digit shapes well. Exam‑style digits drawn with different thickness/angles may need slight augmentation or UI‑side preprocessing (invert colors, center/resize) to match MNIST’s style.
## 🧪 How to Read the Curves & Tables
* **Training vs Validation Accuracy**
  * Both rising and close together → good fit
  * Training ≫ Validation → overfitting → increase dropout, add augmentation, or reduce epochs
* **Confusion Matrix**
  * Concentration on the diagonal is ideal
  * Off‑diagonal hotspots show common mistakes (e.g., 5↔6) → guide augmentation

