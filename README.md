# Animal Face Classification with a Convolutional Neural Network (PyTorch)

A from-scratch Convolutional Neural Network, implemented in PyTorch, that classifies animal face images into **cat**, **dog**, and **wildlife** categories using the [AFHQ (Animal Faces-HQ)](https://www.kaggle.com/datasets/andrewmvd/animal-faces) dataset. This document is a complete engineering and mathematical reference for the model: every layer, every tensor shape, every design decision, and every equation that governs training is derived and justified below.

<p align="center">
<b>3-class image classification&nbsp;·&nbsp;custom CNN&nbsp;·&nbsp;4.29M parameters&nbsp;·&nbsp;~95–96% validation accuracy</b>
</p>

---

## Table of Contents

1. [Project Summary](#1-project-summary)
2. [Results at a Glance](#2-results-at-a-glance)
3. [Repository Structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Dataset](#5-dataset)
6. [Data Pipeline — Engineering Decisions](#6-data-pipeline--engineering-decisions)
7. [Model Architecture](#7-model-architecture)
8. [Mathematical Foundations](#8-mathematical-foundations)
   - [8.1 Convolution](#81-convolution)
   - [8.2 Max Pooling](#82-max-pooling)
   - [8.3 ReLU](#83-relu)
   - [8.4 Flatten](#84-flatten)
   - [8.5 Fully Connected (Linear) Layer](#85-fully-connected-linear-layer)
   - [8.6 Softmax](#86-softmax)
   - [8.7 Cross-Entropy Loss](#87-cross-entropy-loss)
   - [8.8 Backpropagation Through the Network](#88-backpropagation-through-the-network)
   - [8.9 Adam Optimizer](#89-adam-optimizer)
9. [Tensor Shape Trace (Full Forward Pass)](#9-tensor-shape-trace-full-forward-pass)
10. [Computational & Memory Complexity](#10-computational--memory-complexity)
11. [Training Configuration](#11-training-configuration)
12. [Training & Validation Results](#12-training--validation-results)
13. [Test Set Evaluation](#13-test-set-evaluation)
14. [Inference Pipeline](#14-inference-pipeline)
15. [Design Rationale: Advantages & Disadvantages](#15-design-rationale-advantages--disadvantages)
16. [Alternative Approaches](#16-alternative-approaches)
17. [Known Limitations & Future Work](#17-known-limitations--future-work)
18. [How to Reproduce](#18-how-to-reproduce)

---

## 1. Project Summary

This repository trains a **custom convolutional neural network from scratch** (no pretrained backbone, no transfer learning) to classify RGB images of animal faces into one of three classes: `cat`, `dog`, `wild`. The pipeline covers the complete lifecycle of a supervised computer vision system:

`raw image directories → path/label indexing → stratified-by-construction split → PyTorch Dataset/DataLoader → CNN forward pass → cross-entropy loss → Adam optimization → validation → test evaluation → single-image inference`

The network is intentionally simple — three convolutional blocks followed by two fully connected layers — so that every mathematical operation inside it can be reasoned about explicitly rather than treated as a black box. This README documents that reasoning end-to-end.

## 2. Results at a Glance

| Metric | Value |
|---|---|
| Classes | 3 (`cat`, `dog`, `wild`) |
| Input resolution | 128 × 128 × 3 |
| Total trainable parameters | 4,288,067 |
| Optimizer | Adam (lr = 1e-4) |
| Loss function | Cross-Entropy Loss |
| Batch size | 16 |
| Epochs | 10 |
| Best validation accuracy | ~96.4% (epoch 7) |
| Final training accuracy | 99.67% (epoch 10) |
| Device used for training | CPU |

> Full per-epoch numbers are in [Section 12](#12-training--validation-results).

## 3. Repository Structure

```
.
├── Image_Classification.ipynb   # End-to-end notebook: data prep, model, training, eval, inference
├── animal-faces/                # Downloaded dataset (via opendatasets / Kaggle API), gitignored
│   └── afhq/
│       ├── train/{cat,dog,wild}/
│       └── val/{cat,dog,wild}/
├── README.md                    # This file
└── requirements.txt             # Python dependencies (see Installation)
```

## 4. Installation

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2. Create an isolated environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install torch torchvision torchsummary pandas numpy scikit-learn matplotlib pillow opendatasets
```

**Kaggle credentials**: the dataset is pulled programmatically via `opendatasets`, which will prompt for a Kaggle username and [API key](https://www.kaggle.com/docs/api) (`kaggle.json`) the first time it runs.

```python
import opendatasets as od
od.download("https://www.kaggle.com/datasets/andrewmvd/animal-faces")
```

## 5. Dataset

**AFHQ (Animal Faces-HQ)**, published by Choi et al. (StarGAN v2, NVIDIA), contains high-quality animal face images evenly split across three domains:

| Class | Description |
|---|---|
| `cat` | Domestic cat faces |
| `dog` | Domestic dog faces |
| `wild` | Wild animal faces (foxes, lions, tigers, wolves, etc.) |

The images arrive pre-organized as `afhq/{train,val}/{cat,dog,wild}/*.jpg`. Rather than relying on the dataset's own train/val split, this project **flattens both folders into one pool and re-splits it**, which is a deliberate engineering choice explained next.

## 6. Data Pipeline — Engineering Decisions

### 6.1 Path indexing

Every image path and its parent-folder label are walked recursively into a single `pandas.DataFrame` with columns `image_paths` and `labels`. This decouples the *logical* split (train/val/test) from the *physical* directory layout the dataset ships with, giving full control over split ratios and reproducibility.

### 6.2 Why re-split instead of using the dataset's own folders?

The AFHQ Kaggle release ships a train/val split, but it does **not** ship a held-out test set. Pooling all images and re-splitting into train/val/test guarantees a genuinely unseen test set for the final unbiased performance estimate — using the original "val" folder as a test set would conflate validation-time model selection with final evaluation.

### 6.3 Split ratios

```python
train = data_df.sample(frac=0.70, random_state=42)   # 70%
test  = data_df.drop(train.index)
val   = test.sample(frac=0.50, random_state=42)       # 15% of total
test  = test.drop(val.index)                          # 15% of total
```

This yields a **70 / 15 / 15** train/validation/test split.

- **Why 70/15/15?** It follows the standard practical heuristic for small-to-medium datasets (tens of thousands of images): enough training signal to fit ~4.3M parameters, while keeping validation and test sets large enough to give statistically stable accuracy estimates (a common alternative is 80/10/10; 70/15/15 trades a little training data for tighter validation/test confidence).
- **Why `random_state=42`?** Determinism. Given the same dataframe, the same three subsets are produced on every run, which is essential for reproducible experiments and fair comparison across hyperparameter changes.
- **Caveat — no stratification check**: `DataFrame.sample` performs a simple random sample, not a class-stratified sample. Because AFHQ classes are numerically balanced to begin with, the resulting splits end up close to balanced in practice, but for a class-imbalanced dataset this would need `train_test_split(..., stratify=labels)` from scikit-learn instead.

### 6.4 Label encoding

```python
label_encoder = LabelEncoder()
label_encoder.fit(data_df['labels'])
```

`CrossEntropyLoss` in PyTorch expects integer class indices (`0, 1, 2, …`), not string labels. `LabelEncoder` produces a deterministic, alphabetically-ordered string→integer mapping (`cat=0, dog=1, wild=2`) that is fit once on the *full* label set — fitting on the full set (not per-split) guarantees the same encoding is used everywhere, and `inverse_transform` at inference time recovers the human-readable label.

### 6.5 Image transforms

```python
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.ConvertImageDtype(torch.float)
])
```

| Step | Purpose |
|---|---|
| `Resize((128,128))` | CNNs with fully connected heads require a **fixed input size** — the flatten step downstream hard-codes `128×16×16` elements, so every image must be resized to the same resolution before it reaches the network. 128×128 balances retained spatial detail against compute cost (a 224×224 input would roughly 3× the FLOPs of every conv layer). |
| `ToTensor()` | Converts a `PIL.Image` (H×W×C, values 0–255, `uint8`) into a `torch.Tensor` (C×H×W, values 0–1, reordering to PyTorch's channel-first convention). |
| `ConvertImageDtype(torch.float)` | Ensures the tensor is `float32` — required because convolution weights are floats and PyTorch does not silently mix dtypes during matrix multiplication. |

No normalization (mean/std subtraction) or data augmentation (random flips/crops/color-jitter) is applied — see [Limitations](#17-known-limitations--future-work) for the implications.

### 6.6 Custom `Dataset`

```python
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(label_encoder.transform(dataframe['labels'])).to(device)

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
```

This subclasses `torch.utils.data.Dataset` and implements the two methods PyTorch's `DataLoader` requires: `__len__` (dataset size, used for batching/shuffling logic) and `__getitem__` (lazy, on-demand loading of a single sample). **Lazy loading is the key design decision here**: images are read from disk and transformed only when indexed, not all at once at construction time — this keeps memory usage bounded regardless of dataset size, at the cost of repeated disk I/O and repeated resize/decode work every epoch (see [Alternative Approaches](#16-alternative-approaches) for caching strategies).

`.convert("RGB")` normalizes away images that might be grayscale or have an alpha channel (RGBA/PNG), guaranteeing every tensor has exactly 3 channels, matching `conv1`'s `in_channels=3`.

### 6.7 DataLoader

```python
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=True)
```

`shuffle=True` randomizes sample order each epoch, which is important even for validation/test loaders here because it prevents any accidental ordering artifact (e.g., all images of one class appearing consecutively) from affecting batch-level statistics, though for validation/test it has no effect on the final aggregate metric since every sample is still seen exactly once.

## 7. Model Architecture

```python
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(128 * 16 * 16, 128)
        self.output = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)      # (B, 32, 128, 128)
        x = self.pooling(x)    # (B, 32, 64, 64)
        x = self.relu(x)
        x = self.conv2(x)      # (B, 64, 64, 64)
        x = self.pooling(x)    # (B, 64, 32, 32)
        x = self.relu(x)
        x = self.conv3(x)      # (B, 128, 32, 32)
        x = self.pooling(x)    # (B, 128, 16, 16)
        x = self.relu(x)
        x = self.flatten(x)    # (B, 32768)
        x = self.linear(x)     # (B, 128)
        x = self.output(x)     # (B, 3)  <- raw logits
        return x
```

This is a **VGG-style** architecture: a stack of `(Conv → Pool → Nonlinearity)` blocks that progressively trade spatial resolution for channel depth, followed by a two-layer MLP classification head.

### A note on operation order: `Conv → Pool → ReLU`

Most textbook CNNs apply the nonlinearity *before* pooling (`Conv → ReLU → Pool`). This implementation instead pools before the ReLU. Because `MaxPool` and `ReLU` are both **monotonic, order-preserving** operations on the values they see, `ReLU(MaxPool(x)) == MaxPool(ReLU(x))` mathematically — the two orderings are functionally equivalent for max pooling specifically (this would *not* hold for average pooling). The practical benefit of pooling first is computational: ReLU is applied to a 4× smaller tensor (after a 2×2 pool), reducing the number of elementwise operations, with no change to the function the network represents.

### Why no Batch Normalization or Dropout?

Their absence is a real design choice, not an oversight — see [Section 15](#15-design-rationale-advantages--disadvantages) and [Section 16](#16-alternative-approaches) for a full discussion of what they would add and cost.

## 8. Mathematical Foundations

### 8.1 Convolution

A 2D convolution layer (technically **cross-correlation** in the deep learning convention — no kernel flipping is performed) slides a learnable filter across the spatial dimensions of the input and computes a dot product at every position.

For an input feature map $X$ with $C_{in}$ channels and a filter bank $W$ of shape $(C_{out}, C_{in}, K, K)$, the output at spatial location $(i, j)$ for output channel $c$ is:

$$
Y_{c,i,j} = b_c + \sum_{k=0}^{C_{in}-1} \sum_{m=0}^{K-1} \sum_{n=0}^{K-1} W_{c,k,m,n} \cdot X_{k,\, i \cdot s + m - p,\; j \cdot s + n - p}
$$

where $s$ is the stride, $p$ is the padding, and $b_c$ is the bias for output channel $c$.

**Output spatial size formula:**

$$
H_{out} = \left\lfloor \frac{H_{in} + 2p - K}{s} \right\rfloor + 1
$$

For every conv layer in this network, `kernel_size=3`, `padding=1`, `stride=1` (default):

$$
H_{out} = \frac{H_{in} + 2(1) - 3}{1} + 1 = H_{in}
$$

This is exactly why `padding=1` was chosen with a `3×3` kernel — it is the **"same padding"** configuration, which preserves spatial dimensions and lets each conv layer add capacity (channels) without shrinking the feature map, deferring all downsampling to the explicit `MaxPool2d` layers. This separation of concerns (convolution = feature extraction, pooling = downsampling) makes the network's spatial shrinkage schedule easy to reason about and tune independently of receptive field growth.

**Parameter count** for a conv layer:

$$
\#\text{params} = (C_{in} \cdot K \cdot K + 1) \cdot C_{out}
$$

(the `+1` accounts for one bias term per output channel). Verified against this implementation:

| Layer | $C_{in}$ | $C_{out}$ | $K$ | Params |
|---|---|---|---|---|
| conv1 | 3 | 32 | 3 | $(3{\cdot}9{+}1){\cdot}32 = 896$ |
| conv2 | 32 | 64 | 3 | $(32{\cdot}9{+}1){\cdot}64 = 18{,}496$ |
| conv3 | 64 | 128 | 3 | $(64{\cdot}9{+}1){\cdot}128 = 73{,}856$ |

**Receptive field**: each successive `3×3` conv adds 2 pixels to the receptive field radius before pooling, and each `2×2` max-pool doubles the *effective* receptive field of everything downstream. By the output of `conv3`, one output neuron's receptive field spans roughly $46 \times 46$ pixels of the original $128\times128$ input — large enough to capture whole facial features (eyes, ears, muzzle shape) that distinguish cat/dog/wild faces.

**Why increase channels while decreasing spatial size (32 → 64 → 128 as $H,W$ shrink)?** Early layers need few channels because they only detect simple, reusable primitives (edges, color gradients, textures); as spatial resolution shrinks, more channels are needed to represent the combinatorially larger space of higher-level, more abstract patterns (fur texture + ear shape + eye spacing) built from those primitives. This channel-doubling-as-resolution-halves pattern keeps the compute cost roughly constant per stage (see [Section 10](#10-computational--memory-complexity)).

### 8.2 Max Pooling

`MaxPool2d(kernel_size=2, stride=2)` partitions each channel into non-overlapping $2\times2$ windows and keeps only the maximum value in each:

$$
Y_{c,i,j} = \max_{0 \le m,n < 2} X_{c,\, 2i+m,\; 2j+n}
$$

- **Output size**: exactly halves both spatial dimensions ($H_{out} = H_{in}/2$), with **zero learnable parameters**.
- **Why max, not average?** Max pooling preserves the strongest local activation of a feature detector, which tends to correspond to "this feature is present somewhere in this region" — a good inductive bias for classification, where the presence of a discriminating feature (e.g., a pointed ear) matters more than its average strength across a neighborhood. Average pooling would blur sharp feature responses.
- **Gradient**: during backpropagation, the gradient flows *only* to the input location that was the max (a "gradient routing" / winner-take-all mechanism); all other locations in the window receive zero gradient for that output element. This makes max pooling non-differentiable at ties (handled by convention, e.g., routing to the first max) but differentiable almost everywhere.

### 8.3 ReLU

$$
\text{ReLU}(x) = \max(0, x), \qquad \frac{d}{dx}\text{ReLU}(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{undefined (conventionally 0)} & x = 0 \end{cases}
$$

- **Why ReLU over sigmoid/tanh?** Sigmoid and tanh saturate (derivative → 0) for large $|x|$, causing vanishing gradients that slow or stall learning in deeper networks. ReLU's derivative is exactly 1 for all positive inputs, so gradients pass through unattenuated, which is a major reason ReLU enabled training of much deeper CNNs than was previously practical.
- **Zero parameters, zero cost** beyond a single comparison per element — this is also why it is applied after pooling in this implementation (Section 7), to minimize the number of elements it touches.
- **Known failure mode — "dying ReLU"**: if a neuron's weights update such that its pre-activation is negative for every input in the training set, its gradient becomes permanently zero and it stops learning. This is a real risk in this network (no leaky variant is used) and is one reason Adam, rather than plain SGD, is used as the optimizer — see [Section 8.9](#89-adam-optimizer).

### 8.4 Flatten

`nn.Flatten()` reshapes the final pooled feature map from $(B, 128, 16, 16)$ into $(B, 32768)$ by concatenating all channel/spatial elements into a single vector per sample, **without changing any values** — it is a pure reshape (a view over contiguous memory, no computation, no parameters).

$$
128 \times 16 \times 16 = 32{,}768
$$

This is the bridge between the spatially-aware convolutional stage and the spatially-agnostic fully-connected stage: from this point on, the network has no notion of "where" a feature was in the image, only "whether" it fired and how strongly.

### 8.5 Fully Connected (Linear) Layer

A linear layer computes:

$$
y = Wx + b, \qquad W \in \mathbb{R}^{d_{out} \times d_{in}},\; x \in \mathbb{R}^{d_{in}},\; b \in \mathbb{R}^{d_{out}}
$$

**`self.linear = nn.Linear(32768, 128)`**: every one of the 32,768 flattened features is connected to every one of 128 output neurons — a **dense** transformation, contrasted with convolution's *sparse, weight-shared* connectivity. Parameter count: $32{,}768 \times 128 + 128 = 4{,}194{,}432$ — this single layer accounts for **97.8% of the entire model's parameters**, which is the dominant cost driver discussed in [Section 10](#10-computational--memory-complexity).

**`self.output = nn.Linear(128, 3)`**: projects the 128-dimensional learned representation down to 3 raw scores (**logits**), one per class. These are *not* probabilities yet — they are unbounded real numbers whose relative magnitude encodes the model's relative confidence per class.

### 8.6 Softmax

Although this implementation never calls `softmax` explicitly, it is mathematically present *inside* `nn.CrossEntropyLoss`, and is the function that turns logits into a probability distribution during both training (implicitly) and inference (explicitly, if probabilities rather than a single class are desired):

$$
\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_{j=1}^{C} e^{z_j}}, \qquad i = 1, \dots, C
$$

Properties: every output lies in $(0,1)$ and all outputs sum to exactly 1, so the vector can be interpreted as a categorical probability distribution over the $C=3$ classes. In practice, PyTorch computes a numerically stable variant that subtracts $\max_j z_j$ from every logit before exponentiating, which leaves the result mathematically unchanged but prevents floating-point overflow for large logits.

### 8.7 Cross-Entropy Loss

`nn.CrossEntropyLoss()` combines `log_softmax` and negative log-likelihood in a single, numerically stable operation. For one sample with true class $y$ and logits $z$:

$$
\mathcal{L}(z, y) = -\log\left(\text{softmax}(z)_y\right) = -z_y + \log\sum_{j=1}^{C} e^{z_j}
$$

Over a batch of size $N$, the reported loss is the mean:

$$
\mathcal{L}_{\text{batch}} = \frac{1}{N}\sum_{n=1}^{N} \mathcal{L}(z^{(n)}, y^{(n)})
$$

**Why this is the correct loss for multi-class classification**: it is the negative log-likelihood of the true class under the model's predicted distribution, so *minimizing it is exactly equivalent to maximum-likelihood estimation* of the model's parameters given the observed (image, label) pairs — the loss is only small when the model assigns high probability to the correct class.

**Gradient with respect to the logits** — this is the single most important derivative in the whole training loop, since it is where backpropagation *begins*:

$$
\frac{\partial \mathcal{L}}{\partial z_i} = \text{softmax}(z)_i - \mathbb{1}[i = y]
$$

In words: the gradient on each logit is simply *"predicted probability minus 1 if this is the true class, else predicted probability minus 0."* This has an elegant consequence — if the model is perfectly confident and correct ($\text{softmax}(z)_y \to 1$), the gradient vanishes and learning naturally stops for that sample; if the model is confidently *wrong*, the gradient is large, driving a big correction. This clean, well-scaled gradient is a major reason cross-entropy (rather than, say, mean-squared error) is the standard loss for classification.

**Why `CrossEntropyLoss` and not `BCELoss`?** `BCELoss`/`BCEWithLogitsLoss` are for binary or independent multi-label problems; here the three classes are **mutually exclusive** (an image is a cat *or* a dog *or* wild, never a combination), which is exactly the assumption `CrossEntropyLoss`'s softmax normalization encodes.

### 8.8 Backpropagation Through the Network

Backpropagation is the recursive application of the **chain rule** to propagate $\partial \mathcal{L}/\partial z$ (Section 8.7) backward through every layer, producing a gradient for every learnable parameter.

**General chain rule for a layer** $y = f(x; \theta)$:

$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial x}, \qquad \frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial y}{\partial \theta}
$$

Applied layer-by-layer, backward through this network:

| Layer | Local gradient | What flows backward |
|---|---|---|
| `output` (Linear 128→3) | $\partial y/\partial W = x^\top$, $\partial y/\partial x = W^\top$ | Gradient w.r.t. 128-dim features and w.r.t. $W,b$ |
| `linear` (Linear 32768→128) | same form | Gradient w.r.t. flattened 32768-dim vector and w.r.t. $W,b$ |
| `flatten` | identity (reshape) | Gradient reshaped back to $(128,16,16)$, values unchanged |
| `relu` | $1$ where input $>0$, else $0$ | Gradient masked — zeroed wherever the forward activation was clipped |
| `pooling` (MaxPool) | $1$ at the arg-max location, $0$ elsewhere | Gradient routed only to the winning pixel in each $2\times2$ window |
| `conv{1,2,3}` | $\partial \mathcal{L}/\partial W$ is a correlation of the input with the upstream gradient; $\partial \mathcal{L}/\partial x$ is a *full convolution* of the upstream gradient with the (spatially-flipped) filter | Gradient w.r.t. filter weights, biases, and w.r.t. the layer's input feature map |

Formally, for a conv layer, the weight gradient is itself a convolution:

$$
\frac{\partial \mathcal{L}}{\partial W_{c,k,m,n}} = \sum_{i,j} \frac{\partial \mathcal{L}}{\partial Y_{c,i,j}} \cdot X_{k,\,i+m,\,j+n}
$$

and the gradient passed to the previous layer is the *transposed convolution* (upstream gradient convolved with the 180°-rotated filter), which is precisely why convolution and "deconvolution"/transposed-convolution are mathematical duals of each other. PyTorch's autograd computes all of this automatically via `loss.backward()` — the framework builds a dynamic computation graph during the forward pass and walks it in reverse, but the underlying operation at every node is exactly the chain-rule application above.

**In this codebase**, the entire backward pass is triggered by one line:

```python
train_loss.backward()   # populates .grad on every parameter tensor
optimizer.step()        # consumes those gradients to update weights (Section 8.9)
optimizer.zero_grad()   # clears .grad before the next batch (gradients accumulate by default in PyTorch)
```

### 8.9 Adam Optimizer

`Adam(model.parameters(), lr=1e-4)` (Kingma & Ba, 2014) is used instead of vanilla SGD. Adam maintains **per-parameter** adaptive learning rates by tracking exponential moving averages of both the gradient (first moment) and the squared gradient (second moment).

For each parameter $\theta$, at step $t$, with gradient $g_t = \nabla_\theta \mathcal{L}$:

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t \qquad \text{(biased first moment — momentum)}
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2 \qquad \text{(biased second moment — adaptive scale)}
$$

Bias correction (compensates for $m_0=v_0=0$ initialization skewing early estimates toward zero):

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

Parameter update:

$$
\theta_t = \theta_{t-1} - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

with PyTorch defaults $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$, and $\eta = \texttt{LR} = 10^{-4}$ in this implementation.

**Why Adam here specifically:**
- **Per-parameter learning rates** ($\hat m_t/(\sqrt{\hat v_t}+\epsilon)$) mean parameters that consistently receive small gradients (common deeper into a CNN, or for ReLU units near their dying threshold) get an *effectively larger* step size, helping counteract the vanishing/dying-ReLU issues discussed in Section 8.3, without any manual per-layer learning-rate tuning.
- **Momentum** ($m_t$) smooths out noisy gradients from small mini-batches (`BATCH_SIZE=16` here is small, so per-batch gradient estimates are noisy) and helps the optimizer traverse flat regions or shallow local structure in the loss landscape faster than raw SGD.
- **Why `lr=1e-4` and not, e.g., `1e-3`?** `1e-3` is Adam's commonly cited default and often too aggressive for a network with a very large single linear layer (4.19M of 4.29M parameters) trained on a moderately sized dataset — a smaller learning rate trades slower initial convergence for reduced risk of the large FC layer's weights oscillating or diverging early in training, which the smoothly decreasing training loss in [Section 12](#12-training--validation-results) suggests is the correct trade-off here.
- **Cost**: Adam stores two extra tensors ($m$ and $v$) per parameter, i.e. **3× the memory footprint of the raw parameters** during training (params + $m$ + $v$), versus 1× for vanilla SGD. For a 4.29M-parameter model at fp32, that is roughly 17.2 MB for weights and another ~34.3 MB for Adam's moment buffers — trivial on modern hardware, but a real consideration for far larger models.

## 9. Tensor Shape Trace (Full Forward Pass)

For a batch of size $B$ and input resolution $128\times128\times3$:

```
Input                (B,   3, 128, 128)
   │  Conv2d(3→32, k=3, p=1)        preserves H,W  (Sec 8.1: 2p-k+1=0)
   ▼
Conv1 out            (B,  32, 128, 128)
   │  MaxPool2d(2,2)                halves H,W
   ▼
Pool1 out             (B,  32,  64,  64)
   │  ReLU                          shape unchanged
   ▼
Conv2d(32→64, k=3, p=1)
   ▼
Conv2 out             (B,  64,  64,  64)
   │  MaxPool2d(2,2)
   ▼
Pool2 out             (B,  64,  32,  32)
   │  ReLU
   ▼
Conv2d(64→128, k=3, p=1)
   ▼
Conv3 out             (B, 128,  32,  32)
   │  MaxPool2d(2,2)
   ▼
Pool3 out             (B, 128,  16,  16)
   │  ReLU
   ▼
Flatten               (B, 32768)
   │  Linear(32768→128)
   ▼
FC1 out               (B, 128)
   │  Linear(128→3)
   ▼
Logits (output)       (B, 3)          ← raw scores, softmax applied inside CrossEntropyLoss
```

## 10. Computational & Memory Complexity

### 10.1 Parameter count (per layer)

| Layer | Output shape (per sample) | Parameters | % of total |
|---|---|---|---|
| Conv2d-1 (3→32) | 32×128×128 | 896 | 0.02% |
| MaxPool2d-2 | 32×64×64 | 0 | — |
| ReLU-3 | 32×64×64 | 0 | — |
| Conv2d-4 (32→64) | 64×64×64 | 18,496 | 0.43% |
| MaxPool2d-5 | 64×32×32 | 0 | — |
| ReLU-6 | 64×32×32 | 0 | — |
| Conv2d-7 (64→128) | 128×32×32 | 73,856 | 1.72% |
| MaxPool2d-8 | 128×16×16 | 0 | — |
| ReLU-9 | 128×16×16 | 0 | — |
| Flatten-10 | 32768 | 0 | — |
| Linear-11 (32768→128) | 128 | 4,194,432 | **97.82%** |
| Linear-12 (128→3) | 3 | 387 | 0.01% |
| **Total** | | **4,288,067** | 100% |

*(Verified via `torchsummary.summary(model, input_size=(3,128,128))`, matching the notebook output exactly.)*

**Key observation**: the first fully connected layer alone holds **97.8%** of all trainable parameters — a direct consequence of connecting *every* one of 32,768 flattened conv features to *every* one of 128 hidden units. This is the classic weakness of flattening late-stage conv features straight into a wide dense layer (see [Alternative Approaches](#16-alternative-approaches) for how `AdaptiveAvgPool2d`/Global Average Pooling eliminates this cost).

### 10.2 FLOPs (compute cost) per image, forward pass

Convolution FLOPs (multiply-accumulate, counted as 2 FLOPs each) follow:

$$
\text{FLOPs}_{\text{conv}} = 2 \cdot C_{in} \cdot C_{out} \cdot K^2 \cdot H_{out} \cdot W_{out}
$$

| Layer | MACs | FLOPs (≈2×MACs) |
|---|---|---|
| conv1 | 14,155,776 | 28,311,552 |
| conv2 | 75,497,472 | 150,994,944 |
| conv3 | 75,497,472 | 150,994,944 |
| linear1 | 4,194,304 | 8,388,608 |
| output | 384 | 768 |
| **Total** | **~169.3M MACs** | **~338.7M FLOPs** |

For context, this is roughly **two orders of magnitude cheaper** than a full-resolution ImageNet-scale model like ResNet-50 (~4.1 GFLOPs at 224×224) — a direct consequence of the small 128×128 input and shallow 3-conv-layer design. This makes the model comfortably trainable and runnable for inference on CPU, consistent with the notebook's `Device is: cpu` output.

### 10.3 Activation memory (forward pass, per sample, fp32)

| Tensor | Elements | Memory (fp32) |
|---|---|---|
| Input | 49,152 | 192.0 KB |
| conv1 out | 524,288 | 2,048.0 KB |
| pool1 out | 131,072 | 512.0 KB |
| conv2 out | 262,144 | 1,024.0 KB |
| pool2 out | 65,536 | 256.0 KB |
| conv3 out | 131,072 | 512.0 KB |
| pool3 out / flatten | 32,768 | 128.0 KB |
| linear1 out | 128 | 0.5 KB |
| logits | 3 | ~0 |

The single largest activation tensor is **`conv1` output** (2 MB/sample) — the highest-resolution, and one of the higher-channel-count, feature maps in the network. During training, PyTorch's autograd must retain every one of these intermediate activations (to compute gradients in the backward pass), so **peak training memory scales with `BATCH_SIZE × sum of all activation sizes`**, not just parameter count. This is why reducing `BATCH_SIZE` is the standard first lever when a model runs out of memory, even though it doesn't change the parameter count at all.

### 10.4 Total memory footprint (training, fp32, batch=16)

| Component | Approximate size |
|---|---|
| Model parameters | 4,288,067 × 4 B ≈ 16.4 MB |
| Adam optimizer state ($m$ + $v$) | 2 × 16.4 MB ≈ 32.7 MB |
| Gradients (`.grad`, same shape as params) | ≈ 16.4 MB |
| Activations (batch=16, sum of table above ≈ 4.6 MB/sample) | ≈ 74 MB |
| **Approximate peak training memory** | **~140 MB** |

This comfortably fits in CPU RAM (consistent with the notebook training on `cpu`) and would use only a small fraction of even entry-level GPU memory.

## 11. Training Configuration

| Hyperparameter | Value | Rationale |
|---|---|---|
| `LR` (learning rate) | `1e-4` | Conservative rate for Adam given the large single FC layer (Section 8.9) |
| `BATCH_SIZE` | `16` | Small batch → more frequent (noisier) updates, lower peak memory; reasonable given CPU training |
| `EPOCHS` | `10` | Enough passes to reach >99% train accuracy without the notebook's runtime becoming impractical on CPU |
| Optimizer | `Adam` | Adaptive, per-parameter learning rates; robust default for CNNs (Section 8.9) |
| Loss | `CrossEntropyLoss` | Correct loss for mutually-exclusive multi-class classification (Section 8.7) |
| Weight initialization | PyTorch default (Kaiming-uniform for `Conv2d`/`Linear`) | Not explicitly overridden in code; PyTorch's default already accounts for ReLU-style nonlinearities reasonably well |

## 12. Training & Validation Results

Actual output from the training loop:

| Epoch | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) |
|---|---|---|---|---|
| 1 | 2.1161 | 88.73 | 0.3050 | 92.77 |
| 2 | 1.1652 | 94.01 | 0.3186 | 92.48 |
| 3 | 0.7653 | 96.13 | 0.1968 | 95.66 |
| 4 | 0.5593 | 97.15 | 0.1802 | 95.83 |
| 5 | 0.4074 | 97.99 | 0.1953 | 95.58 |
| 6 | 0.3065 | 98.52 | 0.2575 | 94.55 |
| 7 | 0.2312 | 98.93 | **0.1692** | **96.41** |
| 8 | 0.1798 | 99.15 | 0.2149 | 95.91 |
| 9 | 0.1229 | 99.44 | 0.1870 | 96.32 |
| 10 | 0.0852 | **99.67** | 0.2893 | 95.54 |

**Reading the curves**: training loss decreases monotonically and training accuracy climbs smoothly toward ~99.7%, while validation loss *plateaus and begins to creep back up* after epoch 7 (0.169 → 0.215 → 0.187 → 0.289) even as training accuracy keeps rising. This divergence between falling training loss and rising validation loss is the textbook signature of **overfitting** setting in around epoch 7–8: the model increasingly memorizes training-set-specific detail that does not generalize. The best validation loss/accuracy occurs at **epoch 7**, suggesting an early-stopping checkpoint there (or at epoch 9, the runner-up) would generalize at least as well as, and possibly better than, the final epoch-10 model actually deployed for inference.

## 13. Test Set Evaluation

```python
with torch.no_grad():
    ...
print(f"Accuracy Score is: {...} and Loss is {...}")
```

The held-out 15% test split (never seen during training or used for model selection) is run through the trained model with gradient tracking disabled (`torch.no_grad()` — this skips building the autograd graph entirely, reducing memory and compute since no `.backward()` will ever be called on these forward passes). Given the validation trend above, expect test accuracy in the **~95–96%** range, consistent with validation performance.

## 14. Inference Pipeline

```python
def predict_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = transform(image).to(device)
    output = model(image.unsqueeze(0))
    output = torch.argmax(output, axis=1).item()
    return label_encoder.inverse_transform([output])
```

Four-stage pipeline, mirroring the training-time preprocessing exactly (critical — a train/inference preprocessing mismatch is one of the most common silent bugs in deployed ML systems):

1. **Load & normalize channels** — `.convert('RGB')` guarantees 3 channels regardless of source format.
2. **Transform** — the *exact same* `Resize → ToTensor → ConvertImageDtype` pipeline used in training, ensuring the model sees data distributed the same way it was trained on.
3. **Predict** — `image.unsqueeze(0)` adds a batch dimension (model expects `(B,3,128,128)`, a single image is `(3,128,128)`), producing raw logits of shape `(1,3)`; `torch.argmax(..., axis=1)` selects the class index with the highest logit — since softmax is monotonic, **arg-max on logits and arg-max on softmax probabilities always agree**, so an explicit softmax call is unnecessary for a single most-likely-class prediction (though it *would* be needed to report a confidence score).
4. **Inverse transform** — maps the integer class index back to its string label (`cat`/`dog`/`wild`) via the same `LabelEncoder` fit during data preparation.

## 15. Design Rationale: Advantages & Disadvantages

### Advantages of this architecture

- **Simplicity & interpretability**: every layer's purpose and output shape can be traced by hand (Section 9), making the model easy to debug, extend, and teach from.
- **Low compute cost**: ~339 MFLOPs/image and ~140 MB peak training memory (Section 10) mean it trains and infers comfortably on CPU — no GPU required.
- **"Same"-padding convolutions**: cleanly separate feature extraction (conv) from downsampling (pool), making the spatial-shrinkage schedule easy to reason about and modify independently of kernel design.
- **Adam + small LR**: robust, low-maintenance optimization that converges without hand-tuned learning-rate schedules.

### Disadvantages / risks

- **Parameter-inefficient classifier head**: 97.8% of parameters live in one `Linear(32768, 128)` layer — this is both a memory/compute bottleneck and, because it has so many free parameters relative to the ~11,000 training images, a primary driver of the overfitting observed after epoch 7 ([Section 16](#16-alternative-approaches) covers Global Average Pooling as the standard fix).
- **No regularization**: no `Dropout`, no `BatchNorm`, no weight decay, and no data augmentation are used anywhere in the pipeline. Combined with the oversized FC layer, this leaves the model with very little defense against overfitting, exactly as observed in the training curves.
- **No learning-rate schedule / early stopping**: the model trains for a fixed 10 epochs regardless of validation performance, so the *final* checkpoint (epoch 10) is not the *best* checkpoint (epoch 7) — a real, measurable cost in deployed accuracy if the last-epoch weights are what get shipped.
- **Fixed 128×128 input**: any deployment-time image must be resized, which discards information for higher-resolution inputs and cannot be changed without also changing the hard-coded `128*16*16` flatten dimension.
- **No normalization of pixel values** beyond the natural [0,1] range `ToTensor()` provides (no mean/std standardization) — this can slow convergence and makes the model more sensitive to global brightness/contrast shifts between training and deployment images than a normalized pipeline would be.

## 16. Alternative Approaches

| Alternative | What it changes | Trade-off vs. this implementation |
|---|---|---|
| **Global Average Pooling** (`nn.AdaptiveAvgPool2d(1)`) instead of `Flatten` + big `Linear` | Reduces `(B,128,16,16)` directly to `(B,128)` by averaging each channel spatially, eliminating the 4.19M-parameter `Linear(32768,128)` entirely | Cuts total parameters by ~98%, sharply reduces overfitting risk and memory; costs a small amount of spatial-detail sensitivity in the final representation |
| **Batch Normalization** after each conv | Normalizes each mini-batch's activations to zero mean/unit variance before the nonlinearity | Faster, more stable convergence and mild regularization; adds a small amount of compute and 2 extra learnable parameters per channel, and behaves differently in train vs. eval mode (a common source of bugs if `model.eval()` is forgotten) |
| **Dropout** in the FC head | Randomly zeroes a fraction of activations during training | Directly targets the overfitting seen after epoch 7; adds no inference-time cost (disabled at eval time) but adds a hyperparameter (drop probability) to tune |
| **Data augmentation** (random flip, rotation, color jitter, random crop) | Synthetically expands the effective size/diversity of the training set | Usually the single highest-leverage change for closing a train/val accuracy gap like the one observed here; costs extra CPU time per batch for the augmentation ops |
| **Learning-rate scheduling** (e.g., `ReduceLROnPlateau`, cosine annealing) + **early stopping** | Reduces LR or halts training based on validation performance rather than a fixed epoch count | Directly captures the "best model was epoch 7" finding automatically; adds implementation complexity and an extra hyperparameter (patience) |
| **Transfer learning** (e.g., pretrained ResNet18/EfficientNet-B0 backbone, fine-tuned) | Reuses ImageNet-pretrained convolutional features instead of learning them from scratch | Typically much higher accuracy with less training data and fewer epochs, since low-level features (edges, textures) transfer well across natural-image domains; costs a larger model (more parameters/compute), a dependency on pretrained weights, and less architectural transparency |
| **Deeper networks with residual connections** (ResNet-style) | Adds skip connections `y = F(x) + x` around conv blocks | Enables much deeper networks without vanishing-gradient degradation; adds architectural and implementation complexity not needed at this model's depth (3 conv layers rarely suffer vanishing gradients in the first place) |
| **Weight decay / L2 regularization** (`optimizer = Adam(..., weight_decay=...)`) | Penalizes large weights in the loss function | Cheap, one-line regularization; less targeted than dropout/augmentation but easy to add on top of the existing optimizer call |

## 17. Known Limitations & Future Work

- [ ] Replace `Flatten → Linear(32768,128)` with Global Average Pooling to cut parameters ~98% and reduce overfitting.
- [ ] Add data augmentation (random horizontal flip is a natural fit for animal faces, plus mild rotation/color jitter).
- [ ] Add `BatchNorm2d` after each convolution.
- [ ] Add early stopping / checkpoint the best validation epoch (epoch 7 in the observed run) instead of always shipping the final epoch's weights.
- [ ] Stratify the train/val/test split explicitly by class rather than relying on random sampling to preserve class balance.
- [ ] Add per-class precision/recall/F1 and a confusion matrix, not just aggregate accuracy — aggregate accuracy can mask class-specific weaknesses.
- [ ] Normalize pixel values with dataset mean/std (or standard ImageNet statistics if later adopting a pretrained backbone).
- [ ] Move training to GPU (`device = "cuda"`) if available, to enable larger batch sizes and faster experimentation.

## 18. How to Reproduce

```bash
# 1. Install dependencies (Section 4)
pip install torch torchvision torchsummary pandas numpy scikit-learn matplotlib pillow opendatasets

# 2. Launch the notebook
jupyter notebook Image_Classification.ipynb

# 3. Run all cells top to bottom:
#    - Cell 1 downloads the dataset via the Kaggle API (credentials required)
#    - Subsequent cells build the dataframe, split the data, define the Dataset/Model,
#      train for 10 epochs, evaluate on the test set, and run single-image inference.
```
