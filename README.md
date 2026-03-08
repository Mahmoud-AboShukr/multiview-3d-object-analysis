# Multi-View 3D Object Classification and Retrieval

This project explores **3D object understanding from mesh data** using a multi-view learning pipeline built on the **ModelNet10** dataset. The workflow starts from raw `.off` CAD meshes, renders multiple 2D views for each object, and then applies deep learning models for both **classification** and **content-based retrieval**.

The project includes two complementary approaches:

- a **single-view / aggregated-view baseline** using a ResNet18 backbone trained on rendered object views
- an **MVCNN-style multi-view architecture** that jointly processes multiple rendered views of the same 3D object

Beyond classification, the project extracts learned descriptors and evaluates them for **similarity-based retrieval** using metrics such as Top-k Accuracy, Precision@K, Recall@K, and mAP.

## Features

- Loads and organizes **ModelNet10** mesh data from `.off` files
- Renders multiple viewpoints per object using **Trimesh** and **Pyrender**
- Builds a 2D image dataset from 3D meshes
- Trains a **ResNet18** classifier on rendered views
- Extracts object-level descriptors from learned visual features
- Performs **3D object retrieval** via cosine similarity
- Implements and trains an **MVCNN** model for multi-view object recognition
- Evaluates retrieval with standard ranking metrics
- Visualizes query objects and top retrieved examples

## Project Structure

```text
.
├── multiview-3d-object-analysis.ipynb
├── README.md
├── requirements.txt
├── ModelNet10/
├── views/
├── views_test/
├── resnet18_views_modelnet10.pth
├── train_X.npy
├── train_Y.npy
├── test_X.npy
├── test_Y.npy
├── mv_train_X.npy
├── mv_train_Y.npy
├── mv_test_X.npy
└── mv_test_Y.npy
```

## Workflow

### 1. Mesh loading and inspection
The notebook begins by loading `.off` mesh files from the ModelNet10 dataset and checking that the directory structure is correct.

### 2. Multi-view rendering
Each 3D mesh is rendered from several viewpoints into 2D RGB images. These views are stored in a structured folder hierarchy for train and test splits.

### 3. View-based classification baseline
A pretrained **ResNet18** is adapted for classification on rendered views. The final classification layer is replaced to match the selected ModelNet10 classes.

### 4. Descriptor extraction
The trained model is converted into a feature extractor. Features from multiple views of the same object are pooled into a single descriptor.

### 5. Retrieval evaluation
Object descriptors are compared using **cosine similarity**, and ranking metrics are computed:

- Top-1 / Top-5 / Top-10 Accuracy
- Precision@K
- Recall@K
- Mean Average Precision (mAP)

### 6. MVCNN extension
An **MVCNN** variant is implemented to aggregate information across multiple views before classification, enabling a more native multi-view learning setup.

## Dataset

This project uses the **ModelNet10** dataset, a benchmark collection of 3D CAD models organized by object class.

Typical classes used in the notebook include:

- chair
- table
- sofa
- bed
- desk

The notebook expects the dataset in a structure similar to:

```text
ModelNet10/
└── ModelNet10/
    ├── chair/
    │   ├── train/
    │   └── test/
    ├── table/
    └── ...
```

## Installation

Create and activate a virtual environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

## How to Run

1. Download and place the **ModelNet10** dataset in the expected local directory.
2. Open the notebook:

```bash
jupyter notebook 3D_data_analysis.ipynb
```

3. Run the notebook cells in order to:
   - inspect meshes
   - generate rendered views
   - train the baseline model
   - extract descriptors
   - evaluate retrieval
   - train and evaluate the MVCNN model

## Outputs

Depending on which notebook sections are executed, the project generates:

- rendered 2D object views
- trained model weights
- saved feature descriptors in `.npy` format
- retrieval metrics
- qualitative retrieval visualizations

## Notes

- The notebook was developed in an experimental, interactive format, so some installation commands are included inside notebook cells.
- Rendering 3D meshes off-screen may require additional graphics compatibility depending on the operating system and environment.
- Some saved artifacts such as generated views, model weights, and descriptor arrays can become large; they may be better excluded from GitHub and regenerated locally.

## Suggested `.gitignore` entries

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/
env/
views/
views_test/
*.npy
*.pth
.DS_Store
Thumbs.db
```

## Author

Personal 3D data analysis project focused on **multi-view object classification and retrieval from 3D meshes**.
