# Kolmogorov-Arnold Network (KAN) - Analysis Project

## Description
This project implements and analyzes the Kolmogorov-Arnold Network (KAN), an innovative type of neural network based on the Kolmogorov-Arnold theorem. The project includes practical implementations and comparisons with other machine learning approaches.

## Requirements
- Python 3.9.6
- Main dependencies:
````text
matplotlib==3.6.2
numpy==1.24.4
scikit_learn==1.1.3
torch==2.2.2
pandas==2.0.1
````

## Installation

1. Clone the repository:
````bash
git clone https://github.com/xtreamsrl/KAN
cd KAN
````

2. Install dependencies:
````bash
pip install -r requirements.txt
````

## Project Structure

### Main Notebooks
1. **kan_presentation.ipynb**: A comprehensive introduction to the Kolmogorov-Arnold Network (KAN) and its applications
2. **kaggle_vs_kan.ipynb**: Demonstration of using KAN to solve a Kaggle competition

### Key Features
- KAN implementation using PyTorch
- Network simplification techniques:
  - Sparsification
  - Refinement
  - Network pruning
- Learning visualization through animations
- Comparison with traditional neural networks

## Core Functionality

### Model Creation
````python
model = KAN(width=[2,5,1], grid=3, k=3, device=device)
````

### Training
````python
_ = model.fit(dataset, opt="LBFGS", steps=150)
````

### Network Optimization
````python
model = model.prune()  # Pruning
model = model.refine(10)  # Refinement
````

## Visualizations
The project includes functionality to generate dynamic visualizations of network learning, saved as MP4 videos.

## References
For in-depth tutorials and video content, visit:
````text
https://www.youtube.com/@umarjamilai/videos
````

## Notes
- Project is optimized for both CPU and CUDA GPU execution
- Includes implementations for both regression and classification problems
- Provides tools for comparative analysis with other machine learning models

## License
This project is distributed under an open source license. See LICENSE file for details.