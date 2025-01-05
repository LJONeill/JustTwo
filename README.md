# Machine Learning Project

# Fashion Image Classification with Machine Learning

## Project Overview

This project aims to classify fashion images into distinct categories using machine learning techniques. We implemented models from scratch, including Decision Trees and Feed-forward Neural Networks, and also utilized established models from libraries like scikit-learn and TensorFlow.

## Dataset

We used the Fashion-MNIST dataset, which consists of 70,000 grayscale images of fashion products, each sized at 28x28 pixels. The dataset is divided into 60,000 training images and 10,000 testing images, categorized into 10 classes. :contentReference[oaicite:0]{index=0}

## Getting Started

### Dependencies

These versions were selected to ensure all libraries work together smoothly:

1. **Python**: 3.11.9  
   Compatible with all chosen libraries.

2. **Numpy**: 1.24.3  
   Required for scikit-learn 1.2.2 and TensorFlow 2.18.0.

3. **Scikit-Learn**: 1.2.2  
   Works well with numpy 1.24.x. Later versions need newer numpy, which isn’t compatible with TensorFlow 2.18.0.

4. **TensorFlow**: 2.18.0  
   Needs numpy 1.24.x to avoid issues with scikit-learn.

5. **Matplotlib**: 3.7  
   Matches numpy 1.24.x.

6. **Seaborn**: 0.13.2  
   Fully compatible with matplotlib 3.7.

7. **Mlxtend**: 0.23.3  
   Works with scikit-learn 1.2.2.

---

### Installing

Set up the environment using:

**With Conda**:
```bash
conda create -n ml_project_env python=3.11 numpy=1.24.3 scikit-learn=1.2.2 tensorflow=2.18.0 matplotlib=3.7 seaborn=0.13.2 mlxtend=0.23.3
conda activate ml_project_env
```


## References

1. **Feed-Forward Neural Network (FFNN) Model**: We referenced [this Colab notebook](https://colab.research.google.com/drive/16w3TDn_tAku17mum98EWTmjaLHAJcsk0?usp=sharing) for the implementation of a feed-forward neural network using TensorFlow. The notebook served as a guide for structuring our model, including defining layers, compiling the model, training, validation, and evaluation processes.

2. **Data Preprocessing and Model Evaluation**: We referenced [this Colab notebook](https://colab.research.google.com/drive/1m2cg3D1x3j5vrFc-Cu0gMvc48gWyCOuG#forceEdit=true&sandboxMode=true&scrollTo=jqVqT_Cxh4Ho) for guidance on data preprocessing, splitting the dataset into train-validation-test sets, and model evaluation techniques.

3. **Neural Network Implementations in Python**: We utilized resources from [this GitHub repository](https://github.com/andresberejnoi/PublicNotebooks/tree/master/Neural%20Networks), which contains scripts and notebooks related to building neural networks from scratch using Python and NumPy. The repository includes implementations of feedforward propagation and backpropagation algorithms, which were instrumental in understanding the underlying mechanics of neural networks.

4. **Decision Tree Implementation from Scratch**:  We referenced [this GitHub file](https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/decision_tree.py), which contains an implementation of a decision tree algorithm built from scratch using Python. This resource provided insights into the construction and functionality of decision trees, aiding in our understanding and application of this machine learning technique.




