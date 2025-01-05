# Machine Learning Project

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
   Works seamlessly with scikit-learn 1.2.2.

---

### Installing

Set up the environment using:

**With Conda**:
```bash
conda create -n ml_project_env python=3.11 numpy=1.24.3 scikit-learn=1.2.2 tensorflow=2.18.0 matplotlib=3.7 seaborn=0.13.2 mlxtend=0.23.3
conda activate ml_project_env
