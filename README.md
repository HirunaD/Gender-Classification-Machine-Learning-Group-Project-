# Gender-Classification-Machine-Learning-Group-Project-

This repository contains a gender classification project implemented using two different machine learning algorithms: 

  1) K-Nearest Neighbors (KNN)
  2) Support Vector Machine (SVM)

The goal of this project is to classify gender based on various features provided in the dataset.

## Contents

- `Gender_Classification_KNN.ipynb`: Jupyter notebook for gender classification using the K-Nearest Neighbors algorithm.
- `Gender_Classification_SVM.ipynb`: Jupyter notebook for gender classification using the Support Vector Machine algorithm.
- `gender_classification.csv`: Dataset used for training and testing the models.

## Dataset

The dataset `gender_classification.csv` contains the following features:

- Height
- Weight
- Shoe Size
- Gender (Label)

## Requirements

To run the notebooks, you need to have the following packages installed:

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- Jupyter Notebook

You can install the required packages using pip:

```sh
pip install pandas numpy scikit-learn matplotlib jupyter
```

## Getting Started

1. **Clone the repository:**

```sh
git clone https://github.com/HirunaD/Gender-Classification-Machine-Learning-Group-Project-.git
cd gender-classification
```

2. **Open Jupyter Notebook:**

```sh
jupyter notebook
```

3. **Run the notebooks:**

- Open `Gender_Classification_KNN.ipynb` to see the implementation of the K-Nearest Neighbors algorithm.
- Open `Gender_Classification_SVM.ipynb` to see the implementation of the Support Vector Machine algorithm.

## Project Overview

### K-Nearest Neighbors (KNN)

The `Gender_Classification_KNN.ipynb` notebook covers:

- Data loading and preprocessing
- Splitting the dataset into training and testing sets
- Training a KNN classifier
- Evaluating the classifier's performance
- Visualizing the results

### Support Vector Machine (SVM)

The `Gender_Classification_SVM.ipynb` notebook covers:

- Data loading and preprocessing
- Splitting the dataset into training and testing sets
- Training an SVM classifier
- Evaluating the classifier's performance
- Visualizing the results

## Results

Both notebooks include sections for evaluating the performance of the classifiers using metrics such as accuracy, precision, recall, and F1-score. Visualizations are also provided to illustrate the classification results.

## Contributing

Contributions are welcome! If you have any suggestions or improvements, please create a pull request or open an issue.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgments

Special thanks to the authors of the libraries and tools used in this project.
