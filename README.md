
# Graduate Admission Prediction

This project predicts a student's chance of admission to a graduate program based on various academic factors. A neural network model is trained on a dataset of past applicants to make these predictions.

## Table of Contents

- [About the Dataset](#about-the-dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## About the Dataset

The dataset used in this project is `Admission_Predict_Ver1.1.csv`. It contains the following features:
- **GRE Score:** Graduate Record Examination score (out of 340).
- **TOEFL Score:** Test of English as a Foreign Language score (out of 120).
- **University Rating:** Rating of the university (out of 5).
- **SOP:** Statement of Purpose strength (out of 5).
- **LOR:** Letter of Recommendation strength (out of 5).
- **CGPA:** Cumulative Grade Point Average (out of 10).
- **Research:** Research experience (0 or 1).
- **Chance of Admit:** The target variable, representing the probability of admission (ranging from 0 to 1).

## Installation

To run this project, you need to have Python and the following libraries installed:

- pandas
- numpy
- matplotlib
- seaborn
- tensorflow
- scikit-learn

You can install these libraries using pip:

```bash
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn
```

## Usage

1. Clone this repository:

```bash
git clone https://github.com/your-username/Graduate-admission-prediction.git
```

2. Navigate to the project directory:

```bash
cd Graduate-admission-prediction
```

3. Run the Jupyter Notebook:

```bash
jupyter notebook prediction.ipynb
```

## Model

The prediction model is a sequential neural network built using TensorFlow/Keras. The architecture is as follows:

- **Input Layer:** 7 neurons (corresponding to the 7 features).
- **Hidden Layer 1:** 7 neurons with ReLU activation.
- **Output Layer:** 1 neuron with a linear activation function to predict the chance of admission.

The model is compiled with the Adam optimizer and uses mean squared error as the loss function.

## Results

The model was trained for 100 epochs and evaluated on a test set. The R-squared score achieved was approximately **0.81**, indicating a good fit to the data.

The training and validation loss over epochs is shown in the following plot:

![Loss Plot](https---.png)

## Contributing

Contributions are welcome! If you have any suggestions or find any issues, please open an issue or create a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
