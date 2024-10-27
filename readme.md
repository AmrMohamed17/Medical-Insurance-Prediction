# Insurance Cost Prediction Analysis

This project analyzes medical insurance costs and builds predictive models using both Linear Regression and Artificial Neural Networks (ANN). The analysis includes data preprocessing, exploratory data analysis, feature engineering, and model comparison.

## Table of Contents
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Implementation Details](#implementation-details)
- [Model Performance](#model-performance)
- [Usage](#usage)
- [Results and Insights](#results-and-insights)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [Future Improvements](#future-improvements)
- [License](#license)
- [Contact](#contact)

## Requirements
```
numpy
pandas
seaborn
matplotlib
scikit-learn
tensorflow
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/AmrMohamed17/insurance-prediction.git
cd insurance-prediction
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset
The project uses the 'insurance.csv' dataset containing the following features:
- age: Age of primary beneficiary
- sex: Insurance contractor gender (female/male)
- bmi: Body mass index
- children: Number of covered dependents
- smoker: Smoking status (yes/no)
- region: Beneficiary's residential area (northeast, southeast, southwest, northwest)
- charges: Individual medical costs billed by health insurance (target variable)

### Data Source
The dataset can be downloaded from [Kaggle's Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance).

## Project Structure
1. **Data Loading and Initial Analysis**
   - Import required libraries
   - Load and examine the dataset
   - Check for missing values
   - Display basic statistics

2. **Feature Engineering**
   - Convert categorical variables (sex, smoker) to binary
   - One-hot encode region variable
   - Scale features using StandardScaler

3. **Data Visualization**
   - Distribution plots for all features
   - Pairwise relationships visualization
   - Correlation analysis with heatmap
   - Regression plots for each feature vs charges

4. **Model Implementation**
   - Linear Regression Model
   - Artificial Neural Network (ANN) Model

## Implementation Details

### Data Preprocessing
- Feature scaling using StandardScaler
- Train-test split (80-20)
- One-hot encoding for categorical variables

### ANN Architecture
```
Layer 1: Dense(50, input_dim=9, activation='relu')
Layer 2: Dense(150, activation='relu')
Layer 3: Dropout(0.5)
Layer 4: Dense(150, activation='relu')
Layer 5: Dropout(0.5)
Layer 6: Dense(50, activation='linear')
Layer 7: Dense(1)
```

### Training Parameters
- Optimizer: Adam
- Loss Function: Mean Squared Error
- Epochs: 100
- Batch Size: 20
- Validation Split: 0.2

## Model Performance
The models are evaluated using multiple metrics:
- Mean Absolute Error (MAE)
- R-squared (R²)
- Adjusted R-squared

Performance visualization includes:
- Training vs Validation loss curves
- Predicted vs Actual values scatter plot

## Usage

### Basic Usage
1. Ensure all required libraries are installed
2. Place 'insurance.csv' in the project directory
3. Run the script sections sequentially
4. Model outputs and visualizations will be generated automatically

### Advanced Usage
```python
# Load and preprocess new data
new_data = pd.read_csv('new_insurance_data.csv')
processed_data = preprocess_data(new_data)

# Make predictions using trained models
linear_predictions = regression_model.predict(processed_data)
ann_predictions = ANN_model.predict(processed_data)
```

## Results and Insights

### Key Findings
1. Feature Importance
   - Smoking status is the strongest predictor of insurance charges
   - BMI and age show moderate correlation with charges
   - Region has minimal impact on insurance costs

2. Model Comparison
   - ANN model generally outperforms Linear Regression
   - Linear Regression provides better interpretability
   - Neural Network shows better handling of non-linear relationships

### Visualization Highlights
- Correlation heatmap reveals key relationships between features
- Distribution plots show clear patterns in demographic data
- Regression plots identify non-linear relationships

## Contributing
We welcome contributions to improve the project! Here's how you can help:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Commit your changes (`git commit -am 'Add new feature'`)
5. Push to the branch (`git push origin feature/improvement`)
6. Create a Pull Request

### Coding Standards
- Follow PEP 8 guidelines
- Include docstrings for new functions
- Add comments for complex logic
- Write unit tests for new features

## Troubleshooting

### Common Issues
1. Missing Dependencies
```bash
pip install -r requirements.txt
```

2. Memory Issues
- Reduce batch size
- Use data generators for large datasets

3. Model Convergence
- Adjust learning rate
- Modify network architecture
- Implement early stopping

## Future Improvements
1. Model Enhancements
   - Implement feature selection
   - Try different architectures (LSTM, CNN)
   - Add cross-validation

2. Features to Add
   - API endpoint for predictions
   - Web interface for model interaction
   - Additional visualization options
   - Model explainability tools

3. Infrastructure
   - Docker containerization
   - CI/CD pipeline
   - Model versioning
   - Automated testing

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact
For questions and support:
- Create an issue in the repository
- Email: your.email@example.com
- Project Link: https://github.com/yourusername/insurance-prediction

Note: The neural network's performance may vary slightly between runs due to random weight initialization and dropout layers.
