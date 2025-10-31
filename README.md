# California Housing Price Prediction with Neural Networks

**Can a Multi-Layer Perceptron beat Random Forest and Linear Regression at predicting house prices?**

## Motivation
I’m currently studying computer science at university, where I’m taking a Machine Learning module. My lecturer’s engaging teaching really got me interested in the topic. ML seemed a little daunting at first but the way he explained complex concepts made everything click and genuinely interesting.   
After learning about neural networks in class, I was fascinated by the maths behind them and how they could identify complex non-linear relationships in data. However, lectures and coursework only go so far, I wanted to apply them to a real-word problem.

I came across NeuralNine’s YouTube video titled “House Price Prediction in Python - Full Machine Learning Project”. He trained two models where Linear Regression achieved **R2 = 0.67** and Random Forest achieved **R2 = 0.81**.  
The random forest model outperformed the linear model by quite a margin, which left me wondering how well could a neural network do and if it could it do better? 

## Dataset
**California Housing Dataset** (1990 Census)  
- 20,640 samples  
- 8 features  
- Target variable: `median_house_value`  
- Challenge: 207 missing values in `total_bedrooms`, extremely high correlation between some features  

## Method
### Exploratory Data Analysis:
- Histogram of target variable  
- Boxplots for all features to spot outliers  
- Correlation matrix to check feature relationships

### Preprocessing:
- •	Impute the missing values with the median of the column  
- •	One-hot encoding for `ocean_proximity`  
- Random 80/20 train-test split  
- `StandardScaler` to normalise both features **and** target

### Hyperparameter Tuning:
- Tested **48 total configurations**  
  - 6 hidden layer structures  
  - 2 activation functions (`relu`, `tanh`)  
  - 4 learning rates (`0.0001`, `0.001`, `0.01`, `0.1`)  
- Implemented **5-fold cross-validation manually**

### Best Configuration:
- Hidden layer sizes: **(100, 50, 25)**  
- Activation function: **0.001**  
- Learning rate: **tanh**

## Results
|       Model       |  R²  | 
|-------------------|------|
| Linear Regression | 0.67 |
| my MLP            | 0.78 |
| Random Forest     | 0.81 |

**MSE: 2913470848.540982**  
**RMSE: $53976.58**

**Key finding:** My neural network beat the Linear Regression model by **11 percentage points**, but fell just **3 points short of Random Forest**.

## Challenges I Encountered
### Understanding when to encode categorical variables  
I was confused about whether to one-hot encode before or after the train-test split. I tried to search the web to find out but got even more confused about the idea of data leakage.  
I learnt that one-hot encoding before the split is fine, but scaling must be done after because encoding  just creates columns from an existing feature, but scaling learns information about the data so must be done after the split to prevent the model gaining insight into the test data.

### Target scaling
Initially, MSE values were about 4 billion, knew there was probably something wrong - I didn’t scale the target variable.  
Applied StandardScaler to y_train and y_test separately, then inverse transformed y_pred.  
I learnt that target scaling is crucial for neural network stability because they struggle with large output values that haven’t been scaled.  

### Manually implementing the hyperparameter search loop
We covered K-fold cross-validation in lectures, but I still had to work out how to automate the process of testing different parameter combinations. I didn’t know then that libraries like scikit-learn already provide tools for this.  
I decided that my hyperparameters would be the structure of the hidden layers, which activation function to use and the learning rate. 
I wrote three nested loops that automatically test all combinations, output the best combination for a given hidden layer structure and then at the end output the overall best MSE and the configuration that achieved it.  
I learnt how to structure hyperparameter searches and ensured that new configurations can be added for testing by just adding new elements to lists.  

### Finding best configuration per architecture by understanding the results data structure
After testing 8 combinations for each hidden layer structure, the results list grows with each configuration test.  
I didn’t know how to find the minimum MSE. I calculated indexes and used manual slicing based on number of combinations per hidden layer structure to access the right part of the list containing that iteration’s set of hyperparameter combinations to retrieve the best configuration.  
I learnt that manual slicing works but is an over-engineered way to accomplish the task. I learnt that you could use lambda functions for finding the minimum, but I decided not to implement it this way because I’m just not too familiar with them.  

### Confusion about cross-validation vs. test set
I was uncertain about when to use the test set.  
I learnt to never touch test set until final evaluation.  
Use CV on training data for hyperparameter tuning.  

*Little note:* I used to be so confused when I heard “my model took x amount of hours to train”, now I understand that often, they’re probably talking about hyperparameter tuning. It takes a surprisingly long time (even with my limited number of configurations).

## Conclusion
So, can a neural network beat Linear Regression and Random Forest on this dataset?  
**Short answer:** Not quite — but it came close.  

Before this project, I thought “NNs are the most advanced, so they should win”. But I’ve found that it’s more a case of “NNs are powerful tools designed for specific problems.”  

My MLP achieved **R² = 0.78**, which fell short of the Random Forest (**0.81**) by about 3%, but beat Linear Regression (**0.67**) by around 11%.

### Why my NN beat Linear Regression
Linear regression can only model straight-line relationships. If the relationship between house price and features is non-linear, linear regression will underperform.  
NNs, on the other hand, can cope with non-linearity because of activation functions such as ReLU and tanh. 


### But why didn’t it beat Random Forest?
1. **Hyperparameter Tuning**  
   - NeuralNine used GridSearch which tested 100s more combinations than I did and optimised for Random Forest specifically. 
   - I manually implemented hyperparameter search and that too, only tested 48 combinations.  

With more thorough hyperparameter tuning, my MLP may have gained a couple R² points, maybe matching or even beating Random Forest.  

2. **Algorithm Suitability**  
   - Random Forests are literally better at tabular data than NNs. 
   - They work with both numerical and categorical data.
   - Tree splits aren’t affected by extreme values so no scaling is required.
   - We can easily calculate how much information is retained at each split.

   - NNs need a lot of preprocessing (scaling, encoding and normalisation).
   - Very sensitive to hyperparameters.
   - It’s harder to find out what the model actually learned or what patterns it recognised.

  
## What I Would Do Differently
- Use an automatic function to (`GridSearchCV` or `RandomizedSearchCV`) to find the best hyperparameter configuration
- Create new features from the ones already existing. E.g. rooms per household, bedrooms per total rooms, population per household

## Tech Stack
- **Language:** Python 3.11 (Anaconda virtual environment)  
- **Core Libraries:**  
  - numpy – numerical operations  
  - pandas – data handling  
  - matplotlib, seaborn – visualisation  
  - scipy – statistical analysis  
  - scikit-learn – preprocessing, model selection, evaluation  
- **Environment:** Jupyter Notebook

## Project Structure
```
california-housing-mlp
 ├── california_housing_mlp.ipynb      ← main notebook
 ├── housing.csv                       ← dataset
 ├── README.md                         ← documentation
```

## How to Run
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/california-housing-mlp.git
   cd california-housing-mlp
   ```
2. **Create and activate a new environment:**
   ```
   conda create -n housing-mlp python=3.11`
   conda activate housing-mlp
   ```
4. **Install dependencies:**
   `pip install numpy pandas matplotlib seaborn scipy scikit-learn`
5. **Run the notebook:**
   `jupyter notebook`
6. **Then open `california_housing_mlp.ipynb` and run all cells.** The dataset is read as CSV in the first cell of the notebook.
