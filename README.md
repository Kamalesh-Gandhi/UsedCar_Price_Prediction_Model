# Car Dekho - Used Car Price Prediction using Machine Learning
This project develops a machine learning model to predict used car prices for CarDekho, integrating it into a streamlit web application. The process includes data cleaning, feature selection, and model evaluation, optimization. The final product is a user-friendly tool that provides instant price estimates based on car details.

## Introduction
Car Dekho - Used Car Price Prediction is a machine learning-based project aimed at predicting the prices of used cars. The platform is designed for evaluating used car prices, empowering buyers, sellers, and dealers to make informed decisions. This project utilizes machine learning for price prediction and Streamlit for interactive web application development. By analysing car-related data, the application offers instant price estimates, enhancing transparency and efficiency in the used car market.

---

## Project Scope
Based on the historical data on used car prices from CarDekho, including various features such as make, model, year, fuel type, transmission type, and other relevant attributes from different cities. The task is to develop a machine learning model that can accurately predict the prices of used cars based on these features. The model should be integrated into a Streamlit-based web application to allow users to input car details and receive an estimated price instantly.

---

## Packages Used
### Pandas:
- Pandas is a powerful and open-source Python library. The Pandas library is used for data manipulation and analysis.
- Pandas consist of data structures and functions to perform efficient operations on data.
- To know more about Pandas, [click here](https://pandas.pydata.org/docs/).

### Streamlit:
- Streamlit turns data scripts into shareable web apps in minutes. All in pure Python. No front-end experience required.
- To know more about Streamlit, [click here](https://docs.streamlit.io/).

### Streamlit-option-menu:
- Streamlit-option-menu is a simple Streamlit component that allows users to select a single item from a list of options in a menu.
- To know more about Streamlit-Option-Menu, [click here](https://discuss.streamlit.io/t/streamlit-option-menu-is-a-simple-streamlit-component-that-allows-users-to-select-a-single-item-from-a-list-of-options-in-a-menu/20514).

### Matplotlib:
- Matplotlib is a powerful data visualization library for Python. It provides a wide range of tools for creating static, animated, and interactive plots, making it a popular choice for visualizing data in fields like data science, machine learning, and scientific research.
- To know more about Matplotlib, [click here](https://matplotlib.org/).

### Seaborn:
- Seaborn is a Python data visualization library based on Matplotlib. It provides a high-level interface for drawing attractive and informative statistical graphics.
- To know more about Seaborn, [click here](https://seaborn.pydata.org/).

### Scikit-learn:
- Simple and efficient tools for predictive data analysis.
- Accessible to everybody and reusable in various contexts.
- Built on NumPy, SciPy, and Matplotlib.
- To know more about Scikit-learn, [click here](https://scikit-learn.org/stable/).

### NumPy:
- NumPy is the fundamental package for scientific computing in Python.
- It is a Python library that provides a multidimensional array object, various derived objects, and an assortment of routines for fast operations on arrays, including mathematical, logical, shape manipulation, sorting, and more.
- To know more about NumPy, [click here](https://numpy.org/doc/stable/).

### SciPy:
- SciPy is a scientific computation library that uses NumPy underneath.
- SciPy stands for Scientific Python.
- It provides more utility functions for optimization, stats, and signal processing.
- To know more about SciPy, [click here](https://docs.scipy.org/doc/scipy/).

### XGBoost:
- XGBoost, or Extreme Gradient Boosting, is an open-source machine learning library that uses gradient boosted decision trees (GBDTs) to train models for regression, classification, and ranking tasks.
- To know more about XGBoost, [click here](https://xgboost.readthedocs.io/en/stable/install.html).

### Joblib:
- Joblib is a set of tools to provide lightweight pipelining in Python.
- To know more about Joblib, [click here](https://joblib.readthedocs.io/en/stable/).

---

## Install Packages
To install the required Python libraries, use the following commands:
- `pip install numpy`
- `pip install pandas`
- `pip install streamlit`
- `pip install streamlit-option-menu`
- `pip install joblib`
- `pip install scipy`
- `pip install seaborn`
- `pip install scikit-learn`
- `pip install matplotlib`
- `pip install xgboost`

---

## Code Flow Plan
This project contains the following seven files:

1. **Data_FormattingProcess.py**  
   - The script preprocesses datasets from six cities and merges them into a single file (`CombindCars.csv`).
   - It imports necessary libraries: Pandas for data manipulation, NumPy for handling missing values, and AST for parsing strings into dictionaries.
   - City-specific datasets are loaded from Excel files using `pandas.read_excel`.
   - Nested data in columns is flattened using dedicated functions for car details, features, overview, and specifications.
   - Each flattening function extracts key information into a structured format.
   - The function processes data for each city, applying flattening functions and concatenating results with city-specific links.
   - Individual processed datasets are saved as CSV files for each city in the Datasets directory.
   - The `combine_files` function merges all city-specific CSV files into a single structured file named `Combined_Cars_Data.csv`.

2. **Data_PreProcessing.ipynb**  
   - The notebook imports libraries like Pandas, Matplotlib, Seaborn, and SciPy for data manipulation, visualization, and statistical analysis.
   - Reads the `CombinedCars.csv` file and displays dataset info, column names, and initial size for understanding the structure.
   - Drops unnecessary or duplicate columns after checking for redundancies and standardizes column names.
   - Formats columns by removing extra characters (e.g., cc, @rpm, kms) and ensures numeric data is converted to appropriate data types.
   - Handles missing values by dropping columns/rows with more than 50% nulls and imputing others using statistical methods.
   - Detects and corrects misspelled categorical values to maintain consistency.
   - Performs exploratory data analysis (EDA) using visualizations and detects outliers using methods like IQR and Z-scores.
   - Detects outliers using methods like IQR and Z-scores.
   - Handled only Technically True Outliers Based on the Domain knowledge
   - Saved the Technically outliers handled data to `DomainBased_Handled_data.csv`

3. **Testing_HandlingOutliers.ipynb**  
   - The notebook imports libraries like Pandas, Matplotlib, Seaborn, and SciPy for data manipulation, visualization, and statistical analysis.
   - Reads the `Temp_preprocessed_data.csv` file and displays dataset info, column names, and initial size for understanding the structure.
   - Drops unnecessary or duplicate columns after checking for redundancies and standardizes column names.
   - Detects outliers using methods like  Z-scores.
   - Done Two Technique to handle Outliers i.e , Capping and Trimming Techniques
   - Done Correlation ,Compare to Capped Data ,the Trimming Data correlation is better .So Trimmed Data comcluded as Final Data for Model Training
   - Saved the handled data to `Model_TrainTest_data`

3. **CarPrice_Prediction_Model.ipynb**  
   - The dataset `Model_TrainTest_data.csv` is loaded, and its categorical and continuous columns are identified.
   - Categorical columns are encoded using `LabelEncoder`, and mappings are stored for reference.
   - The target variable `price` is separated from features, and the dataset is split into training and testing sets using an 80-20 ratio.
   - Multiple machine learning models are initialized and evaluated (e.g., Linear Regression, Decision Tree, Random Forest, XGBoost).
   - Hyperparameter tuning is performed using `RandomizedSearchCV` for the best-performing models.
   - Random Forest and XGBoost are identified as the best-performing models based on Mean Absolute Error (MAE) and R² scores.
   - Hyperparameter tuning is performed using RandomizedSearchCV for both models to optimize parameters such as n_estimators, max_depth, and learning_rate.
   - Feature importance is plotted for Random Forest and XGBoost to analyze which features significantly impact predictions.
   - The best XGBoost model achieves high R² and low MAE, with parameters including n_estimators=200, max_depth=7, and learning_rate=0.05
   - The trained XGBoost model is saved as `xgboost_best_model.pkl` for deployment.
   - The notebook concludes by preparing the model for integration into the Streamlit application using the best parameters.
   - Save the model (xgboost_best_model.pkl) and encoded mapping (categorical_encoders.pkl) using joblib 

4. **Streamlit_UI.py**  
   - Configures a Streamlit application with a title, icon, and wide layout for car price prediction.
   - Loads the dataset (`Temp_preprocessed_data.csv`), encoded mappings, and the trained machine learning model.
   - Allows users to input car details via dropdowns and numeric input fields.
   - Predicts the car price using the trained model and displays the result with formatted styling.

5. **Customization_styles.css**  
   - Contains custom CSS for styling the Streamlit application, enhancing the visual experience.

6. **categorical_encoders.pkl**  
   - Stores the encoded mappings used for categorical data transformation.

7. **xgboost_best_model.pkl**  
   - The trained machine learning model ready for deployment.

---

## How to Run the Code
1. Run `Data_FormattingProcess.ipynb` by selecting **Run All**. .  
2. Run `Data_PreProcessing.ipynb` by selecting **Run All**.  
3. Run `Testing_HandlingOutliers.ipynb` by selecting **Run All**.  
4. Run `Streamlit_UI.py` using the command: `streamlit run Streamlit_UI.py`.  

---

## Model Score
- **Score when training the model using default parameters:**  
  ![Accuracy_Before_Tuning](https://github.com/user-attachments/assets/10727fc0-6f36-4c08-83cf-fee3e00f5e18)

- **Score after training the model with the best parameters:**  
 ![Accuracy_After_Tuning](https://github.com/user-attachments/assets/b42b5897-1170-453a-b9c4-3c0458473854)

---

## Streamlit UI
![Stream UI Before Prediction](https://github.com/user-attachments/assets/964d83b1-3fff-4375-a9e7-cd5e4eab0b57)

![Stream UI After Prediction](https://github.com/user-attachments/assets/153837c6-9afa-4561-9fe8-9e9c805a681f)


