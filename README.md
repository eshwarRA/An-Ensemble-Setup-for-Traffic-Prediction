# An Ensemble Setup for Traffic Prediction Project

This project aims to predict traffic volume based on historical data using various machine learning and neural network models, including a Spiking Neural Network (SNN) and ensemble methods.

## 1. Data Loading and Preprocessing

The project starts by loading the `traffic.csv` dataset. The 'ID' column is removed, and the 'DateTime' column is parsed to extract 'Year', 'Month', 'Day', and 'Hour'. For the models, we focus on 'Junction', 'Month', 'Day', and 'Hour' as features to predict 'Vehicles'.

The data is then split into training and testing sets (80% train, 20% test).

## 2. Model Implementations

Several models were implemented and evaluated for this regression task:

-   **Simple Neural Network (ANN)**: A basic feedforward neural network trained using PyTorch.
-   **Decision Tree Regressor**: A tree-based model from scikit-learn.
-   **Random Forest Regressor**: An ensemble of decision trees from scikit-learn.
-   **K-Nearest Neighbors Regressor**: A non-parametric model from scikit-learn.
-   **Weighted K-Nearest Neighbors Regressor**: A variation of KNN where closer neighbors have more influence.
-   **Spiking Neural Network (SNN)**: A neural network model that uses spikes for computation, implemented using `snntorch`. Features were converted to spike trains using rate coding.
-   **Stacking Ensemble**: An ensemble method that uses the predictions of the individual models as input for a meta-model (Linear Regression in this case) to make the final prediction.
-   **Weighted Average Ensemble**: (Note: While discussed conceptually and included in the comparison logic, a dedicated code cell for training and evaluating the weighted average ensemble was not explicitly shown in the provided notebook snippets).

## 3. Model Evaluation

The performance of each model and the stacking ensemble was evaluated using the following regression metrics on the test set:

-   **Mean Squared Error (MSE)**: Measures the average squared difference between actual and predicted values.
-   **Root Mean Squared Error (RMSE)**: The square root of MSE, providing an error metric in the same units as the target variable.
-   **Mean Absolute Error (MAE)**: Measures the average absolute difference between actual and predicted values, less sensitive to outliers than MSE/RMSE.
-   **R-squared (R²)**: Represents the proportion of the variance in the dependent variable that is predictable from the independent variables.

The following table summarizes the performance metrics for each evaluated model and ensemble:

| Model                        | MSE       | RMSE      | MAE     |
|------------------------------|-----------|-----------|---------|
| Stacking Ensemble            | 87.8506   | 9.3729    | 6.2775  |
| Simple Neural Network (ANN)  | 147.3839  | 12.1402   | 8.2455  |
| K-Nearest Neighbors          | 196.4790  | 14.0171   | 9.5085  |
| Random Forest                | 250.1327  | 15.8156   | 10.1347 |
| Spiking Neural Network (SNN) | 204.1117  | 14.2868   | 10.1728 |
| Decision Tree                | 368.5233  | 19.1970   | 12.2186 |
| Weighted K-Nearest Neighbors | 376.2575  | 19.3974   | 12.8516 |

*(Note: The metrics in this table are based on the output of the final comparison cell and might slightly vary from individual model evaluation outputs due to potential minor differences in test set splitting or evaluation order, though the relative performance should be consistent.)*

## 4. Visualizations

Scatter plots were generated for each model to visualize the relationship between the actual and predicted traffic volumes on the test set. These plots help in understanding how well each model's predictions align with the true values.

Examples of these plots can be found in the notebook cells where each model and the stacking ensemble are evaluated (e.g., after the Simple Neural Network, Decision Tree, Random Forest, KNN, SNN, and Stacking Ensemble evaluation sections).

Bar plots were also generated to visually compare the MAE and RMSE scores across all models and the stacking ensemble. These plots clearly show the relative performance of each approach, highlighting which models and ensembles achieved lower error rates.

## 5. Conclusion

Based on the evaluation metrics, the **Stacking Ensemble** model achieved the lowest Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE), indicating that combining the predictions of the individual models resulted in improved performance compared to any single model.

The Simple Neural Network and K-Nearest Neighbors models also showed relatively strong performance among the individual models. The Spiking Neural Network, while showing promising results, could potentially be further optimized.

This project demonstrates the application of various regression models and ensemble techniques for time series traffic prediction, highlighting the potential benefits of using ensemble methods to enhance predictive accuracy.

## 6. Gradio Interface

A Gradio interface was created to allow interactive predictions from all implemented models and ensemble methods. This interface takes the 'Junction', 'Month', 'Day', and 'Hour' as input and provides the predicted traffic volume from each model. The interface can be launched from the notebook to test predictions with custom inputs.

## 7. Installation
For installing the required dependencies, run the following command on your terminal: 
`pip install pandas os sklearn.model_selection torch torch.nn torch.optim torch.utils.data sklearn.metrics numpy matplotlib.pyplot sklearn.tree sklearn.metrics sklearn.ensemble sklearn.neighbors sklearn.metrics snntorch sklearn.linear_model sklearn.model_selection gradio inspect`
or 
Run the above command as the first cell in your .ipynb preceded by apostrophe `!`
