
# Customer Churn Prediction using Artificial Neural Network (ANN)

## 📚 Project Overview

This project builds an **Artificial Neural Network (ANN)** model to predict **customer churn** — whether a customer will leave a bank or stay, based on their demographics and account information.  
The goal is to help banks **retain valuable customers** by identifying those likely to churn.

The model is trained using the dataset `Churn_Modelling.csv`, which contains various features about customer behavior and account information.

---

## 🗂️ Dataset Information

The dataset contains **10,000 rows** and **14 columns** including:
- **CustomerID**: Unique ID of the customer
- **CreditScore**: Customer's credit score
- **Geography**: Country of the customer
- **Gender**: Male or Female
- **Age**: Age of the customer
- **Tenure**: Number of years the customer has stayed with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of products purchased through the bank
- **HasCrCard**: Does the customer have a credit card? (1=Yes, 0=No)
- **IsActiveMember**: Active bank member? (1=Yes, 0=No)
- **EstimatedSalary**: Estimated salary of the customer
- **Exited**: Target variable (1 = Customer left the bank, 0 = Customer stayed)

---

## 🛠️ Technologies Used

- **Python**
- **TensorFlow & Keras** (for building the ANN)
- **Pandas** (for data analysis)
- **Scikit-Learn** (for preprocessing and splitting the data)
- **Matplotlib** and **Seaborn** (for visualizations)
- **Jupyter Notebook** (for experimentation)

---

## 🧠 Model Architecture

- Input Layer: Number of neurons based on input features (after preprocessing)
- Hidden Layers: Multiple Dense layers with **ReLU activation**
- Output Layer: 1 neuron with **Sigmoid activation** (for binary classification)

The model uses:
- **Binary Crossentropy** as the loss function
- **Adam Optimizer** for training
- **EarlyStopping** to prevent overfitting
- **TensorBoard** for monitoring training visually

---

## 📈 Performance Metrics

- **Accuracy**: Measured to evaluate how well the model predicts churn vs non-churn.
- **Loss**: Monitored during training to assess model improvement.

(Results section can be updated after training completion.)

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. Install required libraries:
   ```bash
   pip install tensorflow pandas scikit-learn matplotlib seaborn
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook churn.ipynb
   ```

4. Train the model and evaluate the results.

---
![Training Accuracy Graph](1st.png)
![Training Accuracy Graph](2nd.png)
![Training Accuracy Graph](3rd.png)
![Training Accuracy Graph](4th.png)
![Training Accuracy Graph](5th.png)
![Training Accuracy Graph](6th.png)

## 📌 Future Improvements

- Hyperparameter tuning for better performance
- Deployment of model as an API
- Exploring different architectures like RNNs or XGBoost
- Feature engineering to improve model accuracy

---

-
