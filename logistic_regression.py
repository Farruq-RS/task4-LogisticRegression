import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve, accuracy_score

#Load data
df = pd.read_csv("data.csv")

#Drop 'id' and empty column
df = df.drop(['id', 'Unnamed: 32'], axis=1) 

#Convert labels to binary: Malignant = 1, Benign = 0
df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0}) 

#Features and Target
X = df.drop('diagnosis', axis=1) #Features
y = df['diagnosis'] #Target

#Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) #80-20 split

#Standardize
scaler = StandardScaler() #Standardize features
X_train_scaled = scaler.fit_transform(X_train) 
X_test_scaled = scaler.transform(X_test)

#Create and train the model
model = LogisticRegression() 
model.fit(X_train_scaled, y_train) 

#Predict probabilities
y_pred_prob = model.predict_proba(X_test_scaled)[:, 1] #Probabilities for the positive class

#Predict classes
y_pred_class = model.predict(X_test_scaled) 

#Confusion Matrix and Report
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_class))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_class))

print("\nAccuracy Score:", accuracy_score(y_test, y_pred_class))

#ROC-AUC
roc_score = roc_auc_score(y_test, y_pred_prob) 
print("ROC-AUC Score:", roc_score)

#ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob) #Compute ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend()
plt.grid(True)
plt.show()

#Tune threshold, default is 0.5
threshold = 0.3
y_pred_new = (y_pred_prob >= threshold).astype(int) #New predictions based on threshold

#Evaluate with new threshold
print(f"\nConfusion Matrix (Threshold = {threshold}):")
print(confusion_matrix(y_test, y_pred_new))

print(f"\nClassification Report (Threshold = {threshold}):")
print(classification_report(y_test, y_pred_new))

#Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

#Plotting the sigmoid curve
x_vals = np.linspace(-10, 10, 100)
y_vals = sigmoid(x_vals)

plt.figure(figsize=(8, 6))
plt.plot(x_vals, y_vals, label="Sigmoid Function")
plt.title("Sigmoid Curve")
plt.xlabel("Input (z)")
plt.ylabel("Sigmoid Output (Probability)")
plt.grid(True)
plt.legend()
plt.show()
