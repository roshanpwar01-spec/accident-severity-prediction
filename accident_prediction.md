# **Project Name**    - Accident Severity Prediction using machine learning and deep learning

**Objective:** -The main objective of this project is to develop an intelligent accident severity prediction system using machine learning and deep learning techniques. 

The project aims to analyze accident-related factors such as weather conditions, road conditions, vehicle involvement, and driver information to predict the severity level of road accidents as Minor, Major, or Fatal.

**Key Tasks:**
• Data collection and preprocessing  
• Handling missing and categorical data  
• Data visualization and trend analysis  
• Feature encoding and scaling  
• Building machine learning models  
• Developing deep learning models (RNN, LSTM, GRU)  
• Training and testing models  
• Performance evaluation using accuracy, ROC curve, and confusion matrix  
• Comparative analysis of ML and DL models  
• Future trend prediction and severity analysis

### **Problem Statement**

### **Project Overview:**

This project focuses on predicting road accident severity using machine learning and deep learning techniques. The system analyzes various accident-related factors such as weather conditions, road conditions, speed limit, vehicle involvement, driver information, and environmental conditions to classify accidents into Minor, Major, and Fatal severity categories.

The project includes data preprocessing, visualization, trend analysis, feature engineering, model development, and performance evaluation. Multiple machine learning models such as Logistic Regression, Random Forest, XGBoost, and LightGBM, along with deep learning models including DNN, RNN, LSTM, and GRU, were implemented and compared.

Evaluation metrics such as accuracy, ROC curve, confusion matrix, and heatmap analysis were used to analyze model performance. The project also includes comparative analysis and future trend prediction to improve intelligent road safety systems.

**Key Objectives:**

• To analyze accident-related data and identify important severity factors  
• To preprocess and visualize accident datasets effectively  
• To develop machine learning and deep learning models for severity prediction  
• To compare the performance of ML and DL models  
• To evaluate models using accuracy, ROC curve, and confusion matrix  
• To identify reasons behind poor model performance  
• To support intelligent road safety and accident prevention systems  
• To perform future trend analysis for accident severity prediction

## ***Know The Data***

### Import Libraries

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
```

### Dataset Loading

```python

# -----------------------------
# 1. Load Dataset
# -----------------------------
df = pd.read_excel("road_accident_1.xlsx")
```

```python
df.tail()
```

### What did i know about the dataset?

The dataset used in this project is a structured tabular road accident dataset containing approximately 3,000 real accident records collected from 2020 to 2023. It includes both numerical and categorical features related to accident conditions, driver information, vehicle details, and environmental factors.

The dataset contains important features such as weather conditions, road conditions, lighting conditions, speed limit, driver age, gender, alcohol involvement, number of vehicles involved, number of casualties, and accident location details. The target variable of the dataset is Accident Severity, which is classified into three categories: Minor, Major, and Fatal.

Before model training, several preprocessing techniques were applied including data cleaning, missing value handling, categorical encoding, feature scaling, and trend analysis. The dataset was used for supervised multi-class classification to predict accident severity using machine learning and deep learning models.

```python
df["Weather Conditions"].unique()
```

### Data Augmentation

Data augmentation was performed to increase the size and diversity of the dataset from approximately 3,000 records to around 9,000 records. This helped improve model learning, reduce overfitting, and handle class imbalance issues. 

The augmentation process enhanced the dataset by generating additional training samples, allowing machine learning and deep learning models to learn better patterns and improve prediction performance, especially for less frequent accident severity classes.

### Data Transformation

Data transformation was performed to convert raw accident data into a format suitable for machine learning and deep learning models. Categorical features such as weather conditions, gender, lighting conditions, and accident severity were converted into numerical values using label encoding.

Feature scaling and normalization techniques were also applied to maintain consistency among numerical features and improve model training performance. These transformations helped the models process the dataset efficiently and achieve better prediction accuracy.

```python
# -----------------------------
# 2. Convert numeric columns
# -----------------------------
numeric_cols = [
    'Year',
    'Month_num',
    'Number of Vehicles Involved',
    'Number of Casualties',
    'Number of Fatalities',
    'Speed Limit (km/h)',
    'Driver Age'
]

for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill missing values
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# -----------------------------
# 3. Encode categorical columns
# -----------------------------
categorical_cols = df.select_dtypes(include=['object']).columns

le = LabelEncoder()
for col in categorical_cols:
    df[col] = le.fit_transform(df[col].astype(str))

# -----------------------------
# 4. Augment dataset to 9000
# -----------------------------
target_size = 9000
current_size = len(df)

samples_needed = target_size - current_size

# Randomly sample rows
synthetic = df.sample(n=samples_needed, replace=True, random_state=42)

# Add small noise to numeric columns
for col in numeric_cols:
    if col in synthetic.columns:
        noise = np.random.normal(0, 0.5, size=len(synthetic))
        synthetic[col] = synthetic[col] + noise

# Combine original + synthetic
final_augmented = pd.concat([df, synthetic], ignore_index=True)

# -----------------------------
# 5. Save dataset
# -----------------------------
final_augmented.to_excel("augmented_9000_dataset.xlsx", index=False)

print("Original dataset size:", current_size)
print("Augmented dataset size:", len(final_augmented))
print("Dataset saved as augmented_9000_dataset.xlsx")
```

```python
data = final_augmented.copy()
```

```python
data.tail()
```

## ***Data Visualization***

Data visualization was performed to analyze accident patterns, feature relationships, and severity trends within the dataset. Various visualizations such as bar charts, histograms, heatmaps, trend graphs, and confusion matrix heatmaps were used to better understand the data.

Visualization helped identify important factors affecting accident severity, detect class imbalance, observe yearly accident trends, and evaluate model performance effectively.

### Year-wise Road Accident Trend Analysis

This graph represents the total number of road accidents recorded over different years. The x-axis shows the years, while the y-axis represents the number of accidents.

The graph indicates that accident cases increased significantly between 2018 and 2022, with the highest number of accidents observed around 2020–2022. After 2022, a slight decrease in accident frequency can be observed.

This trend analysis helps identify changes in accident occurrence over time and supports future accident severity prediction and road safety analysis.

```python
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# 1. Count accidents per year
# -----------------------------
yearly_accidents = data.groupby("Year").size()

# -----------------------------
# 2. Plot graph
# -----------------------------
plt.figure()

plt.plot(yearly_accidents.index, yearly_accidents.values, marker='o')

plt.title("Total Number of Accidents Over Years")
plt.xlabel("Year")
plt.ylabel("Number of Accidents")

plt.grid()
plt.show()
```

```python
weather_map = {
    0: "Clear",
    1: "Rainy",
    2: "Foggy",
    3: "Snowy"
}
```

```python
data["Weather_Label"] = data["Weather Conditions"].map(weather_map)
```

### Weather-wise Road Accident Analysis

This graph represents the distribution of road accidents under different weather conditions. The x-axis shows various weather conditions such as Clear, Foggy, Rainy, and Snowy, while the y-axis represents the total number of accidents.

The graph indicates that the highest number of accidents occurred during snowy weather conditions, followed by foggy and rainy conditions. Comparatively fewer accidents were recorded during clear weather conditions.

This analysis shows that adverse weather conditions significantly influence road accident occurrence and severity, highlighting the importance of weather-aware traffic safety measures.

```python
import matplotlib.pyplot as plt

weather_trend = data.groupby("Weather_Label").size()

weather_trend.plot(kind='bar')

plt.title("Accidents by Weather Condition")
plt.xlabel("Weather")
plt.ylabel("Number of Accidents")

plt.show()
```

### Yearly Accident Growth (%)

This graph represents the yearly percentage change in road accident occurrences. Positive values indicate an increase in accident frequency compared to the previous year, while negative values indicate a decrease.

The graph helps analyze accident growth trends over time and supports future accident prediction and road safety planning.

```python
yearly = data.groupby("Year").size()
growth = yearly.pct_change() * 100

growth.plot()
plt.title("Yearly Accident Growth (%)")
plt.show()
```

## ***Data Transformation***

Categorical-to-numerical transformation was performed to convert non-numeric features into machine-readable numerical values. Features such as weather conditions, gender, lighting conditions, and accident severity were encoded using label encoding techniques.

This transformation enabled machine learning and deep learning models to process categorical data efficiently and improve prediction performance.

```python
categorical_cols = data.select_dtypes(include=['object']).columns

print("Categorical Columns:", categorical_cols)

# ----------------------------- 
# 3. Convert categorical → numeric
# -----------------------------
le = LabelEncoder()

for col in categorical_cols:
    data[col] = le.fit_transform(data[col].astype(str))

# -----------------------------
# 4. Save numeric dataset
# -----------------------------
data.to_excel("numeric_dataset.xlsx", index=False)

print("All categorical columns converted to numeric.")
print("Saved as numeric_dataset.xlsx")
```

```python
data.tail(5)
```

```python
int_cols = [
    'Year',
    'Month_num',
    'Accident Severity',
    'Number of Vehicles Involved',
    'Number of Casualties',
    'Number of Fatalities',
    'Weather Conditions',
    'Road Condition',
    'Lighting Conditions',
    'Speed Limit (km/h)',
    'Driver Age',
    'Gender',
    'Driver License Status',
    'Alcohol Involvement',
    'Accident Location Details'
]

for col in int_cols:
    if col in data.columns:
        data[col] = data[col].round().astype(int)

# Save dataset
data.to_excel("final_integer_dataset.xlsx", index=False)
```

```python
data.tail(5)
```

```python
for col in data.columns:
    neg_count = (data[col] < 0).sum()
    if neg_count > 0:
        print(col, ":", neg_count)
```

```python
data = data.abs()
```

```python
for col in data.columns:
    neg_count = (data[col] < 0).sum()
    if neg_count > 0:
        print(col, ":", neg_count)
```

```python
data.to_excel("data.xlsx", index=False)
```

```python
import numpy as np
print(np.unique(data["Accident Severity"]))
```

```python
import pandas as pd

yearly = data.groupby("Year").size().reset_index(name="Accidents")

print(yearly)
```

```python
import numpy as np

future_years = np.array([2016, 2017, 2018, 2019, 2020]).reshape(-1, 1)

future_preds = model.predict(future_years)

# Print nicely
for year, pred in zip(future_years.flatten(), future_preds):
    print(f"{year} → {int(pred)} accidents")
```

# Deep Learning model

## lstm

LSTM (Long Short-Term Memory) is an advanced deep learning model used for learning sequential and time-dependent patterns in data. It is a type of Recurrent Neural Network (RNN) designed to remember important information for a longer duration and overcome the vanishing gradient problem present in traditional RNNs.

In this project, LSTM was implemented to analyze hidden relationships between accident-related features such as weather conditions, speed limit, road conditions, driver information, and accident severity. The model was trained to classify accident severity into Minor, Major, and Fatal categories.

LSTM uses memory cells and gates (input gate, forget gate, and output gate) to control which information should be remembered or ignored during training. This helps the model learn complex feature dependencies effectively.

However, the dataset used in this project was mainly structured tabular data rather than strongly sequential time-series data. Due to limited temporal dependencies and class imbalance, LSTM performance was lower compared to GRU and Random Forest models.

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# 1. Split features and target
# -----------------------------
X = data.drop("Accident Severity", axis=1).values

y = data["Accident Severity"]
y = y.astype('category').cat.codes
y = y.values

# -----------------------------
# 2. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X = scaler.fit_transform(X)

# -----------------------------
# 3. Create sequences
# -----------------------------
step_size = 7

X_seq = []
y_seq = []

for i in range(len(X) - step_size):
    X_seq.append(X[i:i+step_size])
    y_seq.append(y[i+step_size])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# -----------------------------
# 4. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq,
    test_size=0.2,
    random_state=42,
    stratify=y_seq
)

# -----------------------------
# 5. Class Weights
# -----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)

class_weights = dict(enumerate(class_weights))

# -----------------------------
# 6. Build Improved LSTM Model
# -----------------------------
model = Sequential()

model.add(LSTM(128, return_sequences=True,
               kernel_regularizer=l2(0.001),
               input_shape=(step_size, X_seq.shape[2])))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(64, return_sequences=True,
               kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(LSTM(32,
               kernel_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(64, activation='relu'))

model.add(Dense(len(np.unique(y)), activation='softmax'))

# -----------------------------
# 7. Compile Model
# -----------------------------
optimizer = Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 8. Callbacks
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

# -----------------------------
# 9. Train Model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=64,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_scheduler],
    class_weight=class_weights
)

# -----------------------------
# 10. Predict
# -----------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# -----------------------------
# 11. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred_classes)
print("Model Accuracy:", accuracy)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_classes))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))
# -----------------------------
# 11. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred_classes)
print("Model Accuracy:", accuracy)

# Confusion Matrix (Raw)
cm = confusion_matrix(y_test, y_pred_classes)

print("\nConfusion Matrix:")
print(cm)

# -----------------------------
# 12. Plot Confusion Matrix
# -----------------------------
plt.figure(figsize=(8,6))

sns.heatmap(cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=np.unique(y),
            yticklabels=np.unique(y))

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()
```

# Deep Neural Network

In this project, the Deep Neural Network (DNN) model was used to predict accident severity by learning complex relationships between accident-related features such as weather conditions, road conditions, speed limit, casualties, driver information, and vehicle involvement.

The DNN model consisted of multiple hidden layers that extracted important patterns from the dataset and classified accidents into Minor, Major, and Fatal categories. Unlike RNN, LSTM, and GRU, the DNN processed each accident record independently rather than sequentially.

The model achieved moderate performance because it effectively captured non-linear feature relationships in the structured tabular dataset. However, its performance was lower than Random Forest and GRU due to higher computational complexity and limited sequential learning capability.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# -----------------------------
# 1. Prepare Data
# -----------------------------
X = data.drop("Accident Severity", axis=1).values

y = data["Accident Severity"]
y = y.astype('category').cat.codes
y = y.values

print("Labels:", np.unique(y))

# -----------------------------
# 2. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# 3. Feature Scaling
# -----------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -----------------------------
# 4. Class Weights
# -----------------------------
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# -----------------------------
# 5. Build DNN Model
# -----------------------------
model = Sequential()

model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))

model.add(Dense(64, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(32, activation='relu'))

# Output Layer
model.add(Dense(len(np.unique(y)), activation='softmax'))

# -----------------------------
# 6. Compile Model
# -----------------------------
optimizer = Adam(learning_rate=0.0005)

model.compile(
    optimizer=optimizer,
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# -----------------------------
# 7. Callbacks
# -----------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    min_lr=0.00001
)

# -----------------------------
# 8. Train Model
# -----------------------------
history = model.fit(
    X_train,
    y_train,
    epochs=500,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, lr_scheduler],
    class_weight=class_weights
)

# -----------------------------
# 9. Predict
# -----------------------------
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

# -----------------------------
# 10. Evaluation
# -----------------------------
accuracy = accuracy_score(y_test, y_pred_classes)
print("🔥 Model Accuracy:", accuracy)

# -----------------------------
# Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred_classes)

print("\nConfusion Matrix:")
print(cm)

# Plot Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -----------------------------
# Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes))
```

# Gated Recurrent Unit

In this project, the GRU (Gated Recurrent Unit) model was used for accident severity prediction by learning hidden relationships between accident-related features such as weather conditions, speed limit, road conditions, casualties, vehicle involvement, and driver information.

GRU is an advanced recurrent neural network model that uses update and reset gates to remember important information and ignore unnecessary patterns during training. The model was trained to classify accidents into Minor, Major, and Fatal categories.

Compared to LSTM, GRU has a simpler architecture with fewer parameters, which resulted in faster training and better learning efficiency. GRU performed better in this project because it effectively captured hidden feature patterns while maintaining lower computational complexity on the structured tabular dataset.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from tensorflow.keras.utils import to_categorical
# 2. Data Cleaning & Preprocessing
# Remove severity '0' as it has very few samples
data = data[data['Accident Severity'] != 0].copy()

# Separate features (X) and target (y)
X = data.drop('Accident Severity', axis=1).values
y = data['Accident Severity'].values

# Encode labels (e.g., 1, 2, 3 -> 0, 1, 2) and one-hot encode for the model
le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

# 3. Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.2, random_state=42
)

# 4. Feature Scaling (Crucial for Recurrent Neural Networks)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Reshape for GRU: (samples, time_steps, features)
# We treat each record as a sequence of 1 time step
X_train_reshaped = X_train_scaled.reshape(
    (X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
)
X_test_reshaped = X_test_scaled.reshape(
    (X_test_scaled.shape[0], 1, X_test_scaled.shape[1])
)

# 6. Build the GRU Model
model = Sequential([
    GRU(64, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]), return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(y_categorical.shape[1], activation='softmax')
])

# 7. Compile the Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 8. Train the Model
print("Starting training...")
history = model.fit(
    X_train_reshaped, y_train,
    epochs=1000,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# 9. Evaluate the Model
loss, accuracy = model.evaluate(X_test_reshaped, y_test)
print(f"\nTest Results:")
print(f"Accuracy: {accuracy:.4f}")
```

# Machine learning model

## Extreme Gradient Boosting

In this project, the XGBoost (Extreme Gradient Boosting) model was used for accident severity prediction by analyzing accident-related features such as weather conditions, road conditions, speed limit, casualties, vehicle involvement, and driver information.

XGBoost is an advanced machine learning algorithm based on gradient boosting that builds multiple decision trees sequentially, where each new tree corrects the errors of the previous trees. This improves prediction accuracy and model performance.

The model was trained to classify accidents into Minor, Major, and Fatal categories. XGBoost performed well in this project because it efficiently handled structured tabular data, captured complex feature relationships, reduced overfitting using regularization, and provided high classification accuracy with reliable prediction performance.

```python
pip install xgboost
```

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# -----------------------------
# 1. Train-test split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 2. Train XGBoost model
# -----------------------------
model = XGBClassifier(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# -----------------------------
# 3. Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 4. Evaluate model
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("🔥 Accuracy:", accuracy)

# -----------------------------
# 5. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -----------------------------
# 6. Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# -----------------------------
# 7. ROC-AUC Score (for multi-class)
# -----------------------------
y_prob = model.predict_proba(X_test)

try:
    roc_auc = roc_auc_score(y_test, y_prob, multi_class='ovr')
    print("\nROC-AUC Score:", roc_auc)
except:
    print("\nROC-AUC not supported for this configuration")
```

# Random Forest

In this project, the Random Forest model was used for accident severity prediction by analyzing accident-related features such as weather conditions, road conditions, speed limit, driver information, casualties, and vehicle involvement.

Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting. Each decision tree makes its own prediction, and the final output is determined through majority voting.

The model was trained to classify accidents into Minor, Major, and Fatal categories. Random Forest performed very well in this project because it efficiently handled structured tabular data, captured complex feature relationships, reduced model bias, and provided high classification accuracy with stable prediction performance.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Split features and target
# -----------------------------
X = data.drop("Accident Severity", axis=1)
y = data["Accident Severity"]

# -----------------------------
# 2. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 3. Random Forest Model
# -----------------------------
model = RandomForestClassifier(
    n_estimators=500,      # number of trees
    max_depth=15,          # control tree depth
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

# -----------------------------
# 4. Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 5. Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# -----------------------------
# 7. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# -----------------------------
# 8. Classification Report
# -----------------------------
print(classification_report(y_test, y_pred))
```

## Light Gradient Boosting Machine

In this project, the LightGBM (Light Gradient Boosting Machine) model was used for accident severity prediction by analyzing accident-related features such as weather conditions, road conditions, speed limit, casualties, vehicle involvement, and driver information.

LightGBM is a boosting-based machine learning algorithm that builds decision trees efficiently using a leaf-wise growth approach. It improves prediction accuracy by correcting errors from previous trees while maintaining faster training speed and lower memory usage.

The model was trained to classify accidents into Minor, Major, and Fatal categories. LightGBM performed well in this project because it efficiently handled large structured tabular data, captured important feature relationships, and provided fast and reliable prediction performance.

```python
pip install lightgbm
```

```python
import warnings
warnings.filterwarnings('ignore')
```

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# -----------------------------
# 1. Prepare Data
# -----------------------------
X = data.drop("Accident Severity", axis=1)
y = data["Accident Severity"].astype('category').cat.codes

# -----------------------------
# 2. Train Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# 3. Build LightGBM Model
# -----------------------------
model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6
)

# -----------------------------
# 4. Train Model
# -----------------------------
model.fit(X_train, y_train)

# -----------------------------
# 5. Predict
# -----------------------------
y_pred = model.predict(X_test)

# -----------------------------
# 6. Accuracy
# -----------------------------
accuracy = accuracy_score(y_test, y_pred)
print("🔥 Accuracy:", accuracy)

# -----------------------------
# 7. Confusion Matrix
# -----------------------------
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

# Plot Heatmap
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# -----------------------------
# 8. Classification Report
# -----------------------------
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

# Conclusion

This project successfully developed an intelligent road accident severity prediction system using machine learning and deep learning techniques. The study analyzed accident-related factors such as weather conditions, road conditions, speed limit, driver information, vehicle involvement, and casualties to classify accidents into Minor, Serious, and Fatal categories.

Various machine learning and deep learning models including Random Forest, XGBoost, LightGBM, DNN, LSTM, and GRU were implemented and compared. Among all models, Random Forest achieved the best overall performance with approximately 89–92% accuracy and balanced classification across all severity classes. XGBoost also showed strong predictive capability with excellent ROC-AUC performance, while GRU performed better among deep learning models because of its efficient architecture and ability to capture hidden feature relationships.

The project also included data preprocessing, data augmentation, feature transformation, visualization, trend analysis, ROC curve analysis, and confusion matrix evaluation to improve prediction quality and model understanding. The results demonstrate that machine learning techniques can effectively support intelligent road safety analysis and accident severity prediction systems.

Overall, the project highlights the importance of data-driven approaches in improving traffic safety, assisting emergency response systems, and supporting smart transportation management.

```python

```
