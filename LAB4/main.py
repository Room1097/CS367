import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Step 1: Load the data
data = pd.read_csv('LAB4/thyroid+disease/allbp.data', header=None)  # Replace 'thyroid_data.csv' with your file path

# Split the data into features and target
data.columns = ['age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick',
                'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid',
                'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG',
                'referral source', 'class', 'id']  # Add columns as needed

# Drop the 'id' column
data = data.drop(columns=['id'])

# Step 2: Preprocess the data
# Convert categorical and Boolean variables to numeric
label_encoder = LabelEncoder()

# Encode 'sex' and 'referral source'
data['sex'] = label_encoder.fit_transform(data['sex'])
data['referral source'] = label_encoder.fit_transform(data['referral source'])

# Convert Boolean values (f/t) to numeric
bool_columns = ['on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
                'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
                'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured', 
                'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured']

for col in bool_columns:
    data[col] = data[col].map({'f': 0, 't': 1})

# Handle missing values
data.replace('?', pd.NA, inplace=True)
data.dropna(inplace=True)  # Alternatively, you can impute missing values

# Convert target variable 'class' to numerical
data['class'] = label_encoder.fit_transform(data['class'])  # Assuming 'class' column has categorical values

# Define features and target
features = ['age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
            'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre',
            'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG', 'referral source']

X = data[features]
y = data['class']

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Combine X_train and y_train for Bayesian model training
train_combined = pd.concat([X_train, y_train], axis=1)

# Step 4: Define and train the Bayesian Network
model = BayesianModel([('age', 'class'), ('sex', 'class'), ('TSH', 'class'), ('T3', 'class'), ('goitre', 'class')])

model.fit(train_combined, estimator=MaximumLikelihoodEstimator)

# Step 5: Make predictions on the test data
infer = VariableElimination(model)

y_pred = []
for index, row in X_test.iterrows():
    evidence = {feature: row[feature] for feature in features}
    prediction = infer.map_query(variables=['class'], evidence=evidence)
    y_pred.append(prediction['class'])

# Step 6: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
