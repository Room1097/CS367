import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, SimpleImputer
from sklearn.metrics import accuracy_score, classification_report
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination

# Define column names based on the dataset structure
column_names = [
    'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick',
    'pregnant', 'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid',
    'lithium', 'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'TSH', 'T3 measured', 
    'T3', 'TT4 measured', 'TT4', 'T4U measured', 'T4U', 'FTI measured', 'FTI', 'TBG measured', 
    'TBG', 'referral source', 'class'
]

# Load the dataset
data = pd.read_csv('LAB4\\thyroid+disease\\allbp.data', header=None)
data.columns = column_names

# Convert categorical and Boolean variables to numeric
label_encoder = LabelEncoder()
data['sex'] = label_encoder.fit_transform(data['sex'])
data['referral source'] = label_encoder.fit_transform(data['referral source'])

# Convert Boolean values (f/t) to numeric
bool_columns = [
    'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
    'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium',
    'goitre', 'tumor', 'hypopituitary', 'psych', 'TSH measured', 'T3 measured', 
    'TT4 measured', 'T4U measured', 'FTI measured', 'TBG measured'
]
for col in bool_columns:
    data[col] = data[col].map({'f': 0, 't': 1})

# Handle missing values
data.replace('?', pd.NA, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')
data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Convert target variable 'class' to numerical
data['class'] = label_encoder.fit_transform(data['class'])

# Define features and target
features = [
    'age', 'sex', 'on thyroxine', 'query on thyroxine', 'on antithyroid medication', 'sick', 'pregnant',
    'thyroid surgery', 'I131 treatment', 'query hypothyroid', 'query hyperthyroid', 'lithium', 'goitre',
    'tumor', 'hypopituitary', 'psych', 'TSH', 'T3', 'TT4', 'T4U', 'FTI', 'TBG', 'referral source'
]

X = data[features]
y = data['class']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Combine X_train and y_train for Bayesian model training
train_combined = pd.concat([X_train, y_train.reset_index(drop=True)], axis=1)

# Define and train the Bayesian Network
model = BayesianModel([
    ('age', 'class'), ('sex', 'class'), ('TSH', 'class'), ('T3', 'class'), ('goitre', 'class')
])
model.fit(train_combined, estimator=MaximumLikelihoodEstimator)

# Load and preprocess the test data
test_data = pd.read_csv('LAB4\\thyroid+disease\\allbp.test', header=None)
test_data.columns = column_names

# Preprocess the test data similarly to the training data
test_data['sex'] = label_encoder.transform(test_data['sex'])
test_data['referral source'] = label_encoder.transform(test_data['referral source'])
for col in bool_columns:
    test_data[col] = test_data[col].map({'f': 0, 't': 1})
test_data.replace('?', pd.NA, inplace=True)
test_data = pd.DataFrame(imputer.transform(test_data), columns=test_data.columns)
test_data['class'] = label_encoder.transform(test_data['class'])

X_test_data = test_data[features]
y_test_data = test_data['class']

# Make predictions on the test data
infer = VariableElimination(model)
y_pred = []
for index, row in X_test_data.iterrows():
    evidence = {feature: row[feature] for feature in features}
    prediction = infer.map_query(variables=['class'], evidence=evidence)
    y_pred.append(prediction['class'])

# Evaluate the model
accuracy = accuracy_score(y_test_data, y_pred)
print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test_data, y_pred))