import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder

def train_model():

    data = pd.read_csv("mental_health.csv")

    # remove empty rows
    data = data.dropna()

    le_gender = LabelEncoder()
    le_exercise = LabelEncoder()
    le_stress = LabelEncoder()
    le_condition = LabelEncoder()

    data["Gender"] = le_gender.fit_transform(data["Gender"])
    data["Exercise Level"] = le_exercise.fit_transform(data["Exercise Level"])
    data["Stress Level"] = le_stress.fit_transform(data["Stress Level"])

    X = data[[
        "Age",
        "Sleep Hours",
        "Stress Level",
        "Gender",
        "Exercise Level"
    ]]

    y = le_condition.fit_transform(data["Mental Health Condition"])

    model = DecisionTreeClassifier()
    model.fit(X, y)

    return model, le_gender, le_exercise, le_stress, le_condition