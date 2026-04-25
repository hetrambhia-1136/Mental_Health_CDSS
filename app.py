from flask import Flask, render_template, request, send_file
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet

import os

app = Flask(__name__)

# -----------------------
# Create bigger dataset
# -----------------------

np.random.seed(42)

data = pd.DataFrame({

"Age": np.random.randint(18,60,300),

"Sleep": np.random.uniform(4,9,300),

"Stress": np.random.choice(["Low","Moderate","High"],300),

"Exercise": np.random.choice(["Low","Moderate","High"],300),

"Screen": np.random.randint(1,12,300),

"Work": np.random.randint(1,12,300),

"Social": np.random.randint(1,10,300)

})

# create realistic condition

conditions=[]

for i in range(len(data)):

    if data.loc[i,"Sleep"] < 5 and data.loc[i,"Stress"]=="High":

        conditions.append("Depression")

    elif data.loc[i,"Stress"]=="High":

        conditions.append("Stress")

    elif data.loc[i,"Social"] < 4:

        conditions.append("Anxiety")

    else:

        conditions.append("Healthy")

data["Condition"] = conditions

# -----------------------
# Encoding
# -----------------------

le_stress = LabelEncoder()
le_exercise = LabelEncoder()
le_condition = LabelEncoder()

data["Stress"] = le_stress.fit_transform(data["Stress"])
data["Exercise"] = le_exercise.fit_transform(data["Exercise"])

X = data[[
"Age","Sleep","Stress","Exercise","Screen","Work","Social"
]]

y = le_condition.fit_transform(data["Condition"])

# -----------------------
# Train model
# -----------------------

X_train,X_test,y_train,y_test = train_test_split(

X,y,test_size=0.2,random_state=42

)

model = RandomForestClassifier(

n_estimators=500,
max_depth=12,
random_state=42

)

model.fit(X_train,y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# store prediction

prediction_result=""
explanation_result=""

# -----------------------
# Home
# -----------------------

@app.route("/", methods=["GET","POST"])

def home():

    global prediction_result, explanation_result

    prediction=""
    explanation=""

    if request.method == "POST":

        age = int(request.form["age"])
        sleep = float(request.form["sleep"])

        stress = request.form["stress"]
        exercise = request.form["exercise"]

        screen = int(request.form["screen"])
        work = int(request.form["work"])
        social = int(request.form["social"])

        stress_enc = le_stress.transform([stress])[0]
        exercise_enc = le_exercise.transform([exercise])[0]

        result = model.predict([[

            age,sleep,stress_enc,exercise_enc,screen,work,social

        ]])

        prediction = le_condition.inverse_transform(result)[0]

        # explanation

        if prediction=="Healthy":

            explanation="""
Balanced lifestyle detected.
Maintain regular sleep schedule, moderate screen time and good social activity.
"""

        elif prediction=="Stress":

            explanation="""
High work pressure or excessive screen time may cause stress.
Consider relaxation techniques and time management.
"""

        elif prediction=="Anxiety":

            explanation="""
Lower social interaction detected.
Try increasing communication and healthy lifestyle habits.
"""

        elif prediction=="Depression":

            explanation="""
Low sleep combined with high stress detected.
Professional mental health consultation recommended.
"""

        # create graph file

        values = [sleep,screen,work,social]

        labels = ["Sleep","Screen","Work","Social"]

        plt.figure()

        plt.bar(labels,values)

        plt.title("Lifestyle Factors")

        if not os.path.exists("static"):

            os.makedirs("static")

        plt.savefig("static/graph.png")

        plt.close()

        prediction_result = prediction
        explanation_result = explanation

    return render_template(

        "index.html",

        prediction = prediction,

        explanation = explanation,

        accuracy = round(accuracy*100,1)

    )

# -----------------------
# PDF
# -----------------------

@app.route("/download")

def download():

    file = "mental_health_report.pdf"

    styles = getSampleStyleSheet()

    elements = []

    elements.append(

        Paragraph(

            "AI Mental Health CDSS Report",

            styles["Title"]

        )

    )

    elements.append(Spacer(1,20))

    elements.append(

        Paragraph(

            f"Prediction: {prediction_result}",

            styles["Normal"]

        )

    )

    elements.append(Spacer(1,10))

    elements.append(

        Paragraph(

            f"Model Accuracy: {round(accuracy*100,1)}%",

            styles["Normal"]

        )

    )

    elements.append(Spacer(1,10))

    elements.append(

        Paragraph(

            explanation_result,

            styles["Normal"]

        )

    )

    elements.append(Spacer(1,20))

    elements.append(

        Image(

            "static/graph.png",

            width=350,

            height=220

        )

    )

    doc = SimpleDocTemplate(file)

    doc.build(elements)

    return send_file(

        file,

        as_attachment=True

    )

# -----------------------

if __name__ == "__main__":
    app.run(debug=True, port=5050)