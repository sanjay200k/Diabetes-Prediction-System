from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.datasets import load_diabetes

app = Flask(__name__)

# Load the Diabetes dataset
diabetes = load_diabetes()
X = diabetes.data
y = diabetes.target

# Convert target variable to binary (1 for diabetes, 0 for non-diabetes)
y = np.where(y >= 200, 1, 0)

# Initialize the Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        new_patient_data = [float(request.form[f'feature{i+1}']) for i in range(10)]
        prediction = model.predict([new_patient_data])
        result = "Diabetes" if prediction[0] == 1 else "Non-Diabetes"

        # Visualize Feature Importance
        feature_names = diabetes.feature_names
        importances = model.feature_importances_

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(importances)), importances, align='center')
        ax.set_yticks(range(len(importances)))
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("Feature Importance")
        ax.set_ylabel("Feature")
        ax.set_title("Feature Importance in Diabetes Prediction")

        output = io.BytesIO()
        FigureCanvas(fig).print_png(output)
        encoded_img = base64.b64encode(output.getvalue()).decode('utf-8')
        plt.close(fig)

        return render_template("index.html", result=result, image=encoded_img)

    return render_template("index.html", result=result, image=None)

if __name__ == "__main__":
    app.run(debug=True)
