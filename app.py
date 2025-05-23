from flask import Flask, request, render_template
from src.pipelines.predict_pipeline import PredictPipeline

app = Flask(__name__, template_folder="templates", static_folder="static",static_url_path="/")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("index.html", score="⛔")
    data = request.form
    print(data)
    prediction = PredictPipeline(gender=data["gender"],race_ethnicity=data["race_ethnicity"],parental_level_of_education=data["parental_level_of_education"],lunch=data["lunch"],test_preparation_course=data["test_preparation_course"],reading_score=int(data["reading_score"]),writing_score=int(data["writing_score"]))
    score = prediction.predict()
    return render_template("index.html", score=score)
    

if __name__ == "__main__":
    app.run("0.0.0.0",debug=True)