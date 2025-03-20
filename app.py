from flask import Flask, render_template, request

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)

app = application


# Route for a home page
@app.route("/")
def index() -> str:
    return render_template("index.html")


@app.route("/predictdata", methods=["GET", "POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("home.html")
    else:
        data = CustomData(
            description=request.form.get("description"),
            director=request.form.get("director"),
            cast=request.form.get("cast"),
        )

        pred_df = data.get_data_as_frame()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template("home.html", results=results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=8080)
