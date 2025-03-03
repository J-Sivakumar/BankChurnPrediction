import model_building
from flask import *
import pickle, os, time, queue
import numpy as np
import pandas as pd
import plotly.express as px

app = Flask(__name__)
app.secret_key = "123#4%^&8(!9*!7!)"

import os

# Define required directories
directories = [
    "./static/images/",
    "./models/pretrained/",
    "./static/custom_images/",
    "./models/custom/",
    "./uploads/",
    "./static/custom_images/roc_curves/",
    "./static/images/roc_curves/",
    "./static/custom_images/confusion_matrices/",
    "./static/images/confusion_matrices/",
    "./static/custom_images/visualizations/",
    "./static/images/visualizations/"
]

for directory in directories:
    os.makedirs(directory, exist_ok=True)

file_path = "./data/Bank Customer Churn Data.csv"
image_dir = "./static/images/"
model_dir = "./models/pretrained/"

custom_image_dir = "./static/custom_images/"
custom_model_dir = "./models/custom/"
UPLOAD_FOLDER = "./uploads/"
custom_file_name = None

results = None
progress_queue = queue.Queue()

@app.route('/')
def home():
    session.pop('use_custom_data', None)
    return render_template('index.html')

@app.route('/file_upload')
def file_upload():
    return render_template('file_upload.html')

@app.route("/progress")
def progress():
    def generate():
        while True:
            try:
                category, message = progress_queue.get(timeout=10)
                yield f"data: {category}|{message}\n\n"
            except queue.Empty:
                yield "event: end\ndata: done\n\n"
                break
    return Response(generate(), mimetype="text/event-stream")

ALLOWED_EXTENSIONS = {"csv"}

def allowed_file(file):
    if "." in file.filename and file.filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS:
        return file.content_type in ["text/csv", "application/vnd.ms-excel"]
    return False

def has_same_columns(default_file, uploaded_file, progress_queue):
    try:
        default_df = pd.read_csv(default_file, nrows=1)
        uploaded_df = pd.read_csv(uploaded_file, nrows=1)

        if list(default_df.columns) == list(uploaded_df.columns):
            return True
        else:
            message = f"Column mismatch. Expected: {list(default_df.columns)}, but got: {list(uploaded_df.columns)}"
            progress_queue.put(("danger", message))
            return False

    except Exception as e:
        message = f"Error reading CSV: {str(e)}"
        progress_queue.put(("danger", message))
        return False

@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        progress_queue.put(("danger", "No file part. Please upload a valid dataset."))
        return redirect(url_for("file_upload"))
    
    file = request.files["file"]
    
    if file.filename.strip() == "":
        progress_queue.put(("danger", "No file selected. Please upload a valid dataset."))
        return redirect(url_for("file_upload"))
    
    if not allowed_file(file):
        progress_queue.put(("danger", "Invalid file type. Please upload a CSV file."))
        return redirect(url_for("file_upload"))

    custom_file_name = file.filename
    custom_file_path = os.path.join(UPLOAD_FOLDER, custom_file_name)
    file.save(custom_file_path)

    file_path = "./data/Bank Customer Churn Data.csv"
    columns_match = has_same_columns(file_path, custom_file_path, progress_queue)

    if not columns_match:
        os.remove(custom_file_path)
        return redirect(url_for("file_upload"))

    progress_queue.put(("success", f"File '{custom_file_name}' uploaded successfully!"))
    time.sleep(1)
    
    try:
        progress_queue.put(("info", "📂 Loading data..."))
        time.sleep(1)
        df = model_building.load_data(custom_file_path)
        df_copy = df.copy(deep=True)

        progress_queue.put(("info", "🛠 Preprocessing data..."))
        time.sleep(1)
        x, y = model_building.preprocess_data(df)

        progress_queue.put(("info", "🔀 Splitting dataset..."))
        time.sleep(1)
        x_train, x_test, y_train, y_test = model_building.train_test_split(x, y, test_size=0.2, random_state=6, stratify=y)

        progress_queue.put(("info", "🤖 Training machine learning models..."))
        time.sleep(1)
        models = model_building.train_models(x_train, y_train)

        progress_queue.put(("info", "📈 Evaluating models..."))
        time.sleep(1)
        results = model_building.evaluate_models(models, x_test, y_test)

        progress_queue.put(("info", "💾 Saving trained models..."))
        time.sleep(1)
        model_building.save_models(models, custom_model_dir)

        progress_queue.put(("info", "📊 Generating dashboard..."))
        time.sleep(1)
        model_building.dashboard(df_copy, custom_image_dir)
        model_building.plot_confusion_matrices(models, x_test, y_test, os.path.join(custom_image_dir, "confusion_matrices/"))
        model_building.save_roc_curve(models, x_test, y_test, os.path.join(custom_image_dir, "roc_curves/"))
        model_building.plot_model_comparison(results, custom_image_dir)

        progress_queue.put(("success", "✅ Model training completed successfully!"))
        session['use_custom_data'] = True
        session.modified = True  # Ensure session changes are saved

    except Exception as e:
        progress_queue.put(("danger", f"❌ Error processing file: {str(e)}"))
    
    return redirect(url_for("file_upload"))

@app.route('/check-session')
def check_session():
    return f"Session Value: {session.get('use_custom_data', None)}"


@app.route('/reset', methods=['GET'])
def reset_form():
    return redirect(url_for('predict'))

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'GET':
            return render_template('prediction.html', prediction=None)
        
        model_path = custom_model_dir if session.get('use_custom_data') else model_dir
        with open(os.path.join(model_path, 'random_forest.pkl'), 'rb') as file:
            model = pickle.load(file)
        
        columns = ['AgeGroup', 'AvgSatisfaction', 'Balance', 'ComplaintsSeverity', 'DoesChargesImpact', 'EstimatedSalary', 'FreqUse', 'Gender', 'HasCrCard', 'LowSatisfactionIndicator', 'NoOfComplaints', 'NoOfServices', 'OverallSatisfaction', 'PastChurn', 'RecomToOthers', 'SatOfRes', 'Tenure', 'hasComplaints', 'isActiveMember']
        satisfaction_columns = ["OnlineBankingExperience", "OfflineBankingExperience", "ATMexperience", "PersonalizedService"]
        
        input_data = {key: int(request.form[key]) for key in request.form.keys() if key != 'Name'}
        input_data['LowSatisfactionIndicator'] = int(any(input_data[col] <= 2 for col in satisfaction_columns))
        input_data['AvgSatisfaction'] = sum(input_data[col] for col in satisfaction_columns) / len(satisfaction_columns)
        
        x = np.array([input_data[col] for col in columns]).reshape(1, -1)
        prediction = model.predict(x)
        pieData = model.predict_proba(x)[0]
        
        df = pd.DataFrame({"Category": ["Not Churn", "Churn"], "Probability": pieData})
        fig = px.pie(df, values="Probability", names="Category", title="Churn Prediction Probability", color="Category", color_discrete_map={"Not Churn": "blue", "Churn": "red"})
        fig.write_html("./static/images/churn_pie.html")
        
        return render_template('prediction.html', prediction="Churn" if prediction[0] == 1 else "Not Churn", form_data=request.form)
    except Exception as e:
        return jsonify({'error': str(e)})

@app.route('/dashboard')
def dashboard():
    global results
    if not session.get("use_custom_data"):
        results = model_building.main(file_path=file_path, image_dir=image_dir, model_dir=model_dir)
    image_directory = custom_image_dir if session.get("use_custom_data") else image_dir
    return render_template("base.html", image_dir=image_directory)

@app.route('/data-visualization')
def data_visualization():
    image_directory = os.path.join(custom_image_dir, "visualizations") if session.get("use_custom_data") else os.path.join(image_dir, "visualizations")
    visualization_images = os.listdir(image_directory) if os.path.exists(image_directory) else []
    path = image_directory.split('static/')[-1] 
    return render_template("data_visualization.html", visualization_images=visualization_images, path=path)

@app.route('/confusion-matrices')
def confusion_matrices():
    matrix_directory = os.path.join(custom_image_dir, "confusion_matrices") if session.get("use_custom_data") else os.path.join(image_dir, "confusion_matrices")
    confusion_matrices = os.listdir(matrix_directory) if os.path.exists(matrix_directory) else []
    path = matrix_directory.split('static/')[-1]
    return render_template("confusion_matrices.html", confusion_matrices=confusion_matrices, path=path)

@app.route('/model-results')
def model_results():
    global results
    if results is None:
        file_path_to_use = file_path if not session.get("use_custom_data") else os.path.join(UPLOAD_FOLDER, custom_file_name)
        results = model_building.main(file_path_to_use)
    image_directory = custom_image_dir if session.get("use_custom_data") else image_dir
    image_path = os.path.join(image_directory, "model_comparison.html")

    return render_template("model_results.html", results=results, image_path=image_path)


@app.route('/roc-curve')
def roc_curve():
    roc_directory = os.path.join(custom_image_dir, "roc_curves") if session.get("use_custom_data") else os.path.join(image_dir, "roc_curves")
    roc_curve_images = os.listdir(roc_directory) if os.path.exists(roc_directory) else []
    return render_template("roc_curve.html", roc_curve_images=roc_curve_images, relative_roc_path = roc_directory.split("static/")[-1])

if __name__ == '__main__':
    app.run(debug=True)
