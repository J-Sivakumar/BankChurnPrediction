import pandas as pd
import os
import plotly.express as px
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import plotly.graph_objects as go

def load_data(file_path):
    df = pd.read_csv(file_path).drop(columns=['Name'])
    df.fillna({"Balance": df["Balance"].median(), "EstimatedSalary": df["EstimatedSalary"].median()}, inplace=True)
    df["Target"] = df["Target"].map({"Yes": 0, "No": 1})
    return df

def plot_pie_chart(df, column, title, filename, image_dir):
    file_path = os.path.join(image_dir, f"{filename}.html")
    fig = px.pie(
        df, names=column, title=title, color=column, 
        color_discrete_map={"Not Churn": 'blue', "Churn": 'red'}, hole=0.3
    )
    fig.write_html(file_path)
    # print(f"Pie chart saved as:{filename}")

def plot_histogram(df, image_dir, column, title, xaxis_title, filename, color_batch=0):
    file_path = os.path.join(image_dir+'visualizations/', f"{filename}.html")
    df[column] = df[column].astype(str) 
    color_batches = [
        {"Not Churn": "blue", "Churn": "red"},
        {"Not Churn": "green", "Churn": "orange"},
        {"Not Churn": "purple", "Churn": "pink"},
        {"Not Churn": "cyan", "Churn": "magenta"},
        {"Not Churn": "teal", "Churn": "yellow"}
    ]
    colors = color_batches[color_batch % len(color_batches)]
    fig = px.histogram(
        df, x=column, color='Target', barmode='group', 
        color_discrete_map=colors, title=title
    )
    fig.update_layout(
        xaxis_title=xaxis_title, 
        yaxis_title='Count', 
    )
    fig.write_html(file_path)
    # print(f"Histogram saved as: {file_path}")

def dashboard(df, image_dir):
    df['Target'] = df['Target'].replace({0: "Not Churn", 1: "Churn"})
    plot_pie_chart(df, 'Target', 'Churn Distribution', 'churn_pie2', image_dir)
    histograms = [
        {'column': 'Gender', 'title': "Churn Rate by Gender", 'xaxis': "Gender", 'filename': "gender_churn"},
        {'column': 'HasCrCard', 'title': "Churn vs. Credit Card Ownership", 'xaxis': "Has Credit Card", 'filename': "credit_card_churn"},
        {'column': 'hasComplaints', 'title': "Impact of Complaints on Churn", 'xaxis': "Has Complaints", 'filename': "complaints_churn"},
        {'column': 'PersonalizedService', 'title': "Personalized Service Impact on Churn", 'xaxis': "Personalized Service Rating", 'filename': "personalized_service_churn"},
        {'column': 'Tenure', 'title': "Tenure Distribution by Churn Status", 'xaxis': "Tenure", 'filename': "tenure_churn"},
        {'column': 'AgeGroup', 'title': "Age Group Distribution by Churn Status", 'xaxis': "Age Group", 'filename': "agegroup_churn"},
        {'column': 'isActiveMember', 'title': "Active Member vs Churn", 'xaxis': "Active Member", 'filename': "active_member_churn"},
        {'column': 'NoOfComplaints', 'title': "No of Complaints vs Churn", 'xaxis': "No of Complaints", 'filename': "no_of_complaints_churn"},
        {'column': 'ComplaintsSeverity', 'title': "Complaints Severity vs Churn", 'xaxis': "Complaints Severity", 'filename': "complaints_severity_churn"},
        {'column': 'hasIssueSolved', 'title': "Issue Solved vs Churn", 'xaxis': "Has Issue Solved", 'filename': "issue_solved_churn"},
        {'column': 'FreqUse', 'title': "Frequency of Use vs Churn", 'xaxis': "Frequency of Use", 'filename': "freq_use_churn"},
        {'column': 'DoesChargesImpact', 'title': "Does Charges Impact vs Churn", 'xaxis': "Does Charges Impact", 'filename': "charges_impact_churn"},
        {'column': 'OverallSatisfaction', 'title': "Overall Satisfaction vs Churn", 'xaxis': "Overall Satisfaction", 'filename': "overall_satisfaction_churn"},
        {'column': 'OfflineBankingExperience', 'title': "Offline Banking Experience vs Churn", 'xaxis': "Offline Banking Experience", 'filename': "offline_banking_experience_churn"},
    ]
    histograms = sorted(histograms, key=lambda x: x['filename'])
    for i, hist in enumerate(histograms):
        plot_histogram(df,image_dir, column=hist['column'], title=hist['title'], xaxis_title=hist['xaxis'], filename=hist['filename'],color_batch= i)

def preprocess_data(df):
    scaler = StandardScaler()
    df[['Balance', 'EstimatedSalary']] = scaler.fit_transform(df[['Balance', 'EstimatedSalary']])
    satisfaction_mapping = {"Very Dissatisfied": 1, "Dissatisfied": 2, "Neutral": 3, "Satisfied": 4, "Very Satisfied": 5}
    overall_mapping = {"Very poor": 1, "Poor": 2, "Average": 3, "Good": 4, "Excellent": 5}
    year_order = {"Less than 1 year": 0, "1 - 3 years": 1, "3 - 5 years": 2, "5 - 10 years": 3, "More than 10 years": 4}
    issues_order = {"No Issue": 0, "Minor Issue": 1, "Moderate Issue": 2, "Major Issue": 3, "Severe Issue": 4}
    time_order = {"Daily": 0, "Weekly": 1, "Monthly": 2, "Rarely": 3, "Never": 4}
    age_order = {"Below 18": 0, "18 - 25": 1, "26 - 35": 2, "36 - 45": 3, "46 - 60": 4, "Above 60": 5}
    satisfaction_columns = ["OnlineBankingExperience", "OfflineBankingExperience", "ATMexperience", "PersonalizedService"]
    df[satisfaction_columns] = df[satisfaction_columns].replace(satisfaction_mapping)
    df["SatOfRes"] = df["SatOfRes"].replace(satisfaction_mapping)
    df["OverallSatisfaction"] = df["OverallSatisfaction"].replace(overall_mapping)
    df["AvgSatisfaction"] = df[satisfaction_columns].mean(axis=1)
    df["LowSatisfactionIndicator"] = (df[satisfaction_columns] <= 2).any(axis=1).astype(int)
    df["Tenure"] = df["Tenure"].replace(year_order)
    df["ComplaintsSeverity"] = df["ComplaintsSeverity"].replace(issues_order)
    df["FreqUse"] = df["FreqUse"].replace(time_order)
    df["AgeGroup"] = df["AgeGroup"].replace(age_order)
    df.drop(columns=satisfaction_columns, inplace=True)
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    features = ['AgeGroup', 'AvgSatisfaction', 'Balance', 'ComplaintsSeverity', 'DoesChargesImpact', 
                'EstimatedSalary', 'FreqUse', 'Gender', 'HasCrCard', 'LowSatisfactionIndicator', 
                'NoOfComplaints', 'NoOfServices', 'OverallSatisfaction', 'PastChurn', 'RecomToOthers', 
                'SatOfRes', 'Tenure', 'hasComplaints', 'isActiveMember']
    
    return df[features], df['Target']

def train_models(x_train, y_train):
    model_params = [
        ("Logistic Regression", LogisticRegression(solver='liblinear', max_iter=100)),
        ("Decision Tree", DecisionTreeClassifier(max_depth=5, random_state=7)),
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=7, max_features="sqrt", max_leaf_nodes=30)),
        ("KNN", KNeighborsClassifier(n_neighbors=5, algorithm='auto', weights='uniform')),
        ("Gradient Boosting", GradientBoostingClassifier(loss='log_loss', n_estimators=100, learning_rate=0.1)),
        ("Naive Bayes", GaussianNB()),
        ("XGBoost", XGBClassifier(objective='binary:logistic', n_estimators=100, learning_rate=0.1)),
        ("LightGBM", LGBMClassifier(objective='binary', n_estimators=100, learning_rate=0.5, verbose=-1))
    ]
    models = {name: model.fit(x_train, y_train) for name, model in model_params}
    return models


def evaluate_models(models, x_test, y_test):
    results = {}
    for name, model in models.items():
        y_pred = model.predict(x_test)
        y_prob = model.predict_proba(x_test)[:, 1]
        results[name] = {
            'Accuracy': f"{accuracy_score(y_test, y_pred) * 100:.2f}%",
            'Precision': f"{precision_score(y_test, y_pred) * 100:.2f}%",
            'Recall': f"{recall_score(y_test, y_pred) * 100:.2f}%",
            'F1-Score': f"{f1_score(y_test, y_pred) * 100:.2f}%",
            'ROC-AUC': f"{roc_auc_score(y_test, y_prob) * 100:.2f}%"
        }
    # print(results)
    return results

def plot_confusion_matrices(models, x_test, y_test,  save_dir):
    colors = ['Greens', 'RdYlGn', 'Cividis', 'Oranges', 'RdBu', 'Spectral', 'PiYG', 'PuOr']
    i=0
    for name, model in models.items():
        y_pred = model.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        # Convert confusion matrix to DataFrame for Plotly Express
        cm_df = pd.DataFrame(cm, index=["Not Churned", "Churned"], columns=["Not Churned", "Churned"])
        # Create heatmap using Plotly Express
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale=colors[i],
                        labels=dict(x="Predicted Outcome", y="Actual Outcome", color="Count"))
        fig.update_layout(title=f"Confusion Matrix of {name}")
        i+=1
        # Save as an HTML file
        file_path = os.path.join(save_dir, f"{name.lower().replace(' ', '_')}_confusion_matrix.html")
        # print(file_path)
        fig.write_html(file_path)

def save_models(models, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for name, model in models.items():
        filename = os.path.join(save_dir, f"{name.replace(' ', '_').lower()}.pkl")
        with open(filename, 'wb') as file:
            pickle.dump(model, file)
        # print(f"Model saved as {filename}")

def save_roc_curve(models, x_test, y_test, image_dir ):
    save_path= image_dir + "/roc_curve.html"
    fpr, tpr, roc_auc = {}, {}, {}
    # Compute ROC Curve and AUC for each model
    for model_name, model in models.items():
        y_prob = model.predict_proba(x_test)[:, 1]  # Get probabilities for the positive class
        fpr[model_name], tpr[model_name], _ = roc_curve(y_test, y_prob)
        roc_auc[model_name] = auc(fpr[model_name], tpr[model_name])
    fig = go.Figure()
    for model_name in models.keys():
        fig.add_trace(go.Scatter(
            x=fpr[model_name], 
            y=tpr[model_name], 
            mode='lines',
            name=f'{model_name} (AUC = {roc_auc[model_name]:.2f})'
        ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash'), name='Random (Baseline)'
    ))
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        template="gridon",
    )
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.write_html(save_path)
    # print(f"ROC Curve saved at {save_path}")

def plot_model_comparison(results, image_dir):
    os.makedirs(image_dir, exist_ok=True)

    data = {
        "Model": [],
        "Metric": [],
        "Score": []
    }

    for model, scores in results.items():
        for metric, value in scores.items():
            score_numeric = float(value.strip('%'))  
            data["Model"].append(model)
            data["Metric"].append(metric)
            data["Score"].append(score_numeric)
    custom_colors = [
        "#636EFA", "#EF553B", "#00CC96", "#AB63FA",  "#FFA15A",  "#19D3F3", "#FF6692", "#B6E880"  
    ]
    # Create a bar chart with a vibrant color palette
    fig = px.bar(data, 
                 x="Model", 
                 y="Score", 
                 color="Metric", 
                 barmode="group",
                 title="Comparison of Models on Accuracy, Precision, Recall, F1 Score, and ROC AUC",
                 labels={"Score": "Percentage (%)", "Model": "Models"},
                 color_discrete_sequence=custom_colors)  # Use a rich color scheme

    fig.update_layout(
        xaxis_tickangle=-30, 
        margin=dict(l=50, r=50, t=80, b=120),  
        font=dict(family="Arial, sans-serif", size=14),  
        bargap=0.12,  
        plot_bgcolor="rgba(240, 240, 250, 1)",  
        hovermode="x unified" 
    )
    save_path = os.path.join(image_dir, "model_comparison.html")
    fig.write_html(save_path)

    # print(f"Plot saved at: {save_path}")

def main(file_path = "./data/Bank Customer Churn Data.csv", image_dir="./static/images/", model_dir="./models/pretrained/"):
    # print('main is called-------------------')
    df = load_data(file_path)
    df_copy = df.copy(deep=True)
    dashboard(df_copy, image_dir)
    x, y = preprocess_data(df)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=6, stratify=y)
    models = train_models(x_train, y_train)
    results = evaluate_models(models, x_test, y_test)
    save_models(models, model_dir)
    plot_confusion_matrices(models, x_test, y_test, os.path.join(image_dir, "confusion_matrices/"))
    save_roc_curve(models, x_test, y_test, os.path.join(image_dir, "roc_curves/"))
    plot_model_comparison(results, image_dir)
    return results
# main(image_dir="./static/custom_images/", model_dir="./models/custom")
# if __name__ == "__main__":
#     main()
