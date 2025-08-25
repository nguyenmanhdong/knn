from flask import Flask, render_template, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# ------------------------
# Khởi tạo Flask
# ------------------------
app = Flask(__name__)

# ------------------------
# Load dữ liệu và train model
# ------------------------
df = pd.read_csv("Iris.csv")
X = df.drop(["Id", "Species"], axis=1)
y = df["Species"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# ------------------------
# Routes
# ------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        try:
            sl = float(request.form["sl"])
            sw = float(request.form["sw"])
            pl = float(request.form["pl"])
            pw = float(request.form["pw"])

            sample = [[sl, sw, pl, pw]]
            pred_species = knn.predict(sample)
            prediction = pred_species[0]
        except:
            prediction = "Lỗi: Vui lòng nhập số hợp lệ!"

    return render_template("index.html", prediction=prediction)

# ------------------------
# Run server
# ------------------------
if __name__ == "__main__":
    app.run(debug=True)
