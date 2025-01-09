from flask import Flask, request, jsonify
import mlflow
import mlflow.xgboost
import xgboost as xgb
import pandas as pd

app = Flask(__name__)

model = xgb.Booster()
model.load_model('/app/models/xgb_model.xgb')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_data = pd.DataFrame([data])
        dmatrix_data = xgb.DMatrix(input_data)
        prediction = model.predict(dmatrix_data)

        return jsonify({
            "HousePrice": prediction.tolist()
        })

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=9696, debug=True)



#  The target variable (MedHouseVal) typically represents the median house value in units of $100,000
# docker build -t house-price-predictor .
# docker run -p 9696:9696 house-price-predictor

# mlflow server \
#   --backend-store-uri sqlite:///mlflow.db \
#   --default-artifact-root ./artifacts \
#   --host 0.0.0.0 --port 5000
