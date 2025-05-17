from flask import Flask, request, jsonify
import numpy as np
import joblib  # or pickle if you used that
import traceback

app = Flask ( __name__ )

# Load your trained model (replace with your actual model path)
try:
    model = joblib.load ( "premium_model.pkl" )  # or "model.pkl"
except Exception as e:
    print ( "Model could not be loaded:", e )
    model = None


@app.route ( '/' )
def home():
    return "Premium Prediction API is running."


@app.route ( '/predict', methods=[ 'POST' ] )
def predict():
    if model is None:
        return jsonify ( {"error": "Model not loaded."} ), 500

    try:
        data = request.get_json ( force=True )

        # Assume the data is a dict with keys: age, gender, income, etc.
        features = [
            data [ 'age' ],
            data [ 'gender' ],  # might need to encode this
            data [ 'income' ]
        ]
        features = np.array ( features ).reshape ( 1, -1 )

        prediction = model.predict ( features )

        return jsonify ( {'premium_prediction': float ( prediction [ 0 ] )} )

    except Exception as e:
        return jsonify ( {'error': str ( e ), 'trace': traceback.format_exc ()} ), 400


if __name__ == '__main__':
    app.run ( debug=True )
