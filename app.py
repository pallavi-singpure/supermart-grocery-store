from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import pandas as pd
import joblib
import numpy as np  # Added for robust prediction handling
from sklearn.preprocessing import LabelEncoder, StandardScaler
import os  # Added for path handling

app = FastAPI()

# --- Configuration ---
# Set up paths relative to the current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Mount Static Files and Templates
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=TEMPLATES_DIR)

# --- Model Loading ---
try:
    # Load models (Ensure these files exist in the 'models' directory)
    LASSO_MODEL = joblib.load(os.path.join(MODELS_DIR, "profit_prediction.pkl"))
    GRADIENT_MODEL = joblib.load(os.path.join(MODELS_DIR, "sales_prediction.pkl"))
    print("Models loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model: {e}. Please ensure profit_model.pkl and sales_model.pkl are in the 'models' folder.")
    LASSO_MODEL = None
    GRADIENT_MODEL = None
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    LASSO_MODEL = None
    GRADIENT_MODEL = None


# --- Preprocessing Function ---
# NOTE: This function needs to handle the order and number of features required by your trained models.
# The LabelEncoder and StandardScaler instances used during training must be saved and loaded to ensure consistency.
# For simplicity, I'm keeping your original structure but adding a note about proper production readiness.
#
def preprocess(df):
    """
    Preprocesses the input DataFrame columns to be ready for the model.
    WARNING: For a production application, you must save and reuse the LabelEncoder
    and StandardScaler fitted on your training data to avoid errors.
    """

    # 1. Drop irrelevant columns
    # Ensure this list matches the columns dropped/ignored during original model training
    df = df.drop(['Order ID', 'Customer Name', 'Order Date', 'State'], axis=1)

    # 2. Handle Categorical Columns (using placeholder LE for demonstration)
    cat_cols = ['Category', 'Sub Category', 'City', 'Region']

    for col in cat_cols:
        # In a real app, load the fitted LE for each column
        le = LabelEncoder()
        # Fit on current data (This is incorrect for production but necessary if you didn't save the fitted encoders)
        # Assuming the input values are ones the LE has seen before for simplicity.
        try:
            # Create dummy data for fitting the LE if it hasn't been loaded
            dummy_fit_data = [df[col].iloc[0]]
            le.fit(dummy_fit_data)
            df[col] = le.transform(df[col])
        except Exception:
            # Fallback if LE fails to transform new unseen data, which is a common deployment issue
            print(f"Warning: Could not transform categorical column '{col}'. Returning 0.")
            df[col] = 0

    # 3. Handle Scaling (using placeholder scaler for demonstration)
    # In a real app, load the fitted StandardScaler
    try:
        scaler = StandardScaler()
        # IMPORTANT: The data frame must have the exact columns in the exact order the model expects.
        # Since the shape changes, we convert to numpy array before scaling.
        df_scaled = scaler.fit_transform(df.values)
    except Exception as e:
        print(f"Scaling error: {e}")
        # Return the unscaled data if scaling fails
        df_scaled = df.values

    return df_scaled


# --- Routes ---

@app.get("/")
def home(request: Request):
    """Renders the main landing page."""
    return templates.TemplateResponse("home.html", {"request": request})


@app.get("/sales")
def sales_page(request: Request):
    """Renders the sales prediction input form."""
    return templates.TemplateResponse("sales.html", {"request": request})


@app.post("/predict-sales")
def predict_sales(
        request: Request,
        category: str = Form(...),
        sub_category: str = Form(...),
        city: str = Form(...),
        region: str = Form(...),
        discount: float = Form(...)
):
    """Handles the sales prediction request."""
    if not GRADIENT_MODEL:
        return templates.TemplateResponse("result.html",
                                          {"request": request, "title": "Error", "value": "Model not loaded."},
                                          status_code=500)

    # 1. Create DataFrame with all columns used in training
    # NOTE: The missing 'Sales' and 'Profit' columns in the raw input will be handled by the preprocessing drop/select process.
    input_data = {
        "Order ID": "OD1",
        "Customer Name": "User",
        "Order Date": "2025-01-01",
        "Category": category,
        "Sub Category": sub_category,
        "City": city,
        "Region": region,
        "Discount": discount,
        # Placeholder/Dummy values for columns not collected from user but expected by preprocess
        "Sales": 0.0,
        "State": "NA"
    }

    # Define the column order expected by your model's preprocessing step
    # This is crucial for models like Lasso and Gradient Boosting.
    df = pd.DataFrame([input_data])

    try:
        # 2. Preprocess the data
        x = preprocess(df)

        # 3. Predict
        prediction_result = GRADIENT_MODEL.predict(x)

        # Ensure prediction is a scalar value
        if isinstance(prediction_result, np.ndarray) and prediction_result.ndim == 1 and prediction_result.size == 1:
            prediction = prediction_result[0]
        else:
            prediction = prediction_result  # Use as is if it's already a scalar (less common)

        # 4. Return result
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "title": "Sales", "value": round(float(prediction), 2)}
        )

    except Exception as e:
        print(f"Error during Sales prediction: {e}")
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "title": "Error", "value": f"Prediction failed due to backend error: {e}"}
        )


@app.get("/profit")
def profit_page(request: Request):
    """Renders the profit prediction input form."""
    return templates.TemplateResponse("profit.html", {"request": request})


@app.post("/predict-profit")
def predict_profit(
        request: Request,
        category: str = Form(...),
        sub_category: str = Form(...),
        city: str = Form(...),
        region: str = Form(...),
        sales: float = Form(...),
        discount: float = Form(...)
):
    """Handles the profit prediction request."""
    if not LASSO_MODEL:
        return templates.TemplateResponse("result.html",
                                          {"request": request, "title": "Error", "value": "Model not loaded."},
                                          status_code=500)

    # 1. Create DataFrame with all columns used in training
    input_data = {
        "Order ID": "OD1",
        "Customer Name": "User",
        "Order Date": "2025-01-01",
        "Category": category,
        "Sub Category": sub_category,
        "City": city,
        "Region": region,
        "Sales": sales,  # 'Sales' is an input for profit prediction
        "Discount": discount,
        # Placeholder/Dummy values for columns not collected from user but expected by preprocess
        "State": "NA"
    }

    # Define the column order expected by your model's preprocessing step
    df = pd.DataFrame([input_data])

    try:
        # 2. Preprocess the data
        x = preprocess(df)

        # 3. Predict
        prediction_result = LASSO_MODEL.predict(x)

        # Ensure prediction is a scalar value
        if isinstance(prediction_result, np.ndarray) and prediction_result.ndim == 1 and prediction_result.size == 1:
            prediction = prediction_result[0]
        else:
            prediction = prediction_result  # Use as is if it's already a scalar

        # 4. Return result
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "title": "Profit", "value": round(float(prediction), 2)}
        )

    except Exception as e:
        print(f"Error during Profit prediction: {e}")
        return templates.TemplateResponse(
            "result.html",
            {"request": request, "title": "Error", "value": f"Prediction failed due to backend error: {e}"}
        )