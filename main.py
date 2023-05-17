from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import pickle


app = FastAPI()


class modelShema2(BaseModel):
    creditscore:int
    age:int
    tenure:int
    balance:int
    numofproducts:int
    estimatedsalary:int
    


@app.get("/")
def home():
    return {"mesaj": "Logistic Regression Ml model için predict/log_reg kısmına gidin"}

@app.post("/predict/log_reg")
async def predict_logreg_ml(predict_value:modelShema2):
    filename = "logreg1_model.pkl"
    load_model = pickle.load(open(filename, "rb"))
    
    df = pd.DataFrame(
        [predict_value.dict().values()],
        columns=predict_value.dict().keys()
    )


    predict = load_model.predict(df)
    if int(predict[0]) == 0:
        result = "Not_Churn"  # Tahmin 0 ise "churn x" mesajı döndürülür
    else:
        result = "Churn"  # Tahmin 0 değilse "churn y" mesajı döndürülür
    
    return {"Prediction": int(predict[0]) , "Result": result}

