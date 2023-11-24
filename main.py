from fastapi import FastAPI
from pydantic import BaseModel
from prediction import predict_disease,similar

class Input(BaseModel):
    disease:list

app=FastAPI()

@app.get('/')
def root():
    return {"message":"Root"}

@app.post('/disease')
def create_item(input:Input):
    try:
        result=predict_disease(input.disease)
        return result
    except Exception as e:
        return ({'error':str(e)}),500
    
@app.post('/similar_disease')
def predict_similar(input:Input):
    try:
        result=predict_disease(input.disease[0])
        return result
    except Exception as e:
        return ({'error':str(e)}),500

