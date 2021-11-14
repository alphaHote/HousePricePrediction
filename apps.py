from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import uvicorn
from regression_model import predict
import pandas as pd

app=FastAPI()

templates = Jinja2Templates(directory="templates")

data= pd.read_csv('test.csv')


from sklearn.linear_model import LogisticRegression
import pandas as pd 
import pickle

class houses(BaseModel):
	MSSubClass:int
	MSZoning:object
	LotFrontage:float
	LotShape:object
	LandContour:object
	LotConfig:object
	Neighborhood:float
	OverallQual:float
	OverallCond:float
	YearRemodAdd:int
	RoofStyle:object
	Exterior1st:object
	ExterQual:object
	Foundation:object
	stFlrSF:int
	ndFlrSF:int
	GrLivArea:int
	BsmtFullBath:float
	HalfBath:int
	KitchenQual:object
	TotRmsAbvGrd:int
	Functional:object
	Fireplaces:int
	WoodDeckSF:int
	ScreenPorch:int
	SaleCondition:object

	class Config:
		arbitrary_types_allowed = True

#model=pickle.load(open('regression_model_output_v0.0.1.pkl','rb'))

@app.get("/")
def home():
	return {'ML model to predict house price '}

@app.post('/make_predictions')
async def make_predictions():
	return predict.make_prediction(input_data=data)

@app.get("/prediction/{id}", response_class=HTMLResponse)
async def read_item(request: Request, id: str):
    return templates.TemplateResponse("iteml.html",{"request": request})

if __name__=="__main__":
	uvicorn.run("apps:app",host="0.0.0.0",port=8080,reload=True)
