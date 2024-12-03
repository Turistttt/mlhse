from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import pickle

app = FastAPI()

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)


class Item(BaseModel):
    name: str
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class Items(BaseModel):
    objects: List[Item]


def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    # Создал базовый пример состоящий из медианных значений, который будет заполняться на основе входных json, таким образом если что-либо в json'е будет Nan, оно будет автоматически заполнено
    base_example = {'year': 2014.0,
                    'km_driven': 70000.0,
                    'mileage': 19.3,
                    'engine': 1248.0,
                    'max_power': 82.0,
                    'fuel_Diesel': 0.0,
                    'fuel_LPG': 0.0,
                    'fuel_Petrol': 0.0,
                    'seller_type_Individual': 0.0,
                    'seller_type_Trustmark Dealer': 0.0,
                    'transmission_Manual': 0.0,
                    'owner_Fourth & Above Owner': 0.0,
                    'owner_Second Owner': 0.0,
                    'owner_Test Drive Car': 0.0,
                    'owner_Third Owner': 0.0,
                    'seats_4': 0.0,
                    'seats_5': 0.0,
                    'seats_6': 0.0,
                    'seats_7': 0.0,
                    'seats_8': 0.0,
                    'seats_9': 0.0,
                    'seats_10': 0.0,
                    'seats_14': 0.0}

    data = data.fillna('Nan')
    df_processed = pd.DataFrame([base_example for i in range(len(data))])
    for i in range(len(data)):

        df_processed.loc[i, 'year'] = data.loc[i, 'year']
        df_processed.loc[i, 'km_driven'] = data.loc[i, 'km_driven']
        df_processed.loc[i, 'engine'] = data.loc[i, 'km_driven']
        df_processed.loc[i, 'max_power'] = data.loc[i, 'km_driven']

        # Если колонка не пропущенна и не является значением дропунтым при one-hot кодировании - оно получит 1
        if data.loc[i, 'fuel'] not in ['CNG', 'Nan']:
            df_processed.loc[i, f'fuel_{data.loc[i, 'fuel']}'] = 1

        if data.loc[i, 'seller_type'] not in ['Dealer', "Nan"]:
            df_processed.loc[i, f'seller_type_{data.loc[i, 'seller_type']}'] = 1

        if data.loc[i, 'transmission'] not in ['Automatic', "Nan"]:
            data.loc[i, f'transmission_{data.loc[i, 'transmission']}'] = 1

        if data.loc[i, 'owner'] not in ['First Owner', "Nan"]:
            data.loc[i, f'owner_{data.loc[i, 'owner']}'] = 1

        if data.loc[i, 'seats'] not in [2.0, "Nan"]:
            data.loc[i, f'seats_{int(data.loc[i, 'seats'])}'] = 1

    return df_processed


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    df = pd.DataFrame([item.dict()])

    df_processed = preprocess_input(df)

    prediction = model.predict(df_processed)

    return float(prediction[0])


@app.post("/predict_items")
def predict_items(items: List[Item]) -> List[float]:
    df = pd.DataFrame([item.dict() for item in items])

    df_processed = preprocess_input(df)

    predictions = model.predict(df_processed)

    return predictions.tolist()