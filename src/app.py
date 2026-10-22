import math
import warnings

from fastapi import FastAPI, HTTPException

import config
from data_loader import DataLoader
from pipeline import process_trafos

warnings.filterwarnings('ignore')

app = FastAPI(title="LV battery sizing API")


@app.get("/")
def read_root():
    return {"message": "Welcome to the LV battery sizing API"}


def sanitize_records(df):
    """Converts dataframe to a list of records, NaN, inf and -inf are replaced with None (JSON null)"""
    return [{
        key: None if isinstance(value, float) and not math.isfinite(value) else value
        for key, value in record.items()
    } for record in df.to_dict(orient="records")]


@app.post("/get_results/")
@app.get("/get_results/")
def get_results(number_of_trafos: int = None, all_feeders: bool = False):
    """
    Finds transformer candidates and determines optimal battery parameters for each feeder.
     Args:
        --------
            number_of_trafos: int
                number of transformers to process
            all_feeders: bool
                if True, results for all feeders are returned

        Returns:
        --------
            list of results for feeders, where battery is needed
    """
    try:
        trafos_list = DataLoader().find_trafo_candidates()
        if number_of_trafos is not None:
            trafos_list = trafos_list[:number_of_trafos]
        battery_res = process_trafos(trafos_list, config.NETWORK_PATH,
                                     all_feeders=all_feeders)
        return sanitize_records(battery_res)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
