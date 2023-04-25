import pandas as pd

from .base import HouseLoader


class InsideAirbnbLoader(HouseLoader):
    def __init__(self, file_name):
        super().__init__(file_name=file_name, postal_codes=None, transform="to_belgian")

    def _load_dataframe(self):
        df = pd.read_csv(self.file_name)
        df = df[~df["price"].isnull()]
        df.loc[df["bedrooms"].isnull(), "bedrooms"] = 1
        df["price"] = (
            df["price"]
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.replace(" ", "", regex=False)
            .astype(float)
        )
        df = df[df["room_type"] == "Entire home/apt"]
        discount_price = pd.to_numeric(df["price"] / df["bedrooms"], errors="coerce")
        df["discount_price"] = discount_price.round(6)
        df["lat"] = df["latitude"].astype(float)
        df["lon"] = df["longitude"].astype(float)
        df = df[~df["discount_price"].isnull()]
        df = df.drop_duplicates(subset=["lat", "lon"], keep="last")
        return df
