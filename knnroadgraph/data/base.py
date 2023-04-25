import logging

import fiona
import overpass
import pyproj
from shapely import ops
from shapely.geometry import LineString, MultiLineString, Point

POSTAL_CODES_ANTWERP = [
    "2000",
    "2018",
    "2020",
    "2030",
    "2040",
    "2050",
    "2060",
    "2100",
    "2140",
    "2150",
    "2170",
    "2180",
    "2600",
    "2610",
    "2660",
]
POSTAL_CODES_BRUSSELS = (
    "1007",
    "1000",
    "1020",
    "1030",
    "1040",
    "1050",
    "1060",
    "1070",
    "1080",
    "1090",
    "1120",
    "1130",
    "1140",
    "1150",
    "1160",
    "1170",
    "1180",
    "1190",
    "1200",
    "1210",
    "1081",
    "1082",
    "1083",
)


class DataLoader:
    def __init__(self, transform=None):
        self.transformer = None
        self.transformation = None

        # NOTE: Transform keys: (
        #   fromCRS, toCRS,
        #   reverseXYBefore, reverseXYAfter,
        #   invertXBefore, invertYBefore,
        #   invertXAfter, invertYAfter
        # )

        if transform == "to_belgian":
            transform = {
                "fromCRS": pyproj.CRS("EPSG:4326"),
                "toCRS": pyproj.CRS("EPSG:31370"),
            }
        elif transform == "to_us":
            transform = {
                "fromCRS": pyproj.CRS("EPSG:4326"),
                "toCRS": pyproj.CRS("EPSG:4414"),
                "reverseXYAfter": True,
                "invertYAfter": True,
            }
        if transform:
            if transform["toCRS"].coordinate_system.name != "cartesian":
                logging.warning(
                    "Transforming to a non-cartesian CRS. "
                    "This will cause issues for techniques assuming cartesian spaces"
                )
            self.transformer = pyproj.Transformer.from_crs(
                transform["fromCRS"], transform["toCRS"], always_xy=True
            )
            self.transformation = transform

    def transform(self, lon, lat):
        if not self.transformer:
            return lon, lat
        if self.transformation.get("reverseXYBefore"):
            lon, lat = lat, lon
        if self.transformation.get("invertXBefore"):
            lon = -lon
        if self.transformation.get("invertYBefore"):
            lat = -lat
        x, y = self.transformer.transform(lon, lat)
        tx, ty = (x, y) if not self.transformation.get("reverseXYAfter") else (y, x)
        if self.transformation.get("invertXAfter"):
            tx = -tx
        if self.transformation.get("invertYAfter"):
            ty = -ty
        return tx, ty

    def load(self):
        raise NotImplementedError()


class PointTargetLoader(DataLoader):
    pass


class MultiLineStringLoader(DataLoader):
    pass


class PointTargetDataFrameLoader(PointTargetLoader):
    def __init__(
        self,
        file_name,
        transform=None,
        lon_nm="lon",
        lat_nm="lat",
        price_nm="discount_price",
    ):
        super().__init__(transform)
        self.file_name = file_name
        self.lon_nm = lon_nm
        self.lat_nm = lat_nm
        self.price_nm = price_nm

    def load(self):
        df = self._load_dataframe()
        points = []
        targets = []
        for lon, lat, target in zip(
            df[self.lon_nm], df[self.lat_nm], df[self.price_nm]
        ):
            points.append(Point(self.transform(lon, lat)))
            targets.append(target)
        return points, targets

    def _load_dataframe(self):
        raise NotImplementedError()


class HouseLoader(PointTargetDataFrameLoader):
    def __init__(
        self,
        file_name,
        postal_codes,
        transform=None,
        lon_nm="lon",
        lat_nm="lat",
        price_nm="discount_price",
    ):
        super().__init__(file_name, transform, lon_nm, lat_nm, price_nm)
        self.postal_codes = postal_codes


class RoadLoader(MultiLineStringLoader):
    def load(self):
        lines = self._load_lines()
        lines_processed = []
        for line in lines:
            line_processed = []
            for point in line:
                line_processed.append(Point(self.transform(point[0], point[1])))
            lines_processed.append(LineString(line_processed))
        mls = MultiLineString(lines_processed)
        mls = ops.transform(lambda *args: args[:2], mls)
        return mls

    def _load_lines(self):
        raise NotImplementedError()


class FionaRoadLoader(RoadLoader):
    def __init__(self, file_name, transform=None):
        super().__init__(transform)
        self.file_name = file_name

    def _load_lines(self):
        geo_data = fiona.open(self.file_name)
        roads_processed = []
        for road in geo_data:
            if road["geometry"]["type"] == "LineString":
                roads_processed.append(
                    [
                        Point(point).coords[0]
                        for point in road["geometry"]["coordinates"]
                    ]
                )
            elif road["geometry"]["type"] == "MultiLineString":
                for line in road["geometry"]["coordinates"]:
                    roads_processed.append([Point(point).coords[0] for point in line])
            else:
                logging.warning(
                    "Unsupported geometry type encountered: "
                    f"{road['geometry']['type']}, ignoring"
                )
        return roads_processed


class OverpassRoadLoader(RoadLoader):
    def __init__(self, bbox, transform=None):
        super().__init__(transform)
        self.bbox = bbox

    def _load_lines(self):
        roads_processed = {}
        api = overpass.API(timeout=600)
        response = api.get(
            f'way({self.bbox[1]},{self.bbox[0]},{self.bbox[3]},{self.bbox[2]})[highway~"^(motorway|motorway_link|trunk|trunk_link|primary|secondary|tertiary|unclassified|residential)$"];',
            verbosity="geom",
        )
        for way in response.features:
            if way["geometry"]["type"] != "LineString":
                continue
            roads_processed[way.id] = way["geometry"]["coordinates"]
        return roads_processed.values()
