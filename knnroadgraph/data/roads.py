import fiona
from shapely.geometry import Point

from .base import FionaRoadLoader


class FlandersRoadLoader(FionaRoadLoader):
    def __init__(self, file_name, municipality):
        super().__init__(file_name=file_name, transform=None)
        self.municipality = municipality

    def _load_lines(self):
        roads_processed = []
        geo_data = fiona.open(self.file_name)
        for road in geo_data:
            if (
                road["properties"]["LGEMEENTE"] != self.municipality
                and road["properties"]["RGEMEENTE"] != self.municipality
            ):
                continue
            roads_processed.append(
                [Point(point).coords[0] for point in road["geometry"]["coordinates"]]
            )
        return roads_processed
