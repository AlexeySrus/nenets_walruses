from sqlmodel import SQLModel, Session, Field, create_engine, select
from typing import Optional
from datetime import datetime
from PIL import Image, ExifTags
from uuid import uuid4
from pathlib import Path
import os


class Photo(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: int = Field(default=None, primary_key=True)
    filepath: str
    fname: str
    created: datetime = Field(default_factory=datetime.now, nullable=False)
    latitude: Optional[float]
    longitude: Optional[float]


class WalrusCoord(SQLModel, table=True):
    __table_args__ = {"extend_existing": True}
    id: int = Field(default=None, primary_key=True)
    photo_id: int = Field(default=None, foreign_key='photo.id')
    x: float
    y: float
    is_young: bool = Field(default=False)


sqlite_filename = 'database.db'
sqlite_url = f'sqlite:///{sqlite_filename}'
engine = create_engine(sqlite_url, echo=False)
IMAGES_STORAGE_FOLDER = './images'


def create_db_and_tables():
    SQLModel.metadata.create_all(engine)


def get_exif_data(image):
    """Returns a dictionary from the exif data of an PIL Image item. Also converts the GPS Tags"""
    exif_data = {}
    info = image._getexif()
    if info:
        for tag, value in info.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_data = {}
                for t in value:
                    sub_decoded = ExifTags.GPSTAGS.get(t, t)
                    gps_data[sub_decoded] = value[t]

                exif_data[decoded] = gps_data
            else:
                exif_data[decoded] = value
    return exif_data


def _get_if_exist(data, key):
    if key in data:
        return data[key]

    return None


def _convert_to_degress(value):
    deg = value[0]
    minute = value[1]
    sec = value[2]
    return deg + (minute / 60.0) + (sec / 3600.0)


def get_lat_lon(exif_data):
    """Returns the latitude and longitude, if available, from the provided exif_data (obtained through get_exif_data above)"""
    lat = None
    lon = None

    if "GPSInfo" in exif_data:
        gps_info = exif_data["GPSInfo"]

        gps_latitude = _get_if_exist(gps_info, "GPSLatitude")
        gps_latitude_ref = _get_if_exist(gps_info, 'GPSLatitudeRef')
        gps_longitude = _get_if_exist(gps_info, 'GPSLongitude')
        gps_longitude_ref = _get_if_exist(gps_info, 'GPSLongitudeRef')

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degress(gps_latitude)
            if gps_latitude_ref != "N":
                lat = 0 - lat

            lon = _convert_to_degress(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon

    return lat, lon

def add_photo(image: Image, fname: str, session: Session):
    uid = uuid4()
    ext = os.path.splitext(fname)[1]
    fpath = (Path(IMAGES_STORAGE_FOLDER) / f'{uid}{ext}').absolute()
    exif_data = get_exif_data(image)
    lat, lon = get_lat_lon(exif_data)

    image.save(fpath)
    photo = Photo(
        filepath=str(fpath),
        fname=fname,
        latitude=lat,
        longitude=lon
    )
    session.add(photo)
    session.commit()
    return photo


def add_walrus_coords(coords: list, photo: Photo, session: Session):
    for x, y in coords:
        session.add(WalrusCoord(photo_id=photo.id, x=x, y=y))
    session.commit()


coords = [
    [0.2, 0.2],
    [0.6, 0.6],
]

import pandas as pd
from sqlalchemy.orm import load_only

def create_df():
    with Session(engine) as session:
        photos = session.exec(
            select(Photo).options(load_only('id', 'fname', 'created', 'latitude', 'longitude'))
            # select(
            #     Photo.fname,
            #     Photo.created,
            #     Photo.latitude,
            #     Photo.longitude,
            # )
        ).all()
    # print(photos)
    df = pd.DataFrame(list(map(lambda x: x.dict(), photos)))
    # df.set_index('id', inplace=True)
    df = df[['id', 'fname', 'created', 'latitude', 'longitude']]
    df.sort_values(by=['created'], inplace=True)
    return df
        # for ph in photos:
        #     print(ph.json())


def get_path(photo_id):
    with Session(engine) as session:
        photo = session.exec(
            select(Photo).where(Photo.id == photo_id)
        ).one()
        if photo is not None:
            return photo.filepath
        else:
            raise RuntimeError(f'Wrong id: {photo}')


def get_points(photo_id: int):
    with Session(engine) as session:
        coords = session.exec(
            select(WalrusCoord).where(WalrusCoord.photo_id == photo_id)
        ).all()
    return coords

    # if photo is not None:

def main():
    # df = create_df()
    # print(df)
    # create_db_and_tables()

    image = Image.open('/home/fateev/dev/walruses/DJI_0005 (2).jpg')

    # with Session(engine) as session:
    #     ph = add_photo(image, 'walruses.jpg', session)
    #     add_walrus_coords(coords, ph, session)


    # create_photos()
    # select_photos()


if __name__ == '__main__':
    # print(datetime.utcnow())
    main()
