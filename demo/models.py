import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import pandas as pd
from PIL import Image, ExifTags
from sqlalchemy.orm import load_only
from sqlmodel import SQLModel, Session, Field, create_engine, select

SQLITE_FILENAME = 'database.db'
SQLITE_URL = f'sqlite:///{SQLITE_FILENAME}'
ENGINE = create_engine(SQLITE_URL, echo=False)
IMAGES_STORAGE_FOLDER = './images'


class Photo(SQLModel, table=True):
    """Data model for the photo entity."""

    __table_args__ = {"extend_existing": True}
    id: int = Field(default=None, primary_key=True)
    filepath: str
    fname: str
    created: datetime = Field(default_factory=datetime.now, nullable=False)
    latitude: Optional[float]
    longitude: Optional[float]


class WalrusCoord(SQLModel, table=True):
    """Data model for the walrus'es coordinates instance."""

    __table_args__ = {"extend_existing": True}
    id: int = Field(default=None, primary_key=True)
    photo_id: int = Field(default=None, foreign_key='photo.id')
    x: int
    y: int
    is_young: bool = Field(default=False)


def add_walrus_coords(
        coords: list, classes: list, photo: Photo, session: Session
):
    """Add and new record with walruses coordinates."""
    for (x, y), cls in zip(coords, classes):
        session.add(WalrusCoord(photo_id=photo.id, x=x, y=y, is_young=cls))
    session.commit()


def add_photo(image: Image, fname: str, session: Session):
    """Add and new record in Photo table."""
    uid = uuid4()
    ext = os.path.splitext(fname)[1]
    folder = Path(IMAGES_STORAGE_FOLDER)
    if not folder.exists():
        folder.mkdir(parents=True)
        print(f'Created directory: {folder}')
    fpath = (folder / f'{uid}{ext}').absolute()
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


def add_record_in_db(image: Image, fname:str, coords, classes):
    """Add new data into WalrusCoords and Photo tables."""
    with Session(ENGINE) as session:
        ph = add_photo(image, fname, session)
        add_walrus_coords(coords, classes, ph, session)


def create_db_and_tables():
    """Create table."""
    SQLModel.metadata.create_all(ENGINE)


def check_db_exists():
    """Check if table is exist."""
    return os.path.exists(SQLITE_FILENAME)


def get_exif_data(image):
    """
    Returns a dictionary from the exif data of an PIL Image item.

    Also converts the GPS Tags.
    """
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

    if 'GPSInfo' in exif_data:
        gps_info = exif_data['GPSInfo']

        gps_latitude = gps_info.get('GPSLatitude', None)
        gps_latitude_ref = gps_info.get('GPSLatitudeRef', None)
        gps_longitude = gps_info.get('GPSLongitude', None)
        gps_longitude_ref = gps_info.get('GPSLongitudeRef', None)

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            lat = _convert_to_degress(gps_latitude)
            if gps_latitude_ref != "N":
                lat = 0 - lat

            lon = _convert_to_degress(gps_longitude)
            if gps_longitude_ref != "E":
                lon = 0 - lon

    return lat, lon


def create_df():
    """Create a Pandas DataFrame from the database."""
    with Session(ENGINE) as session:
        photos = session.exec(
            select(Photo).options(
                load_only('id', 'fname', 'created', 'latitude', 'longitude')
            )
        ).all()
    data = list(map(lambda x: x.dict(), photos))
    for i in range(len(data)):
        walruses_coords = get_points(data[i]['id'])
        data[i]['walruses_count'] = len(walruses_coords)
    df = pd.DataFrame(data)
    df = df[
        ['id', 'fname', 'created', 'walruses_count', 'latitude', 'longitude']
    ]
    df.sort_values(by=['created'], inplace=True)
    return df


def get_path(photo_id):
    """Get a path to the source image of photo id."""
    with Session(ENGINE) as session:
        photo = session.exec(
            select(Photo).where(Photo.id == photo_id)
        ).one()
        if photo is not None:
            return photo.filepath
        else:
            raise RuntimeError(f'Wrong id: {photo}')


def get_points(photo_id: int):
    """Get all walruses coordinates by given photo."""
    with Session(ENGINE) as session:
        coords = session.exec(
            select(WalrusCoord).where(WalrusCoord.photo_id == photo_id)
        ).all()
    return coords
