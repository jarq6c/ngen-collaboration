"""Explore IBTrACS data.
https://www.ncei.noaa.gov/products/international-best-track-archive
"""
from pathlib import Path
from typing import Literal
import xarray as xr
import pandas as pd
import geopandas as gpd

GLOBAL_CRS: Literal["ESRI:102010"] = "ESRI:102010"
"""Distance preserving CRS string."""

def list_unique(s: pd.Series) -> str:
    """Returns all values as a comma-delimited list."""
    return ",".join(s.unique())

def main(
        cyclone_tracks: Path = Path("IBTrACS.ALL.v04r01.nc"),
        boundary_source: Path = Path("rfc_nwm_domain_boundaries_4326.geojson"),
        output_file: Path = Path("rfc_tropical_storms.md")
) -> None:
    """Main."""
    ds = xr.open_dataset(cyclone_tracks)

    # Extract distance to land, nature, and categories
    variables = ["dist2land", "name", "nature", "usa_sshs"]
    df = ds[variables].to_dataframe()

    # Determine life-time maximum intensities
    lmi = df[["usa_sshs"]].dropna().reset_index().groupby("storm").max()
    lmi["usa_sshs"] = lmi["usa_sshs"].apply("{:.0f}".format)

    # Drop NaN
    df = df[~df["dist2land"].isna()]

    # Retain near land cyclones (less than 400 km)
    df = df[df["dist2land"] < 400]

    # Limit to post-AORC period
    df = df[df["time"] >= pd.Timestamp("1979-01-01T00:00:00.0")]

    # String conversions
    df["name"] = df["name"].str.decode("utf-8")
    df["nature"] = df["nature"].str.decode("utf-8")
    df["usa_sshs"] = df["usa_sshs"].map("{:.0f}".format)

    # Add geometry
    df["geometry"] = gpd.points_from_xy(
        x=df["lon"],
        y=df["lat"],
        crs="EPSG:4326"
    )

    # Project to distance preserving CRS
    gdf = gpd.GeoDataFrame(df.reset_index()).to_crs(GLOBAL_CRS)

    # Load RFC boundaries
    boundaries = gpd.read_file(boundary_source).to_crs(GLOBAL_CRS)

    # Handle duplicate VI entries
    boundaries.iloc[14, 1] = "VI1"
    boundaries.iloc[16, 1] = "VI2"
    boundaries.iloc[17, 1] = "VI3"

    columns: list[str]  = []
    for row in boundaries.itertuples():
        # Column
        col = f"{row.rfc}_{row.domain}"
        columns.append(col)
        distances = gdf.geometry.distance(row.geometry)
        gdf[col] = distances

    # Generate tables
    includes = ["storm", "name", "time", "nature", "usa_sshs"]
    output = "# Tropical storms by RFC\n\n"
    for col in columns:
        # Retain locations within 400,000 m (400 km)
        data = gdf.loc[gdf[col] < 400_000.0, includes].groupby("storm").agg(
            Name=pd.NamedAgg(column="name", aggfunc="first"),
            Start=pd.NamedAgg(column="time", aggfunc="min"),
            End=pd.NamedAgg(column="time", aggfunc="max"),
            Designations=pd.NamedAgg(column="nature", aggfunc=list_unique),
            Categories=pd.NamedAgg(column="usa_sshs", aggfunc=list_unique)
        ).reset_index().rename(columns={"storm": "Storm ID"})
        data["Start"] = data["Start"].dt.strftime("%Y-%m-%d %H:%M")
        data["End"] = data["End"].dt.strftime("%Y-%m-%d %H:%M")
        data["Lifetime Maximum Intensity"] = data["Storm ID"].map(lmi["usa_sshs"])

        output += f"## {col}\n\n"
        output += data.to_markdown() + "\n\n"

    with output_file.open(mode="w", encoding="utf-8") as fo:
        fo.write(output)

if __name__ == "__main__":
    main()
