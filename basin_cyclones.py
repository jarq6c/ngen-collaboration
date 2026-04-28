"""Explore IBTrACS data.
https://www.ncei.noaa.gov/products/international-best-track-archive
"""
from pathlib import Path
from typing import Literal, Optional
import xarray as xr
import pandas as pd
import geopandas as gpd
from shapely import Polygon, MultiPolygon

GLOBAL_CRS: Literal["ESRI:102010"] = "ESRI:102010"
"""Distance preserving CRS string."""

def list_unique(s: pd.Series) -> str:
    """Returns all values as a comma-delimited list."""
    return ",".join(s.unique())

def load_IBTrACS(
        ifile: Path,
        ofile: Path,
        mask: Optional[Polygon | MultiPolygon] = None
) -> gpd.GeoDataFrame:
    """Load, process, and return GeoDataFrame from IBTrACS NetCDF."""
    # Check for preprocessed file
    if ofile.exists():
        return gpd.read_file(ofile, mask=mask)

    ds = xr.open_dataset(ifile)

    # Extract distance to land, nature, and categories
    variables = ["dist2land", "name", "nature", "usa_sshs"]
    df = ds[variables].to_dataframe()

    # Determine life-time maximum intensities
    lmi = df[["usa_sshs"]].dropna().reset_index().groupby("storm").max()

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

    # Add LMI
    gdf["lmi"] = gdf["storm"].map(lmi["usa_sshs"])

    # Save
    gdf.to_file(ofile, driver="GPKG")
    return gpd.read_file(ofile, mask=mask)

def main(
        cyclone_tracks: Path = Path("IBTrACS.ALL.v04r01.nc"),
        boundary_source: Path = Path("nwm_usgs_assimilation_gages_basin_boundaries_epsg_4269.gpkg"),
        output_file: Path = Path("usgs_basin_tropical_storms.parquet"),
        cyclone_gpkg: Path = Path("data") / "IBTrACS.ALL.v04r01.gpkg"
) -> None:
    """Main."""
    # Pre-process cyclone tracks
    load_IBTrACS(
        cyclone_tracks,
        cyclone_gpkg
    )

    # Load basin boundaries
    boundaries: gpd.GeoDataFrame = gpd.read_file(boundary_source).to_crs(GLOBAL_CRS)
    number_of_basins = boundaries["geometry"].count()

    # Find tracks within 400 km of boundaries
    dfs: list[pd.DataFrame] = []
    for _, provider_id, catchment in boundaries.itertuples():
        # Track progress
        print(provider_id, f"{len(dfs)} / {number_of_basins}")

        # Retrieve locations within 400,000 m (400 km)
        catchment_points = load_IBTrACS(
            cyclone_tracks,
            cyclone_gpkg,
            mask=catchment.buffer(400_000.0)
        )

        # Check for empty
        if catchment_points.empty:
            continue

        # Aggregate storms
        catchment_storms = catchment_points.groupby("storm").agg(
            name=pd.NamedAgg(column="name", aggfunc="first"),
            impact_start=pd.NamedAgg(column="time", aggfunc="min"),
            impact_end=pd.NamedAgg(column="time", aggfunc="max"),
            impact_designations=pd.NamedAgg(column="nature", aggfunc=list_unique),
            impact_intensity=pd.NamedAgg(column="usa_sshs", aggfunc=list_unique),
            lifetime_max_intensity=pd.NamedAgg(column="lmi", aggfunc="first")
        ).reset_index().rename(columns={"storm": "storm_id"})

        # Add site code
        catchment_storms["usgs_site_code"] = provider_id

        # Add to list
        dfs.append(catchment_storms)

    # Concat
    data = pd.concat(dfs, ignore_index=True)
    print(data)
    data.to_parquet(output_file)

if __name__ == "__main__":
    main()
