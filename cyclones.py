"""Explore IBTrACS data.
https://www.ncei.noaa.gov/products/international-best-track-archive
"""
import xarray as xr
import pandas as pd
import geopandas as gpd

def list_unique(s: pd.Series) -> str:
    """Returns all values as a comma-delimited list."""
    return ",".join(s.unique())

def main() -> None:
    """Main."""
    ds = xr.open_dataset("IBTrACS.ALL.v04r01.nc")

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
    crs: str = "ESRI:102010"
    gdf = gpd.GeoDataFrame(df.reset_index()).to_crs(crs)

    # Load RFC boundaries
    rfc_boundaries = gpd.read_file(
        "rfc_nwm_domain_boundaries_4326.geojson").to_crs(crs)

    # Handle duplicate VI entries
    rfc_boundaries.iloc[14, 1] = "VI1"
    rfc_boundaries.iloc[16, 1] = "VI2"
    rfc_boundaries.iloc[17, 1] = "VI3"

    columns: list[str]  = []
    for row in rfc_boundaries.itertuples():
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

    with open("rfc_tropical_storms.md", mode="w", encoding="utf-8") as fo:
        fo.write(output)

if __name__ == "__main__":
    main()
