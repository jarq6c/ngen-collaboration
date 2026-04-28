"""Plot TS basins."""
from pathlib import Path
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go

def main(
        storm_basin_filepath: Path,
        gage_location_filepath: Path
) -> None:
    """Main."""
    # Load data
    storms = pd.read_parquet(storm_basin_filepath)
    gages = pd.read_hdf(gage_location_filepath)

    # Count storms
    storm_count = storms.groupby("usgs_site_code").count()
    start_date = storms["impact_start"].min().strftime("%Y-%m-%d")
    end_date = storms["impact_end"].max().strftime("%Y-%m-%d")

    # Plot
    title = (
        "Frequency of potential tropical cyclone impact events per USGS basin<br>"
        f"{start_date} to {end_date}"
    )
    fig = go.Figure([go.Histogram(x=storm_count["storm_id"])])
    fig.update_layout(
        title=title,
        xaxis_title_text="Number of unique storms",
        yaxis_title_text="Number of individual basins",
        bargap=0.1
    )
    # fig.show()

    # Add geometry to gages
    gages["geometry"] = gpd.points_from_xy(
        x=gages["longitude"],
        y=gages["latitude"],
        crs="EPSG:4326"
    )
    gages = gpd.GeoDataFrame(gages)

    # Find gages with storms
    gages = gages[gages["usgs_site_code"].isin(storms["usgs_site_code"].unique())]

    # Add storm count
    gages["storm_count"] = gages["usgs_site_code"].map(storm_count["storm_id"])

    # Save
    gages.to_file("storm_gages.gpkg", driver="GPKG")

if __name__ == "__main__":
    main(
        Path("usgs_basin_tropical_storms.parquet"),
        Path("RouteLink.h5")
    )
