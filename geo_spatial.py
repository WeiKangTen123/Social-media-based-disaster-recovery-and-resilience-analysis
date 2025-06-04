import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

def plot_disaster_post_distribution(df, shapefile_path, save_path="shapefile/disaster_post_distribution.png"):
    """
    Plots a choropleth map showing the distribution of disaster-related social media posts per location.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("❌ Expected a Pandas DataFrame, but got a different type.")

    if "Location" not in df.columns:
        raise KeyError("❌ Error: 'Location' column is missing in DataFrame.")

    df_geo = df.copy()

    # Ensure 'Location' is a string before processing
    df_geo["Location"] = df_geo["Location"].astype(str).str.strip().str.lower()

    # Load the UK shapefile
    gdf_map = gpd.read_file(shapefile_path).explode(index_parts=False)
    gdf_map["CTYUA23NM"] = gdf_map["CTYUA23NM"].str.strip().str.lower()

    # Filter location counts to only show relevant locations
    relevant_locations = gdf_map["CTYUA23NM"].unique()
    location_counts = df_geo[df_geo["Location"].isin(relevant_locations)]
    location_counts = location_counts.groupby("Location").size().reset_index(name="post_count")

    # Display filtered location post counts in Streamlit
    st.subheader("📌 Disaster-related posts count by relevant locations:")
    st.dataframe(location_counts)

    # Merge geospatial boundary data with filtered post counts
    gdf_map = gdf_map.merge(location_counts, left_on="CTYUA23NM", right_on="Location", how="left")

    # Fill missing values with 0
    gdf_map["post_count"] = gdf_map["post_count"].fillna(0).astype(int)

    # Plot the map
    fig, ax = plt.subplots(figsize=(14, 10))
    gdf_map.plot(column="post_count", cmap="Blues", edgecolor="black", legend=True, ax=ax, alpha=0.85)
    ax.set_title("Geospatial Distribution of Disaster-Related Social Media Posts", fontsize=14)
    ax.axis("off")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)  # Ensure figure is closed before displaying
    
    # Display image in Streamlit
    with st.container():  # Prevent automatic detection outside the container
        st.image(save_path, caption="Geospatial Distribution of Disaster-Related Social Media Posts")
