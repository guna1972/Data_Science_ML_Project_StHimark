                                                    Final Project Report

Project Objective
The primary goal of this project is to analyze radiation contamination levels in St. Himark using data from mobile and static radiation sensors, incorporating wind direction and speed data, and geolocating the results onto a map. 

Workflow of the Code
The code workflow is broken into several stages, ensuring alignment with the project's objectives and requirements. Below is the detailed breakdown:

1. Data Loading
Input Files:
•	MobileSensorReadings.csv: Contains timestamped radiation measurements from mobile sensors.
•	StaticSensorReadings.csv: Contains timestamped radiation measurements from static sensors.
•	StaticSensorLocations.csv: Provides geolocation data (latitude, longitude) for static sensors.
•	WindDirection.csv: Contains wind direction and speed data with timestamps.
•	StHimarkNeighborhoodShapefile: A shapefile representing the geometry of St. Himark neighborhoods.
•	StHimarkNeighborhoodMap.png: A labeled map image of St. Himark for visualization.
Process:
•	Load all the above datasets using pandas and geopandas for handling tabular and spatial data.
•	Ensure the shapefile's coordinate system is standardized (e.g., WGS84 for latitude and longitude).

2. Preprocessing
Steps:
1.	Datetime Conversion:
o	Convert all timestamps to datetime format for consistency and easy matching between datasets.
2.	Handling Missing/Corrupted Data:
o	Drop invalid or missing rows for radiation values and coordinates (longitude/latitude).
o	Retain only significant radiation values (greater than zero).
3.	Wind Data Association:
o	Match wind speed and direction to each mobile sensor reading using the nearest timestamp via merge_asof.

Output:
•	Cleaned and reduced datasets for mobile and static sensors, along with wind data integrated into the mobile sensor readings.

3. Gaussian Plume Heatmap Creation
Goal: To model radiation dispersion using a Gaussian Plume Model, where each radiation source contributes a spread of contamination across its surrounding area.
Steps:
1.	Grid Creation:
o	Define a grid over the St. Himark area using the bounding box of the shapefile. The grid size (e.g., 200x200) determines the resolution of the heatmap.
2.	Iterative Heatmap Computation:
o	For each radiation source (mobile sensor data):
	Calculate a Gaussian spread based on radiation intensity (Value).
	Adjust the spread using wind speed and direction from the wind data.
o	Accumulate contributions from all radiation sources to form the heatmap.
3.	Normalization:
o	Smooth the heatmap using scipy's gaussian_filter.
o	Normalize values to the range [0, 1] for easy visualization.

4. Overlay Heatmap onto St. Himark Map
Goal: Visualize the computed heatmap on the provided St. Himark neighborhood map (StHimarkNeighborhoodMap.png) along with key landmarks.
Steps:
1.	Base Map Alignment:
o	Load the map image (StHimarkNeighborhoodMap.png) as the base layer.
o	Use the shapefile's bounding box (neighborhoods.total_bounds) to align the map with the heatmap grid.
2.	Heatmap Overlay:
o	Use matplotlib's contourf to overlay the heatmap onto the base map with transparency (alpha).
3.	Add Landmarks:
o	Plot hospital locations and the nuclear plant using their provided coordinates.
o	Use distinct markers for differentiation:
	Blue circles for hospitals.
	Yellow star for the nuclear plant.
4.	Add Colorbar and Labels:
o	Include a colorbar to indicate normalized radiation levels.
o	Label the axes and provide a title for the visualization.
Output:
•	A comprehensive radiation contamination map with hotspots, overlaid on the labeled St. Himark neighborhood map.

Conclusion
This project successfully integrates spatial data, radiation sensor readings, wind conditions, and geospatial visualization techniques to deliver a meaningful analysis of radiation contamination levels in St. Himark. The outputs not only provide actionable insights but also meet the final project requirements comprehensively.
 

