import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly

# Groundwater salinity data from recent work.
df = pd.read_csv("model-grid-subsample.csv")
df = df[df.dem_m > df.zkm * 1e3] # Remove data above the ground.

grid = df[['xkm', 'ykm', 'zkm', 'mean_tds']].to_numpy()
x = grid[:, 0] * 1e3 # Kilometers to meters.
y = grid[:, 1] * 1e3
z = grid[:, 2] * 1e3
u = grid[:, 3]


# Digital Elevation Model (DEM) - the land surface in the study area.
dem_grid = df[['xkm', 'ykm', 'dem_m']].to_numpy()

x_dem = dem_grid[:, 0] * 1e3
y_dem = dem_grid[:, 1] * 1e3
z_dem = dem_grid[:, 2]


# Interpolate the land surface point data.
xi_dem = np.linspace(min(x_dem), max(x_dem), 200)
yi_dem = np.linspace(min(y_dem), max(y_dem), 200)
zi_dem = griddata((x_dem, y_dem), z_dem, 
                  (xi_dem.reshape(1, -1), yi_dem.reshape(-1, 1)))


# Make the 3-d graph.
trace_dem = go.Surface(x=xi_dem, y=yi_dem, z=zi_dem,
                       colorscale='Earth',
                       name='Land surface',
                       showscale=False,
                       showlegend=True
                         )

trace_groundwater = go.Scatter3d(
    x=x, y=y, z=z, 
    mode='markers',
    name='Groundwater salinity',
    showlegend=True,
    marker=dict(size=3, symbol='square', 
                colorscale='RdYlBu_r', color=np.log10(u), # Log the colorscale.
                colorbar=dict(title=dict(text='Salinity (mg/L)', side='right'),
                                                        x=0.94,  # Move cbar over.
                                                        len=0.5,  # Shrink cbar.
                                                        ticks='outside',
                              tickvals=np.log10([400, 500, 600, 700, 800, 900, 1000, 
                                                 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]),
                              ticktext=[400, 500, 600, 700, 800, 900, 1000, 
                                        2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                                                        ))
)

data = [
    trace_dem,
    trace_groundwater
]

fig = go.Figure(data=data)

fig.update_layout(
    margin=dict(l=20, r=50, b=20, t=20),
    scene=dict(
             xaxis=dict(title='Easting (m)', color='black', showbackground=True, backgroundcolor='gray'), 
             yaxis=dict(title='Northing (m)', color='black', showbackground=True, backgroundcolor='gray'), 
             zaxis=dict(title='Elevation (m)', color='black', showbackground=True, backgroundcolor='gray'), 
             aspectratio=dict(x=1, y=1, z=0.25), # Scale the z-direction.
             camera = dict( # Make north pointing up.
                        up=dict(x=0, y=0, z=1),
                        center=dict(x=0, y=0, z=-0.2),
                        eye=dict(x=-1., y=-1.3, z=1.)
                        )
                    ),
    legend=dict(x=0, y=0.8,
               font=dict(size=13, color='black'),
               bgcolor='rgb(230,230,230)',
               bordercolor='black',
               borderwidth=2,
               title='<b> Explanation </b><br> (click each to toggle) <br>'
                              ),
    )


plotly.offline.plot(fig, filename='3d-salinity-example.html')
