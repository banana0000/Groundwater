import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly

# --- Load Data ---

# Groundwater salinity data
df = pd.read_csv("model-grid-subsample.csv")
df = df[df.dem_m > df.zkm * 1e3]  # Remove data above ground

x = df['xkm'].to_numpy() * 1e3
y = df['ykm'].to_numpy() * 1e3
z = df['zkm'].to_numpy() * 1e3
u = df['mean_tds'].to_numpy()

# DEM (land surface)
x_dem = df['xkm'].to_numpy() * 1e3
y_dem = df['ykm'].to_numpy() * 1e3
z_dem = df['dem_m'].to_numpy()

# Rock layers
rock1 = pd.read_csv("rock-layer-1.csv")
rock2 = pd.read_csv("rock-layer-2.csv")

# --- Interpolate Surfaces ---

# DEM surface
xi_dem = np.linspace(x_dem.min(), x_dem.max(), 200)
yi_dem = np.linspace(y_dem.min(), y_dem.max(), 200)
zi_dem = griddata(
    (x_dem, y_dem), z_dem,
    (xi_dem[None, :], yi_dem[:, None]),
    method='linear'
)

# Rock layer 1
xi_rock1 = np.linspace(rock1['xkm'].min() * 1e3, rock1['xkm'].max() * 1e3, 200)
yi_rock1 = np.linspace(rock1['ykm'].min() * 1e3, rock1['ykm'].max() * 1e3, 200)
zi_rock1 = griddata(
    (rock1['xkm'] * 1e3, rock1['ykm'] * 1e3),
    rock1['mean_pred'],
    (xi_rock1[None, :], yi_rock1[:, None]),
    method='linear'
)

# Rock layer 2
xi_rock2 = np.linspace(rock2['xkm'].min() * 1e3, rock2['xkm'].max() * 1e3, 200)
yi_rock2 = np.linspace(rock2['ykm'].min() * 1e3, rock2['ykm'].max() * 1e3, 200)
zi_rock2 = griddata(
    (rock2['xkm'] * 1e3, rock2['ykm'] * 1e3),
    rock2['mean_pred'],
    (xi_rock2[None, :], yi_rock2[:, None]),
    method='linear'
)

# --- Salinity Slices ---

# Horizontal slice at z = -100 m
z_slice = -100
xi = np.linspace(x.min(), x.max(), 200)
yi = np.linspace(y.min(), y.max(), 200)
salinity_hslice = griddata(
    (x, y, z), u,
    (xi[None, :, None], yi[:, None, None], np.full((200, 200, 1), z_slice)),
    method='linear'
).squeeze()

# Vertical slice at x = mean(x)
xv = np.mean(x)
yi_v = np.linspace(y.min(), y.max(), 200)
zi_v = np.linspace(z.min(), z.max(), 200)
salinity_xslice = griddata(
    (x, y, z), u,
    (np.full((200, 200), xv), yi_v[:, None], zi_v[None, :]),
    method='linear'
)

# Vertical slice at y = mean(y)
yv = np.mean(y)
xi_v = np.linspace(x.min(), x.max(), 200)
zi_v = np.linspace(z.min(), z.max(), 200)
salinity_yslice = griddata(
    (x, y, z), u,
    (xi_v[:, None], np.full((200, 200), yv), zi_v[None, :]),
    method='linear'
)

# --- Plotly Traces ---

# DEM surface
trace_dem = go.Surface(
    x=xi_dem, y=yi_dem, z=zi_dem,
    colorscale='Earth',
    name='Land surface',
    showscale=False,
    showlegend=True,
    opacity=1.0
)

# Rock layer 1
trace_rock1 = go.Surface(
    x=xi_rock1, y=yi_rock1, z=zi_rock1,
    colorscale='Greys',
    name='Rock Layer 1',
    showscale=False,
    showlegend=True,
    opacity=0.7
)

# Rock layer 2
trace_rock2 = go.Surface(
    x=xi_rock2, y=yi_rock2, z=zi_rock2,
    colorscale='Blues',
    name='Rock Layer 2',
    showscale=False,
    showlegend=True,
    opacity=0.7
)

# Groundwater salinity points
trace_groundwater = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    name='Groundwater salinity',
    showlegend=True,
    marker=dict(
        size=3, symbol='square',
        colorscale='RdYlBu_r', color=np.log10(u),
        colorbar=dict(
            title=dict(text='Salinity (mg/L)', side='right'),
            x=0.94, len=0.5, ticks='outside',
            tickvals=np.log10([400, 1000, 5000, 10000]),
            ticktext=[400, 1000, 5000, 10000]
        )
    )
)

# Horizontal salinity slice
trace_salinity_hslice = go.Surface(
    x=xi, y=yi, z=np.full_like(salinity_hslice, z_slice),
    surfacecolor=np.log10(salinity_hslice),
    colorscale='RdYlBu_r',
    colorbar=dict(
        title=dict(text='Salinity (mg/L)', side='right'),
        x=1.02, len=0.5, ticks='outside',
        tickvals=np.log10([400, 1000, 5000, 10000]),
        ticktext=[400, 1000, 5000, 10000]
    ),
    name=f'Salinity at z={z_slice} m',
    showscale=True,
    opacity=0.8,
    showlegend=True
)

# Vertical salinity slice at x = mean(x)
trace_salinity_xslice = go.Surface(
    x=np.full_like(salinity_xslice, xv),
    y=yi_v[:, None],
    z=zi_v[None, :],
    surfacecolor=np.log10(salinity_xslice),
    colorscale='RdYlBu_r',
    showscale=False,
    opacity=0.7,
    name=f'Salinity at x={xv:.0f} m',
    showlegend=True
)

# Vertical salinity slice at y = mean(y)
trace_salinity_yslice = go.Surface(
    x=xi_v[:, None],
    y=np.full_like(salinity_yslice, yv),
    z=zi_v[None, :],
    surfacecolor=np.log10(salinity_yslice),
    colorscale='RdYlBu_r',
    showscale=False,
    opacity=0.7,
    name=f'Salinity at y={yv:.0f} m',
    showlegend=True
)

# --- Combine and Plot ---

data = [
    trace_dem,
    trace_groundwater,
    trace_rock1,
    trace_rock2,
    trace_salinity_hslice,
    trace_salinity_xslice,
    trace_salinity_yslice
]

fig = go.Figure(data=data)

fig.update_layout(
    margin=dict(l=20, r=50, b=20, t=20),
    scene=dict(
        xaxis=dict(title='Easting (m)', color='black', showbackground=True, backgroundcolor='gray'),
        yaxis=dict(title='Northing (m)', color='black', showbackground=True, backgroundcolor='gray'),
        zaxis=dict(title='Elevation (m)', color='black', showbackground=True, backgroundcolor='gray'),
        aspectratio=dict(x=1, y=1, z=0.25),
        camera=dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=-0.2),
            eye=dict(x=-1., y=-1.3, z=1.)
        )
    ),
    legend=dict(
        x=0, y=0.8,
        font=dict(size=13, color='black'),
        bgcolor='rgb(230,230,230)',
        bordercolor='black',
        borderwidth=2,
        title='<b> Explanation </b><br> (click each to toggle) <br>'
    ),
)

plotly.offline.plot(fig, filename='3d-salinity-rocklayers.html')