import pandas as pd
import numpy as np
from scipy.interpolate import griddata
import plotly.graph_objects as go
import plotly

# --- Load Data ---

df = pd.read_csv("model-grid-subsample.csv")
df = df[df.dem_m > df.zkm * 1e3]

x = df['xkm'].to_numpy() * 1e3
y = df['ykm'].to_numpy() * 1e3
z = df['zkm'].to_numpy() * 1e3
u = df['mean_tds'].to_numpy()

rock1 = pd.read_csv("rock-layer-1.csv")
rock2 = pd.read_csv("rock-layer-2.csv")

# DEM surface
xi_dem = np.linspace(x.min(), x.max(), 100)
yi_dem = np.linspace(y.min(), y.max(), 100)
zi_dem = griddata(
    (x, y), df.dem_m.to_numpy(),
    (xi_dem[None, :], yi_dem[:, None]),
    method='linear'
)

# Rock layers
xi_rock1 = np.linspace(rock1['xkm'].min()*1e3, rock1['xkm'].max()*1e3, 200)
yi_rock1 = np.linspace(rock1['ykm'].min()*1e3, rock1['ykm'].max()*1e3, 200)
zi_rock1 = griddata(
    (rock1['xkm']*1e3, rock1['ykm']*1e3),
    rock1['mean_pred'],
    (xi_rock1[None, :], yi_rock1[:, None]),
    method='linear'
)

xi_rock2 = np.linspace(rock2['xkm'].min()*1e3, rock2['xkm'].max()*1e3, 200)
yi_rock2 = np.linspace(rock2['ykm'].min()*1e3, rock2['ykm'].max()*1e3, 200)
zi_rock2 = griddata(
    (rock2['xkm']*1e3, rock2['ykm']*1e3),
    rock2['mean_pred'],
    (xi_rock2[None, :], yi_rock2[:, None]),
    method='linear'
)

# --- Plotly Traces (Static) ---

trace_dem = go.Surface(
    x=xi_dem, y=yi_dem, z=zi_dem,
    colorscale='Earth',
    name='Land surface',
    showscale=False,
    showlegend=True,
    opacity=1.0
)

trace_rock1 = go.Surface(
    x=xi_rock1, y=yi_rock1, z=zi_rock1,
    surfacecolor=np.zeros_like(zi_rock1),
    colorscale=[[0, 'rgba(255,0,0,1)'], [1, 'rgba(255,0,0,1)']],  # red
    name='Rock Layer 1',
    showscale=False,
    showlegend=True,
    opacity=0.4
)

trace_rock2 = go.Surface(
    x=xi_rock2, y=yi_rock2, z=zi_rock2,
    surfacecolor=np.zeros_like(zi_rock2),
    colorscale=[[0, 'rgba(0,255,0,1)'], [1, 'rgba(0,255,0,1)']],  # green
    name='Rock Layer 2',
    showscale=False,
    showlegend=True,
    opacity=0.4
)

trace_groundwater = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    name='Groundwater salinity',
    showlegend=True,
    marker=dict(
        size=3, symbol='square',
        colorscale='RdYlBu_r', color=np.log10(u),
        showscale=False
    ),
    hovertemplate=
        'Easting: %{x:.0f} m<br>' +
        'Northing: %{y:.0f} m<br>' +
        'Elevation: %{z:.0f} m<br>' +
        'TDS: %{customdata[0]:.0f} mg/L<br>' +
        'log10(TDS): %{marker.color:.2f}<extra></extra>',
    customdata=np.stack([u], axis=-1)
)

# --- Animation Frames ---

xi = np.linspace(x.min(), x.max(), 100)
yi = np.linspace(y.min(), y.max(), 100)
z_slices = np.linspace(z.min(), z.max(), 25)

# Initial slice
z_slice0 = z_slices[0]
salinity_slice0 = griddata(
    (x, y, z), u,
    (xi[None, :, None], yi[:, None, None], np.full((100, 100, 1), z_slice0)),
    method='linear'
).squeeze()

trace_slice = go.Surface(
    x=xi,
    y=yi,
    z=np.full_like(salinity_slice0, z_slice0),
    surfacecolor=np.log10(salinity_slice0),
    cmin=np.log10(400),
    cmax=np.log10(10000),
    colorscale='RdYlBu_r',
    opacity=0.7,  
    showscale=True,
    colorbar=dict(
        title=dict(text='Salinity (mg/L)', side='right'),
        x=1.02, len=0.5, ticks='outside',
        tickvals=np.log10([400, 1000, 5000, 10000]),
        ticktext=[400, 1000, 5000, 10000]
    ),
    name='Salinity slice',
    showlegend=True  # fontos!
)

frames = []
for i, z_slice in enumerate(z_slices):
    salinity_slice = griddata(
        (x, y, z), u,
        (xi[None, :, None], yi[:, None, None], np.full((100, 100, 1), z_slice)),
        method='linear'
    ).squeeze()

    # A slice trace minden frame-ben ugyanazt a nevet kapja, showlegend csak az első frame-ben True
    frame = go.Frame(
        name=f'{z_slice:.1f}',
        data=[
            trace_dem,
            trace_groundwater,
            trace_rock1,
            trace_rock2,
            go.Surface(
                x=xi,
                y=yi,
                z=np.full_like(salinity_slice, z_slice),
                surfacecolor=np.log10(salinity_slice),
                cmin=np.log10(400),
                cmax=np.log10(10000),
                colorscale='RdYlBu_r',
                opacity=0.3,
                showscale=True,
                colorbar=dict(
                    title=dict(text='Salinity (mg/L)', side='right'),
                    x=1.02, len=0.5, ticks='outside',
                    tickvals=np.log10([400, 1000, 5000, 10000]),
                    ticktext=[400, 1000, 5000, 10000]
                ),
                name='Salinity slice',
                showlegend=(i == 0)  # csak az első frame-ben jelenjen meg a legendában
            )
        ],
        layout=go.Layout(
            annotations=[
                dict(
                    text=f"<b>Z = {z_slice:.1f} m</b>",
                    x=0.5, y=1.08, xref='paper', yref='paper',
                    showarrow=False,
                    font=dict(size=24, color='black'),
                    align='center',
                    bgcolor='rgba(255,255,255,0.7)',
                    bordercolor='black',
                    borderwidth=1
                )
            ]
        )
    )
    frames.append(frame)

# --- Initial Slice ---

initial_slice = [
    trace_dem,
    trace_groundwater,
    trace_rock1,
    trace_rock2,
    trace_slice
]

# --- Create Figure ---

fig = go.Figure(
    data=initial_slice,
    frames=frames
)

fig.update_layout(
    title='<b>Groundwater salinity</b>',
    title_x=0.0,  # bal oldalon
    title_y=0.95,
    title_font=dict(size=44, color='black'),
    margin=dict(l=20, r=50, b=20, t=100),
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
    updatemenus=[dict(
        type='buttons',
        showactive=False,
        x=0.05, y=0,
        buttons=[
            dict(label='▶ Play', method='animate', args=[None, {
                'frame': {'duration': 500, 'redraw': True},
                'fromcurrent': True, 'transition': {'duration': 0}
            }]),
            dict(label='⏸ Pause', method='animate', args=[[None], {
                'mode': 'immediate',
                'frame': {'duration': 0, 'redraw': False},
                'transition': {'duration': 0}
            }])
        ]
    )],
    sliders=[dict(
        steps=[dict(method='animate', args=[[f'{z:.1f}'], {
            'mode': 'immediate',
            'frame': {'duration': 0, 'redraw': True},
            'transition': {'duration': 0}
        }], label=f'{z:.1f} m') for z in z_slices],
        x=0.1, y=0,
        len=0.8,
        currentvalue=dict(prefix='Z-slice: ', font=dict(size=14)),
        pad=dict(b=10)
    )]
)

# --- Save and Show ---

plotly.offline.plot(fig, filename='3d-salinity-animated.html')