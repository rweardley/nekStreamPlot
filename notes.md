Extract csv::

```
mpiexec -n 8 python interp.py channel0.f00000 points_solid.csv \
  --bounds 0 0 -1.0869565010070801 1.0869565010070801 -1.0869565010070801 1.0869565010070801 \
  --shape 1 1001 1001 \
  --fields u,v,w,p,t
```

Rescale:

```
python rescale_csv.py points_solid.csv \
  --scale x 0.023 0 \
  --scale y 0.023 0 \
  --scale z 0.023 0 \
  --scale t 45.098039 300 \
  --scale u 0.03 0 \
  --scale w 0.03 0 \
  --scale v 0.03 0
```

Plot:

```
python plot_streamfunc.py points_solid_rescaled.csv \
  --shape 1 1001 1001 \
  --plot-streamfunction 0 \
  --background-colormap t \
  --background-label "Temperature (°C)" \
  --velocity-label "Velocity (m/s)"
```

```
python plot_streamfunc.py points_solid_rescaled.csv \
  --shape 1 1001 1001 \
  --plot-streamfunction 0 \
  --velocity-label "Velocity (m/s)" \
  --velocity-clim 0 0.000438 \
  --background-colormap t \
  --background-label "Temperature (°C)" \
  --background-clim 300 323.75 \
  --integration-length 100 \
  --domain-bounds -0.023 0.023 -0.023 0.023
```

<!-- 
```
python plot_streamfunc.py points_solid.csv \
  --shape 1 1001 1001 \
  --plot-streamfunction 0 \
  --velocity-label "Velocity (ND)" \
  --background-colormap t \
  --background-label "Temperature (ND)" \
  --domain-bounds -1 1 -1 1
``` -->