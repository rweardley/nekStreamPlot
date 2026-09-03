
- Step 0:
  ```
  pip install extremeflow-pysemtools[all]
  ```

- Step 1: interpolate data onto uniform grid via pysemtools      
  Warning, it's unreasonably slow... 

  My demo file has (x,y,z) in [0, 4] x [-1.1, 1.1] x [-1, 1]   
  And, I want to extract at x = 2.0

  ```
  mpiexec -n 8 python interp.py channel0.f00000 points.csv \
    --bounds 2 2 -1 1 -1 1 \
    --shape 1 1001 1001 \
    --fields u,v,w,p
  ```

  You may visualize `points.csv` in paraview.      
  Nothing fancy, just table to point is sufficient    
   
- Step 2: read 2D csv, plot streamline    
  ```
  python plot_streamfunc.py points.csv --shape 1 1001 1001     
  ```
  It solves stream function by laplacian, so don't use too much points.
