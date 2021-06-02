# PointCloudRegistration
Registration of human MRT point clouds. Was tested on self-maded face dataset.<br>
## Samples <br>
![image](https://user-images.githubusercontent.com/29431011/120477759-cfa29380-c3b4-11eb-9520-51ea82385690.png)
![image](https://user-images.githubusercontent.com/29431011/120477789-d6c9a180-c3b4-11eb-9d6f-dcd2b41b2075.png)
![image](https://user-images.githubusercontent.com/29431011/120477821-dd581900-c3b4-11eb-829a-09a051bae0d8.png)
## Usage<br>
`main.py [-h] [--voxel VOXEL] [--silent] [--no_preview] source target`<br>

Process global registration of point clouds and saves result transformation matrix to "finalreg.txt". Positional arguments:<br>
  `source` - source point cloud<br>
  `target` - target point cloud<br>

optional arguments:<br>
 ` --voxel VOXEL, -v VOXEL` - voxel size for image processing. If result is not good, you can try to increase it. Default value is 5.<br>
 ` --silent, -s` - if is set, you will not see debugging message.<br>
 ` --no_preview, -np` - if is set, you will no preview registration result.<br>
