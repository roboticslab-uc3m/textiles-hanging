#####################################
# Will it hang? dataset
#####################################
# 
# Tested in Blender 2.79
#
# Author: David-Estevez
#
#####################################

import bpy
from random import uniform

# Region in which the random garment will be generated (units: m)
x_min, x_max = -0.30, 0.30
y_min, y_max = -0.70, 0.70
z_min, z_max = 1.3, 1.5

hanging_frame = 40
final_frame = 120
bpy.data.scenes['Scene'].frame_current = 0

### Script starts to do stuff here ###

# 1. Randomly place the cloth (within a [x,y,z] region)
bpy.data.objects['Cloth'].bound_box.data.location[0] = uniform(x_max, x_min)
bpy.data.objects['Cloth'].bound_box.data.location[1] = uniform(y_max, y_min)
bpy.data.objects['Cloth'].bound_box.data.location[2] = uniform(z_max, z_min)

# 2. Select (randomly) a vertex for pinning


# 3. Simulate dynamics
bpy.ops.ptcache.bake_all(bake=True)

# 4. At frame 40, grab a depth frame and save it
bpy.data.scenes['Scene'].frame_current = hanging_frame
bpy.data.scenes['Scene'].render.filepath = 'out-40.png'
bpy.ops.render.render(write_still=True)

# 5. For frame 40 to frame 120, record x,y,z position of center of bounding box
com_trajectory = []
for frame in range(hanging_frame, final_frame+1):
    bpy.data.scenes['Scene'].frame_current = frame
    bpy.ops.object.origin_set(type='ORIGIN_CENTER_OF_MASS')
    com_trajectory.append(bpy.data.objects['Cloth'].location)
    
with open('out-traj.csv', 'w') as f:
    for point in com_trajectory:
        f.write("{} {} {}\n".format(point[0], point[1], point[2]))

# 6. Release physics cache
bpy.ops.ptcache.free_bake_all()