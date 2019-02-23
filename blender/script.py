#####################################
# Will it hang? dataset
#####################################
# 
# Tested in Blender 2.79, 2.76b
#
# Author: David-Estevez
#
#####################################

import bpy
from mathutils import Matrix
import os
import argparse
from random import uniform, choice

debug = True

def compute_com(mesh):
    """
    Computes the center of mass of a mesh assuming homogeneous vertex density
    """
    vertices = list(map(lambda v: v.co, mesh.vertices))
    result = vertices[0]
    for vertex in vertices[1:]:  # For some reason, the sum() function doesn't seem to work
        result+=vertex
    return result/float(len(vertices))

def delete_empties():
    for name, empty in filter(lambda x: 'Empty' in x[0], bpy.data.objects.items()):
        empty.select = True
    bpy.ops.object.delete()

def delete_vertex_groups(target):
    bpy.data.scenes['Scene'].objects.active = target
    bpy.ops.object.vertex_group_remove(all=True)

def create_vertex_groups(target):
    bpy.data.scenes['Scene'].objects.active = target
    bpy.data.scenes['Scene'].tool_settings.vertex_group_weight = 1
    bpy.ops.object.mode_set(mode = 'EDIT')
    bpy.ops.mesh.select_mode(type='VERT')
    bpy.ops.mesh.select_all(action='DESELECT')
    for i in range(len(target.data.vertices)):  # Not pythonic, but iterator seems not to work for this one
        bpy.ops.object.mode_set(mode='OBJECT')
        target.data.vertices[i].select = True
        bpy.ops.object.mode_set(mode='EDIT')
        bpy.ops.object.vertex_group_add()
        bpy.ops.object.vertex_group_assign()
        bpy.ops.mesh.select_all(action='DESELECT')
    bpy.ops.object.mode_set(mode='OBJECT')
        
def set_random_pinning_point(target):
    group = choice(target.vertex_groups)
    target.modifiers["VertexWeightMix"].vertex_group_a = group.name
    target.modifiers["Cloth"].settings.vertex_group_mass = group.name
    
def get_pinning_point(target):
    group = target.modifiers["Cloth"].settings.vertex_group_mass 
    index = target.vertex_groups[group].index
    for i, v in enumerate(target.data.vertices):
        for g in v.groups:
            if g.group == index:
                return i
    return -1

def center_in_vertex(vertex_ind, target):
    v = target.data.vertices[vertex_ind]
    v_in_world_frame = target.matrix_world*v.co
    t = Matrix.Translation(v_in_world_frame)
    cloth.bound_box.data.location += t.inverted().to_translation()

# Get script args
try:
    parser = argparse.ArgumentParser()

    _, all_arguments = parser.parse_known_args()
    double_dash_index = all_arguments.index('--')
    script_args = all_arguments[double_dash_index+1:]
except ValueError:
    print("Some error happened. Using default values")
    start = 0
    number = 2
    output_path = './'
else:
    parser.add_argument('-s', '--start', help='starting index')
    parser.add_argument('-n', '--number', help='number of trials')
    parser.add_argument('-o', '--outdir', help='output directory')
    parsed_script_args, _ = parser.parse_known_args(script_args)

    start = int(parsed_script_args.start)
    number = int(parsed_script_args.number)
    output_path = parsed_script_args.outdir

cloth = bpy.data.objects['Cloth']

# Region in which the random garment will be generated (units: m)
x_min, x_max = -0.01, 0.01
y_min, y_max = -0.40, 0.40
z_min, z_max = 1.5, 1.7

hanging_frame = 40
final_frame = 120
bpy.data.scenes['Scene'].frame_current = 0
folder = os.path.abspath(output_path)
bpy.data.scenes['Scene'].node_tree.nodes["File Output png"].base_path = folder
bpy.data.scenes['Scene'].node_tree.nodes["File Output exr"].base_path = folder

### Script starts to do stuff here ###

for i in range(start, start+number):
    # 1. Select (randomly) a vertex for pinning
    set_random_pinning_point(cloth)

    # 1. Randomly place the cloth (within a [x,y,z] region)
    center_in_vertex(get_pinning_point(cloth), cloth)
    bpy.data.objects['Cloth'].bound_box.data.location[0] += uniform(x_max, x_min)
    bpy.data.objects['Cloth'].bound_box.data.location[1] += uniform(y_max, y_min)
    bpy.data.objects['Cloth'].bound_box.data.location[2] += uniform(z_max, z_min)


    # 3. Simulate dynamics
    bpy.ops.ptcache.bake_all(bake=True)

    # 4. At frame 'hanging_frame', grab a depth frame and save it
    bpy.data.scenes['Scene'].frame_current = hanging_frame
    # bpy.data.scenes['Scene'].render.filepath = 'out-40.png'
    bpy.data.scenes['Scene'].node_tree.nodes["File Output png"].file_slots[0].path = "img-{}.png".format(i)
    bpy.data.scenes['Scene'].node_tree.nodes["File Output exr"].file_slots[0].path = "img-{}.exr".format(i)
    bpy.ops.render.render(write_still=True)

    # 5. For frame 'hanging_frame' to frame 'final_frame', record x,y,z position of center of bounding box
    com_trajectory = []
    for frame in range(hanging_frame, final_frame+1):
        bpy.data.scenes['Scene'].frame_current = frame
        cloth = bpy.data.objects['Cloth']
        deformed_mesh = cloth.to_mesh(bpy.data.scenes['Scene'], apply_modifiers=True, settings='PREVIEW')
        com_in_cloth_frame = compute_com(deformed_mesh)
        com_in_world_frame = cloth.matrix_world*com_in_cloth_frame
        com_trajectory.append(com_in_world_frame)
        
    with open(os.path.join(folder, "img-{}.csv".format(i)), 'w') as f:
        for point in com_trajectory:
            f.write("{} {} {}\n".format(point[0], point[1], point[2]))
            
    # Create empties as a visual aid:
    if debug:
        for point in com_trajectory:
            bpy.ops.object.add(type='EMPTY', view_align=False, enter_editmode=False, location=point)
        for name, empty in filter(lambda x: 'Empty' in x[0], bpy.data.objects.items()):
            empty.scale = (0.05, 0.05, 0.05)
        
    # 6. Release physics cache
    bpy.ops.ptcache.free_bake_all()