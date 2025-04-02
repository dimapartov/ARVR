import bpy
import random
import math

primitive_functions = [
    bpy.ops.mesh.primitive_cube_add,
    bpy.ops.mesh.primitive_uv_sphere_add,
    bpy.ops.mesh.primitive_cone_add,
    bpy.ops.mesh.primitive_cylinder_add,
    bpy.ops.mesh.primitive_torus_add
]

created_objects = []

for i in range(10):
    prim_func = random.choice(primitive_functions)
    loc = (random.uniform(-10, 10), random.uniform(-10, 10), random.uniform(-10, 10))
    prim_func(location=loc)
    obj = bpy.context.active_object
    created_objects.append(obj)
    initial_scale = random.uniform(0.5, 2)
    obj.scale = (initial_scale, initial_scale, initial_scale)

for obj in created_objects:
    obj.location = (100, 100, 0)
    rot_x = math.radians(random.uniform(0, 360))
    rot_y = math.radians(random.uniform(0, 360))
    rot_z = math.radians(random.uniform(0, 360))
    obj.rotation_euler = (rot_x, rot_y, rot_z)
    new_scale = random.uniform(0.5, 10)
    obj.scale = (new_scale, new_scale, new_scale)