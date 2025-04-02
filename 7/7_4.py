import bpy
import json
import math

filepath = "/Users/dimapartov/Desktop/model_to_text_task_7_2.txt"

def load_model(filepath):
    with open(filepath, 'r') as file:
        mesh_data = json.load(file)
    vertices = mesh_data["vertices"]
    faces = mesh_data["faces"]
    mesh = bpy.data.meshes.new(mesh_data["object-name"] + "_mesh")
    obj = bpy.data.objects.new(mesh_data["object-name"] + "_obj", mesh)
    bpy.context.scene.collection.objects.link(obj)
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj.location = mesh_data["location"]
    obj.rotation_euler = mesh_data["rotation"]
    obj.scale = mesh_data["scale"]
    print(f"Модель успешно загружена из: {filepath}")
    return obj

loaded_obj = load_model(filepath)

radii = [6.0, 12.0, 18.0]
step_angle_deg = 30
scale_coef = 2.0
center = loaded_obj.location.copy()

for radius in radii:
    num_copies = int(360 / step_angle_deg)
    scale_factor = scale_coef / radius
    for i in range(num_copies):
        angle_rad = math.radians(i * step_angle_deg)
        new_x = center.x + radius * math.cos(angle_rad)
        new_y = center.y + radius * math.sin(angle_rad)
        new_z = center.z
        new_obj = loaded_obj.copy()
        new_obj.data = loaded_obj.data.copy()
        new_obj.location = (new_x, new_y, new_z)
        new_obj.scale = (
            loaded_obj.scale[0] * scale_factor,
            loaded_obj.scale[1] * scale_factor,
            loaded_obj.scale[2] * scale_factor
        )
        bpy.context.scene.collection.objects.link(new_obj)

print("Копии модели успешно размещены по 3 окружностям.")