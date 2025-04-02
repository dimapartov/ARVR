import bpy
import json

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


load_model(filepath)