import bpy
import json

filepath = "/Users/dimapartov/Desktop/model_to_text_task_7_2.txt"


def save_model(filepath):

    obj = bpy.context.active_object
    if obj.type == 'MESH':
        mesh = {
            "object-name": obj.name,
            "vertices": [tuple(v.co) for v in obj.data.vertices],
            "faces": [list(face.vertices) for face in obj.data.polygons],
            "location": tuple(obj.location),
            "rotation": tuple(obj.rotation_euler),
            "scale": tuple(obj.scale),
        }

    with open(filepath, 'w') as file:
        json.dump(mesh, file, indent=2)

    print(f"Модель успешно сохранена в: {filepath}")


save_model(filepath)