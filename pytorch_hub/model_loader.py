import torch

output_file = "torch_hub_models_log.txt"

# To obtain a listing of models
models_list =torch.hub.list("materialsvirtuallab/matgl", force_reload=True)

with open(output_file, "w") as file:
    file.write("Available Models from 'materialsvirtuallab/matgl':\n")
    for model_name in models_list:
        file.write(f"{model_name}\n")

print(f"Model list has been saved to '{output_file}'.")

# To load a model
model = torch.hub.load("materialsvirtuallab/matgl", 'm3gnet_universal_potential')

with open(output_file, "a") as file:
    file.write("\nLoaded Model:\n")
    file.write(f"Model Name: m3gnet_universal_potential\n")
    file.write("Model loaded successfully.\n")

print("Model loading log has been appended to the file.")