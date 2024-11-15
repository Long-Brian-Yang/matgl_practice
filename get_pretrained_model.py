import matgl
available_models = matgl.get_available_pretrained_models()

print(matgl.get_available_pretrained_models())

with open("available_models.txt", "w") as file:
    file.write("Available Pretrained Models:\n")
    for model in available_models:
        file.write(f"{model}\n")

print("The list of available models has been saved to 'available_models.txt'.")