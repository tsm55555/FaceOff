from FaceOff.AFR import load_data, Attack
from PIL import Image

# Load the data.  This will detect and resize the faces
inputs = load_data('./faces/input/')
targets = load_data('./faces/target/')

# Initialize the Attack object with 
adversarial = Attack(inputs[0], targets[3], optimizer='adam')

# Perform attack
adversarial_tensor, mask_tensor, img = adversarial.train(epochs=100, detect=True, verbose=True)
mask_img = adversarial.view(mask_tensor)
# print(adversarial_tensor)
# print(mask_tensor)

# Show the image with mask applied
img.show()
img.save("results.jpg")
mask_img.save("mask.jpg")
inputs[0].save("input.jpg")
targets[3].save("target.jpg")