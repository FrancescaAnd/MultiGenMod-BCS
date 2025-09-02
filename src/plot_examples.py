import matplotlib.pyplot as plt
from PIL import Image
import os

def plot_example_pair(dataset_name, example_folder="src/data_examples/"):
    """
    Plot CC+MLO pair for a given dataset.
    """
    cc_file = os.path.join(example_folder, f"{dataset_name}-CC.png")
    mlo_file = os.path.join(example_folder, f"{dataset_name}-MLO.png")
    
    if not os.path.exists(cc_file) or not os.path.exists(mlo_file):
        print(f"Example images for {dataset_name} not found in {example_folder}")
        return
    
    img_cc = Image.open(cc_file)
    img_mlo = Image.open(mlo_file)
    
    plt.figure(figsize=(8,4))
    plt.subplot(1,2,1)
    plt.imshow(img_cc, cmap='gray')
    plt.title(f"{dataset_name} CC")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(img_mlo, cmap='gray')
    plt.title(f"{dataset_name} MLO")
    plt.axis('off')
    
    plt.show()

plot_example_pair("INBREAST")
plot_example_pair("CBIS")
