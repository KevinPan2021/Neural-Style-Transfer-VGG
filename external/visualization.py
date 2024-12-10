import matplotlib.pyplot as plt


    
    
# plot the images
def plot_image(content, style, generated=None):
    plt.figure(figsize=(15, 15))
    
    if generated is None:
        # plotting img
        plt.subplot(1, 2, 1)
        plt.title("Content Image")
        plt.imshow(content.to('cpu').squeeze().numpy().transpose(1,2,0))
        plt.axis('off')
        
        # plotting label
        plt.subplot(1, 2, 2)
        plt.title("Style Mask")
        plt.imshow(style.to('cpu').squeeze().numpy().transpose(1,2,0))
        plt.axis('off')
        
    
    else:
        # plotting img
        plt.subplot(1, 3, 1)
        plt.title("Content Image")
        plt.imshow(content.to('cpu').squeeze().numpy().transpose(1,2,0))
        plt.axis('off')
        
        # plotting label
        plt.subplot(1, 3, 2)
        plt.title("Style Mask")
        plt.imshow(style.to('cpu').squeeze().numpy().transpose(1,2,0))
        plt.axis('off')
        
        # plotting prediction
        plt.subplot(1, 3, 3)
        plt.title("Generated Mask")
        plt.imshow(generated.to('cpu').squeeze().numpy().transpose(1,2,0))
        plt.axis('off')
    
    plt.show()