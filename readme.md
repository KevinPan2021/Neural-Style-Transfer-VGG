Introduction:
	This project aims to preform Neural Style Transfer using VGG19 pretrained model. The project also provides a GUI that lets you upload your own image.


Build: 
	M1 Macbook Pro
	Miniforge 3 (Python 3.9)
	PyTorch version: 2.2.1

* Alternative Build:
	Windows (NIVIDA GPU)
	Anaconda 3
	PyTorch



Generate ".py" file from ".ui" file:
	1) open Terminal. Navigate to directory
	2) Type "pyuic5 -x qt_main.ui -o qt_main.py"



Core Project Structure:
	GUI.py (Run to generate a GUI)
	main.py (Run to train model)
	model.py
	qt_main.py
	visualization.py
	../pytorch_model_weights (download and place model .pth here)

Credit:
	The algorithm is referenced from "https://pytorch.org/tutorials/advanced/neural_style_tutorial.html"
	