application_name = 'Neural Style Transfer'
# pyqt packages
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog

import sys
import numpy as np
from torchvision import transforms
import torch
import torch.optim as optim
from PIL import Image
import re
import os

from model import VGG19
from qt_main import Ui_Application
from main import GPU_Device, get_style_model_and_losses, load_image


def is_positive_integer(input_str):
    numeric_pattern = r'^\d+$'
    return bool(re.match(numeric_pattern, input_str))


def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()
        


class StyleTransferWorker(QThread):
    iterationUpdated = pyqtSignal(int)
    finished = pyqtSignal()
    
    def __init__(self, input_image, content_image, style_image, model, 
                 iterations, style_weight, content_weight, style_losses, content_losses):
        super().__init__()
        self.input_image = input_image
        self.content_image = content_image
        self.style_image = style_image
        self.model = model
        self.iterations = iterations
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.style_losses = style_losses
        self.content_losses = content_losses
        

    def run(self):        
        optimizer = optim.LBFGS([self.input_image])

        iteration = [0] # pass by reference
        while iteration[0] <= self.iterations:
            
            def closure():
                # clip the image to [0,1]
                with torch.no_grad():
                    self.input_image.clamp_(0, 1)
            
                optimizer.zero_grad()
                self.model(self.input_image)
                style_score = 0
                content_score = 0
            
                for sl in self.style_losses:
                    style_score += sl.loss
                for cl in self.content_losses:
                    content_score += cl.loss
            
                style_score *= self.style_weight
                content_score *= self.content_weight
            
                loss = style_score + content_score
                loss.backward()
                
                iteration[0] += 1
                    
                # Update progress bar
                self.iterationUpdated.emit(iteration[0])
                
                return style_score + content_score

            optimizer.step(closure)

        self.finished.emit()
        


class QT_Action(Ui_Application, QMainWindow):
    
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        self.setupUi(self)
        self.retranslateUi(self)
        self.setWindowIcon(QIcon('favicon.png')) # changed the window icon
        self.setWindowTitle(application_name) # set the title
        self.progressBar.setValue(0)
        
        # runtime variable
        self.model = None
        self.content_image = None
        self.style_image = None
        self.style_ori_size = None # keep this for displaying
        self.output = None
        self.iterations = 300
        self.style_weight = 1000000
        self.content_weight = 1
        
        # load the model
        self.load_model_action()
        
        
            
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.toolButton_import_content.clicked.connect(self.import_content_action)
        self.toolButton_import_style.clicked.connect(self.import_style_action)
        
        self.comboBox_model.activated.connect(self.load_model_action)
        self.lineEdit_content_weight.editingFinished.connect(self.set_content_weight_action)
        self.lineEdit_style_weight.editingFinished.connect(self.set_style_weight_action)
        self.lineEdit_iterations.editingFinished.connect(self.set_iterations_action)
        
        self.toolButton_process.clicked.connect(self.process_action)
        self.toolButton_export.clicked.connect(self.export_action)
        
    
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'VGG19':
            # load the model architechture
            self.model = VGG19().get_feature_layers()
            
        # move model to GPU
        self.model = self.model.to(GPU_Device())
        
        self.model.eval() # Set model to evaluation mode
        
        
        
    
    # clicking the import content button action
    def import_content_action(self,):
        # show an "Open" dialog box and return the path to the selected file
        filename, _ = QFileDialog.getOpenFileName(None, "Select file", options=QFileDialog.Options())
        self.lineEdit_import_content.setText(filename)
        
        # didn't select any files
        if filename is None or filename == '': 
            return
    
        # selected .jpg
        if filename.endswith('.jpg') or filename.endswith('.png'):
            self.content_image, _ = load_image(filename)
            self.lineEdit_import_content.setText(filename)
            
            
            # clear the style image
            self.style_image = None
            self.output = None
            self.lineEdit_import_style.setText('')
            
            # refresh the display
            self.update_display()
            
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .jpg')
            self.import_content_action()
       
            
            
    # clicking the import content button action
    def import_style_action(self,):
        if self.content_image is None:
            show_message(self, title='Load Error', message='load the content image first')
            return
        
        # show an "Open" dialog box and return the path to the selected file
        filename, _ = QFileDialog.getOpenFileName(None, "Select file", options=QFileDialog.Options())
        self.lineEdit_import_style.setText(filename)
        
        # didn't select any files
        if filename is None or filename == '': 
            return
    
        # selected .jpg
        if filename.endswith('.jpg') or filename.endswith('.png'):
            self.style_image, self.style_ori_size = load_image(filename, shape=self.content_image.shape[-2:])
            self.lineEdit_import_style.setText(filename)
            
            # clear output
            self.output = None
            
            self.update_display()
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .jpg')
            self.import_style_action()
               
            
    # pad the image to square
    def pad_to_square(self, image):
        # Get the height and width of the input image
        height, width = image.shape[:2]
    
        # Calculate the size of the square
        square_len = max(height, width)
    
        # Create a new square image filled with zeros
        padded_image = np.zeros((square_len, square_len, 3), dtype=np.uint8)
    
        # Calculate the starting position to append the original image
        start_h = (square_len - height) // 2
        start_w = (square_len - width) // 2
    
        # Append the original image to the center of the square image
        padded_image[start_h:start_h+height, start_w:start_w+width] = image
    
        return padded_image



    
    def update_display(self):
        if self.content_image is None:
            self.label_content_img.clear()
        else:
            image = self.content_image.to('cpu').squeeze().numpy() * 255
            image = image.transpose(1,2,0).astype(np.uint8)
            image = self.pad_to_square(image)
            h, w, ch = image.shape
            q_image = QImage(image.data.tobytes(), w, h, ch*w, QImage.Format_RGB888)  # Create QImage
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_content_img.setPixmap(qpixmap)
            
        if self.style_image is None:
            self.label_style_img.clear()
        else:
            image = self.style_image.clone()
            transform = transforms.Resize((self.style_ori_size[1], self.style_ori_size[0]))
            image = transform(image)
            image = image.to('cpu').squeeze().numpy() * 255
            image = image.transpose(1,2,0).astype(np.uint8)
            # reshape the style image back to original size for displaying
            
            image = self.pad_to_square(image)
            h, w, ch = image.shape
            q_image = QImage(image.data.tobytes(), w, h, ch*w, QImage.Format_RGB888)  # Create QImage
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_style_img.setPixmap(qpixmap)
                
        if self.output is None:
            self.label_generated_img.clear()
        else:
            image = self.output.clone().detach().clip(0, 1).to('cpu').squeeze().numpy() * 255
            image = image.transpose(1,2,0).astype(np.uint8)
            image = self.pad_to_square(image)
            h, w, ch = image.shape
            q_image = QImage(image.data.tobytes(), w, h, ch*w, QImage.Format_RGB888)  # Create QImage
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_generated_img.setPixmap(qpixmap)



    def set_content_weight_action(self):
        text = self.lineEdit_content_weight.text().strip()
        if is_positive_integer(text):
            self.content_weight = int(text)
        else:
            self.lineEdit_content_weight.setText('1')
            self.content_weight = 1
    
    
    def set_style_weight_action(self):
        text = self.lineEdit_style_weight.text().strip()
        if is_positive_integer(text):
            self.style_weight = int(text)
        else:
            self.lineEdit_style_weight.setText('1000000')
            self.style_weight = 1000000
    
    
    def set_iterations_action(self):
        text = self.lineEdit_iterations.text().strip()
        if is_positive_integer(text):
            self.iterations = int(text)
        else:
            self.lineEdit_iterations.setText('300')
            self.iterations = 300
        
    
    def update_progress_bar(self, iteration):
        self.progressBar.setValue(iteration)
    
    
    def process_action(self):
        if None in [self.content_image, self.style_image]:
            show_message(self, title='Process Error', message='Please load images first')
            return
        
        # set the progressbar
        self.progressBar.setMaximum(self.iterations)
        
        # convert to gpu
        content_img = self.content_image.to(GPU_Device())
        style_img = self.style_image.to(GPU_Device())
        
        # target
        self.output = content_img.clone()
        
        # build the cnn model
        model, style_losses, content_losses = get_style_model_and_losses(self.model, style_img, content_img)
                               
                               
        # We want to optimize the input and not the model parameters so we
        # update all the requires_grad fields accordingly
        self.output.requires_grad_(True)
        # We also put the model in evaluation mode, so that specific layers
        # such as dropout or batch normalization layers behave correctly.
        model.eval()
        model.requires_grad_(False)

        
        # Connect signal from worker to slot in main thread to update progress bar
        self.worker = StyleTransferWorker(self.output, content_img, style_img, model, self.iterations, 
                                          self.style_weight, self.content_weight, style_losses, content_losses)
        self.worker.iterationUpdated.connect(self.update_progress_bar)
        self.worker.finished.connect(self.update_display)
        
        # Start the worker thread
        self.worker.start()
        
        
    def export_action(self):
        # Check if output numpy array exists
        if self.output is None:
            show_message(self, title='Export Error', message='Please process first')
            return
    
        # Open a file dialog to select the export folder
        export_folder = QFileDialog.getExistingDirectory(self, "Select Export Folder", os.path.expanduser("~"))
    
        if export_folder:
            image = self.output.clone().detach().clip(0, 1).to('cpu').squeeze().numpy() * 255
            image = image.transpose(1,2,0).astype(np.uint8)
            
            # Create a PIL Image from the output numpy array
            image = Image.fromarray(image)
    
            # Save the image as .jpg file in the selected folder
            file_path = os.path.join(export_folder, "output_image.jpg")
            image.save(file_path, "JPEG")
    
        
        
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()