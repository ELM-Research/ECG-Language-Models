import random
import numpy as np
from PIL import Image

class RGB:
    ### SIGNAL TO IMAGE FUNCTIONS ###
    def signal_to_image(self, ecg_signal: np.array):
        image = self.viz.get_plot_as_image(ecg_signal, 250, lead_names = self.lead_names)  # 250 Hz
        if self.args.augment_rgb and random.random() < 0.6:
            return self.augment_image(image)
        return Image.fromarray(image)

    ### IMAGE AUGMENTATION FUNCTIONS ###
    def augment_image(self, image: np.array):
        image = Image.fromarray(image)
        return self.aug(image)