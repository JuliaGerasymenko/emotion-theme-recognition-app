"""Utilities"""
import re
import base64

from PIL import Image
from io import BytesIO

class Base64Utility:
    def __init__(self):
        pass
    def base64_to_pil(self, img_base64):
        """
        Convert base64 image data to PIL image
        """
        image_data = re.sub('^data:image/.+;base64,', '', img_base64)
        pil_image = Image.open(BytesIO(base64.b64decode(image_data)))
        return pil_image

    def np_to_base64(self, img_np):
        """
        Convert numpy image (RGB) to base64 string
        """
        img = Image.fromarray(img_np.astype('uint8'), 'L')
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return u"data:image/png;base64," + base64.b64encode(buffered.getvalue()).decode("ascii")
