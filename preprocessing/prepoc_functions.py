def CLAHEContrastNorm(self, img, tile_size=(1,1)):
        """
        Function to apply CLAHE normalization to the image.
        Args:
          img: numpy image array
          tile_size:  (Default value = (1,1)) The square dimentions of the tiles that the image will be broken up into. The CLAHE will be applied to each tile. 
        Returns: 
          The image with CLAHE applied.
        """
        clahe = cv2.createCLAHE(tileGridSize=tile_size)
        return clahe.apply(img)
    
def get_square_crop(self, img, base_size=256, crop_size=256):
        """
        Function to crop the image from the center outward or add a border to get to the desired spot. 
        Args:
          img: numpy image array
          base_size:  (Default value = 256) The size to make the image if a border is needed to get to the desired size. 
          crop_size:  (Default value = 256) The desired size to crop the image
        Returns:
          Numpy image array of the desired size. 
        """
        res = img
#         print (res)
#         print (res.shape)
        height, width = res.shape

        if height < base_size:
            diff = base_size - height
            extend_top = diff // 2
            extend_bottom = diff - extend_top
            res = cv2.copyMakeBorder(res, extend_top, extend_bottom, 0, 0, 
                                     borderType=cv2.BORDER_CONSTANT, value=0)
            height = base_size

        if width < base_size:
            diff = base_size - width
            extend_top = diff // 2
            extend_bottom = diff - extend_top
            res = cv2.copyMakeBorder(res, 0, 0, extend_top, extend_bottom, 
                                     borderType=cv2.BORDER_CONSTANT, value=0)
            width = base_size

        crop_y_start = (height - crop_size) // 2
        crop_x_start = (width - crop_size) // 2
        res = res[crop_y_start:(crop_y_start + crop_size), crop_x_start:(crop_x_start + crop_size)]
#         print (res)
#         print (res.shape)
        return res
def reScaleNew(self, img, scale):
        """
        Function to rescale the image in order to make the pixel spacing 1 mm by 1 mm.
        Args:
          img: numpy image array
          scale: list of two values with the scaling in the x direction and y direction
        Returns:
          Numpy array image with the pixel spacing 1 mm by 1 mm
        """
        return cv2.resize(img, (0, 0), fx=scale[0], fy=scale[1])