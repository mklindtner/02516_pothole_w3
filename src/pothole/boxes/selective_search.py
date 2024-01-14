import cv2 as cv


SEARCH_OBJ = cv.ximgproc.segmentation.createSelectiveSearchSegmentation()


def run_selective_search(image, fast=True):
    SEARCH_OBJ.setBaseImage(image)

    if fast:
        SEARCH_OBJ.switchToSelectiveSearchFast()
    else:
        SEARCH_OBJ.switchToSelectiveSearchQuality()

    return SEARCH_OBJ.process()
