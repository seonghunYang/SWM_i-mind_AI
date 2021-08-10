
def checkImgExtension(filename):
    extension = filename[filename.rfind(".")+1:]
    if extension == "png" or extension == "jpg" or extension == "jpeg":
        return True
    return False