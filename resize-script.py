
def main(src: str, ratio = 2):
    import cv2
    
    img = cv2.imread(src, cv2.IMREAD_UNCHANGED)
    
    print('Original Dimensions : ',img.shape)
    
    width = int(img.shape[1] * ratio)
    height = int(img.shape[0] * ratio)
    dim = (width, height)
    
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    from PIL import Image
    return Image.fromarray(resized)

if __name__ == '__main__':
    img = main('example_images/gen9.png')
    img.save('example_images/gen9-1024.png')
