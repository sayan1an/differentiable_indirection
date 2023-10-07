import numpy as np
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# import os
# import utility as ut
# import ssimLoss
# import flipLoss
# import matplotlib.pyplot as plt
#import imageio.v2 as imageio
# Uncomment and run once
#import imageio
#imageio.plugins.freeimage.download()

def toUint8(inp):
    return (255.01 * np.clip(inp, 0, 1)).round().astype(np.uint8)

def saveThumbnailCol(name, arr, ext=".jpg"):
    im = Image.fromarray(arr)
    im.save(name + ext)

# def readExr(fileName):
#     f = imageio.imread(fileName, "exr")
#     return (f + 1.0) * 0.5

# def readHdr(fileName):
#     f = imageio.imread(fileName, "hdr")
#     return f

# def readPng(fileName):
#     f = imageio.imread(fileName, "png")
#     return f.astype(np.float32) / 255.0

# def readJpg(fileName):
#     f = imageio.imread(fileName, "jpg")
#     return f.astype(np.float32) / 255.0

def readImg(fileName):
    f = np.asarray(PIL.Image.open(fileName))
    return f.astype(np.float32) / 255.0

def readImgUint(fileName):
    f = np.asarray(PIL.Image.open(fileName))
    return f

def rgba2rgb(rgba, background=(1.0,1.0,1.0)):
    row, col, ch = rgba.shape
    
    if ch == 3:
        return rgba

    assert ch == 4, 'RGBA image has 4 channels.'

    rgb = np.zeros((row, col, 3), dtype='float32')
    r, g, b, a = rgba[:,:,0], rgba[:,:,1], rgba[:,:,2], rgba[:,:,3]

    a = np.asarray( a, dtype='float32' )

    R, G, B = background

    rgb[:,:,0] = r * a + (1.0 - a) * R
    rgb[:,:,1] = g * a + (1.0 - a) * G
    rgb[:,:,2] = b * a + (1.0 - a) * B

    return rgb

def imgResize(torchImg, size):
    res = (torchImg.shape[0] // size + 1) * size
   
    while (res >= size):
        uvList = ut.generateUvCoord(torchImg.device, res, res).reshape((-1,2))
        torchImg = torch.flip(bilinear2d(uvList, torchImg)[0].reshape((res, res, 3)), [0])

        res = res // 2
        if res < size:
            uvList = ut.generateUvCoord(torchImg.device, size, size).reshape((-1,2))
            torchImg = torch.flip(bilinear2d(uvList, torchImg)[0].reshape((size, size, 3)), [0])

    return torchImg

def readImage(fileName):
    if fileName[-3:] == "jpg":
        return rgba2rgb(readImg(fileName))
    elif fileName[-3:] == "png":
        return rgba2rgb(readImg(fileName))
    elif fileName[-3:] == "bmp":
        return rgba2rgb(readImg(fileName))
    elif fileName[-3:] == "hdr":
        return rgba2rgb(readHdr(fileName))
    else:
        print("ReadImage extension not found")
        return None

# def compareSSIM(imageRef, imageTarget):
#     return ssim(imageRef, imageTarget, data_range=imageTarget.max() - imageTarget.min(), channel_axis=2)

# def getRgbTexList(textureDirectory):
#     textureNames = []
#     textures = []
#     textureIdxs = []
#     idx = 0
#     for filename in os.listdir(textureDirectory):
#         f = os.path.join(textureDirectory, filename)
#         if os.path.isfile(f):
#             rgb = readImage(f)
#             assert rgb.shape[0] == rgb.shape[1], "Do not support non-square textures"
#             assert np.min(rgb) >= 0.0 and np.max(rgb) <= 1.0, "Problem with file " + f
#             textureNames.append(filename)
#             #plt.imshow(rgb[:,:,:3])
#             #plt.show()
#             textures.append(rgb[:,:,:3])
#             textureIdxs.append(idx)
#             idx += 1
    
#     return textureNames, textures, textureIdxs

# def findAvgSSIM():
#     baseDataDirectory = "../"
#     experimentName = "Hash2x_321_3D_CR4"#"Hash2l_PCA_Indp_LVL19_CR16_T2N_Cluster"
#     referenceDirectory = baseDataDirectory + "NeuralTextureMappingData/textureCache/independentTex/"
#     refTexNames, refTex, refTexIdxs = getRgbTexList(referenceDirectory)

#     outputDirectory = baseDataDirectory + "NeuralTextureMappingOutput/textureCompressionPaper3DHT/" + experimentName + "/"

#     i = 0
#     avgSSIM = 0
#     avgL1 = 0
#     for tex in refTexNames:
#         outTex = np.clip(np.load(outputDirectory + tex[:-4] + ".npz")['arr_0'], 0, 1)
#         l1Score = np.mean(np.abs(outTex - refTex[i]))
#         avgL1 += l1Score
#         #plt.imshow(np.abs(outTex - refTex[i]))
#         #plt.show()
#         print(tex)
#         ssimScore = compareSSIM(refTex[i], outTex)
#         print(ssimScore)
#         print(l1Score)
#         avgSSIM += ssimScore
#         i+=1
#     print(avgSSIM / len(refTexNames))
#     print(avgL1 / len(refTexNames))
# findAvgSSIM()