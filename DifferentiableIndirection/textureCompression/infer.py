import numpy as np
import sys
sys.path.insert(1, '../')
import utility as ut
import torch.optim as optim
import networks
import os
import imageTools as it
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

className = "Hash2x_41"
quantizeUV = 8
quantizeRGB = 8

#BlockCompress_Arch0, BlockCompress_Arch1
compressionRatio = int(ut.getSysArgv(1))

baseDataDirectory = ut.getBaseDirectory()
experimentName = className + "_CR" + str(compressionRatio)

#if ut.isCluster():
experimentName += "_Cluster"

textureDirectory = baseDataDirectory + "NeuralTextureMappingData/textureCache/independentTex/"
outputDirectory = baseDataDirectory + "NeuralTextureMappingOutput/textureCompressionPaper/" + experimentName + "/"

if not os.path.exists(outputDirectory):
    os.makedirs(outputDirectory)

torchDevice = ut.getTorchDevice("cuda")

def getRgbTexList():
    textureNames = []
    textures = []
    textureIdxs = []
    idx = 0
    for filename in os.listdir(textureDirectory):
        f = os.path.join(textureDirectory, filename)
        if os.path.isfile(f):
            rgb = it.readImage(f)
            assert rgb.shape[0] == rgb.shape[1], "Do not support non-square textures"
            assert np.min(rgb) >= 0.0 and np.max(rgb) <= 1.0, "Problem with file " + f
            textureNames.append(filename)
            #plt.imshow(rgb[:,:,:3])
            #plt.show()
            textures.append(rgb[:,:,:3])
            textureIdxs.append(idx)
            idx += 1
    
    return textureNames, textures, textureIdxs

def compareSSIM(imageRef, imageTarget):
    return ssim(imageRef, imageTarget, data_range=imageTarget.max() - imageTarget.min(), channel_axis=2)

def infer():
    textureNames, textures, textureIdxs = getRgbTexList()
    batchResolution = 1024

    hashTables = []
    torchTargets = []

    for (texture, textureName) in zip(textures, textureNames):
        modelParams = torch.load(outputDirectory + ut.getFileNameWoExt(textureName) + ".bin")
        assert compressionRatio == modelParams["ht_compressionRatioExpected"]
        assert int(np.around(np.sqrt(texture.shape[0] * texture.shape[1]))) == modelParams["ht_resNative"]
        assert modelParams["ht_class"] == className
        netFn = getattr(networks, className)
        ht = netFn(torchDevice, modelParams["ht_resNative"], modelParams["ht_compressionRatioExpected"])
        ht.load_state_dict(modelParams["ht_state"])

        with torch.no_grad():
            uvSamples = ut.generateUvCoord(torchDevice, texture.shape[0], texture.shape[1]).reshape((-1,2))
            ht.eval()
            #rgb = ht.quantizeInfer(uvSamples, quantizeUV, quantizeRGB)
            rgb = ht(uvSamples)
            output = rgb.reshape((texture.shape[0], texture.shape[1], 3)).cpu().numpy()[::-1]
            ssimScore = compareSSIM(texture, output)
            l1Score = np.mean(np.abs(output - texture))
            print(ssimScore)
            print(l1Score)
            np.savez_compressed(ut.getFileNameWoExt(textureName) + "_noQuant_recon_final", output)
            it.saveThumbnailCol(ut.getFileNameWoExt(textureName) + "_noQuant_recon_final", it.toUint8(output))
            plt.imshow(output)
            plt.show()

infer()