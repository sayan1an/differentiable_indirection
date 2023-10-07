import numpy as np
import sys
sys.path.insert(1, '../')
import utility as ut
import networks
import networksBase as nb
import os
import imageTools as it
import torch
import glob

# cmd: python ./eval.py 6 "Network_p2_c4_401" "6_0"

className = ut.getSysArgv(2)
imageFolder = ut.getSysArgv(3)
experimentName = className
compressionRatio = int(ut.getSysArgv(1))

baseDataDirectory = ut.getBaseDirectory()
experimentName += "_CR" + str(compressionRatio)

outputDirectory = baseDataDirectory + "DifferentiableIndirectionOutput/" + imageFolder + "/" + experimentName + "/"

assert os.path.exists(outputDirectory), "Trained network directory does not exist."

def infer():
    if not os.path.exists(outputDirectory + "eval/"):
        os.makedirs(outputDirectory + "eval/")
    torchDevice = ut.getTorchDevice("cpu")

    binPath = outputDirectory + "*.bin"
    binFiles = glob.glob(binPath)
    
    networkBlob = torch.load(binFiles[0])
    networkName = networkBlob["ht_class"]
    networkFunction = getattr(networks, networkName)
    assert className == networkName
    assert compressionRatio == networkBlob["ht_compressionRatioExpected"]
    network = networkFunction(torchDevice, networkBlob["ht_resNative"], compressionRatio)
    network.load_state_dict(networkBlob["ht_state"])
    network.precompute()
    network.eval()

    with torch.no_grad():
        uvSamples = ut.generateUvCoord(torchDevice, networkBlob["ht_resNative"], networkBlob["ht_resNative"]).reshape((-1,2))
        networkOutput = network.infer(uvSamples).cpu().numpy().reshape((networkBlob["ht_resNative"], networkBlob["ht_resNative"], -1))
        it.saveThumbnailCol(outputDirectory + "eval/networkOutput", it.toUint8(networkOutput[::-1]))

infer()