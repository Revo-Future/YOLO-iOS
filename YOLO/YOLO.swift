import MetalPerformanceShaders
import QuartzCore

let anchors: [Float] = [1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52]

/*
  The tiny-yolo-voc network from YOLOv2. https://pjreddie.com/darknet/yolo/

  This implementation is cobbled together from the following sources:

  - https://github.com/pjreddie/darknet
  - https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/src/org/tensorflow/demo/TensorFlowYoloDetector.java
  - https://github.com/allanzelener/YAD2K
*/


class YOLO {
    
    struct Prediction {
        let classIndex: Int
        let score: Float
        let rect: CGRect
    }
    
    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    
    // The neural network expects a 416x416x3 pixel image. We use a lanczos filter
    // to scale the input image down to these dimensions.
    let lanczos: MPSImageLanczosScale
    
    
    /* The layers in the network: */
    let conv1: MPSCNNConvolution  // input = 416x416x3  , kernels = 3x3x3x16.
    let maxpooling1: MPSCNNPoolingMax  //2x2,stride=2
    let conv2: MPSCNNConvolution // input = 208x208x16  , kernels = 3x3x16x32.
    let maxpooling2: MPSCNNPoolingMax  //2x2,stride=2
    let conv3: MPSCNNConvolution // input = 104x104x32  , kernels = 3x3x32x64.
    let maxpooling3: MPSCNNPoolingMax  //2x2,stride=2
    let conv4: MPSCNNConvolution // input = 52x52x64  , kernels = 3x3x64x128.
    let maxpooling4: MPSCNNPoolingMax  //2x2,stride=2
    let conv5: MPSCNNConvolution // input = 26x26x128  , kernels = 3x3x128x256.
    let maxpooling5: MPSCNNPoolingMax  //2x2,stride=2
    let conv6: MPSCNNConvolution // input = 13x13x256  , kernels = 3x3x256x512.
    let maxpooling6: MPSCNNPoolingMax  //2x2,stride=2, padding=1
    let conv7: MPSCNNConvolution // input = 13x13x512  , kernels = 3x3x512x1024.
    let conv8: MPSCNNConvolution // input = 13x13x1024  , kernels = 3x3x1024x1024.
    let conv9: MPSCNNConvolution // input = 13x13x1024  , kernels = 1x1x1024x125.
    
    
    /* These MPSImage descriptors tell the network about the sizes of the data
     volumes that flow between the layers. */
    
    let input_id  = MPSImageDescriptor(channelFormat: .float16, width: 416, height: 416, featureChannels: 3)
    let conv1_id  = MPSImageDescriptor(channelFormat: .float16, width: 416, height: 416, featureChannels: 16)
    let pool1_id = MPSImageDescriptor(channelFormat: .float16, width: 208, height: 208, featureChannels: 16)
    let conv2_id  = MPSImageDescriptor(channelFormat: .float16, width: 208, height: 208, featureChannels: 32)
    let pool2_id = MPSImageDescriptor(channelFormat: .float16, width: 104, height: 104, featureChannels: 32)
    let conv3_id  = MPSImageDescriptor(channelFormat: .float16, width: 104, height: 104, featureChannels: 64)
    let pool3_id = MPSImageDescriptor(channelFormat: .float16, width: 52, height: 52,  featureChannels: 64)
    let conv4_id  = MPSImageDescriptor(channelFormat: .float16, width: 52, height: 52, featureChannels: 128)
    let pool4_id = MPSImageDescriptor(channelFormat: .float16, width: 26, height: 26,  featureChannels: 128)
    let conv5_id  = MPSImageDescriptor(channelFormat: .float16, width: 26, height: 26, featureChannels: 256)
    let pool5_id = MPSImageDescriptor(channelFormat: .float16, width: 13, height: 13,  featureChannels: 256)
    let conv6_id  = MPSImageDescriptor(channelFormat: .float16, width: 13, height: 13, featureChannels: 512)
    let pool6_id = MPSImageDescriptor(channelFormat: .float16, width: 13, height: 13,  featureChannels: 512)
    let conv7_id  = MPSImageDescriptor(channelFormat: .float16, width: 13, height: 13, featureChannels: 1024)
    let conv8_id  = MPSImageDescriptor(channelFormat: .float16, width: 13, height: 13, featureChannels: 1024)
    let conv9_id  = MPSImageDescriptor(channelFormat: .float16, width: 13, height: 13, featureChannels: 125)
    
    let conv9_img: MPSImage //conv9_img is the final convolution feature map.
    
    public init(device: MTLDevice) {
        print("Setting up neural network...")
        let startTime = CACurrentMediaTime()
        
        self.device = device
        commandQueue = device.makeCommandQueue()
        
        conv9_img = MPSImage(device: device, imageDescriptor: conv9_id) //save the result
        
        lanczos = MPSImageLanczosScale(device: device)
        
        let relu = MPSCNNNeuronReLU(device: device, a: 0.1)
        
        
        conv1 = SlimMPSCNNConvolution(kernelWidth: 3,
                                         kernelHeight: 3,
                                         inputFeatureChannels: 3,
                                         outputFeatureChannels: 16,
                                         neuronFilter: relu,
                                         device: device,
                                         kernelParamsBinaryName: "conv1",
                                         padding: true,
                                         strideXY: (1,1))
        maxpooling1 = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2, strideInPixelsX: 2, strideInPixelsY: 2)
        conv2 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 16,
                                      outputFeatureChannels: 32,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv2",
                                      padding: true,
                                      strideXY: (1,1))
        maxpooling2 = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2, strideInPixelsX: 2, strideInPixelsY: 2)
        conv3 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 32,
                                      outputFeatureChannels: 64,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv3",
                                      padding: true,
                                      strideXY: (1,1))
        maxpooling3 = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2, strideInPixelsX: 2, strideInPixelsY: 2)
        conv4 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 64,
                                      outputFeatureChannels: 128,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv4",
                                      padding: true,
                                      strideXY: (1,1))
        maxpooling4 = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2, strideInPixelsX: 2, strideInPixelsY: 2)
        conv5 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 128,
                                      outputFeatureChannels: 256,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv5",
                                      padding: true,
                                      strideXY: (1,1))
        maxpooling5 = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2, strideInPixelsX: 2, strideInPixelsY: 2)
        conv6 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 256,
                                      outputFeatureChannels: 512,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv6",
                                      padding: true,
                                      strideXY: (1,1))
        maxpooling6 = MPSCNNPoolingMax(device: device, kernelWidth: 2, kernelHeight: 2, strideInPixelsX: 1, strideInPixelsY: 1)
        //offset setting is necessary to make sure 13x13->13x13 after pooling
        maxpooling6.offset = MPSOffset(x: 2, y: 2, z: 0)
        maxpooling6.edgeMode = MPSImageEdgeMode.clamp
        conv7 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 512,
                                      outputFeatureChannels: 1024,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv7",
                                      padding: true,
                                      strideXY: (1,1))
        conv8 = SlimMPSCNNConvolution(kernelWidth: 3,
                                      kernelHeight: 3,
                                      inputFeatureChannels: 1024,
                                      outputFeatureChannels: 1024,
                                      neuronFilter: relu,
                                      device: device,
                                      kernelParamsBinaryName: "conv8",
                                      padding: true,
                                      strideXY: (1,1))
        conv9 = SlimMPSCNNConvolution(kernelWidth: 1,
                                      kernelHeight: 1,
                                      inputFeatureChannels: 1024,
                                      outputFeatureChannels: 125,
                                      neuronFilter: nil,
                                      device: device,
                                      kernelParamsBinaryName: "conv9",
                                      padding: false,
                                      strideXY: (1,1))
        
        let endTime = CACurrentMediaTime()
        print("Elapsed time: \(endTime - startTime) sec")
    }
    
    /* Performs the inference step. This takes the input image, converts it into
     the format the network expects, then feeds it into the network. The result from network
     is a 13x13x125 tensor. The final result is [Predicion] which indicate the detected box */
    public func predict(image inputImage: MPSImage, bgr: Bool) -> [Prediction] {
        let startTime = CACurrentMediaTime()
        
        autoreleasepool{
            let commandBuffer = commandQueue.makeCommandBuffer()
            
            // Scale the input image to 416x416 pixels.
            let img1 = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: input_id)
            lanczos.encode(commandBuffer: commandBuffer, sourceTexture: inputImage.texture, destinationTexture: img1.texture)
            
            
            let conv1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv1_id)
            conv1.encode(commandBuffer: commandBuffer, sourceImage: img1, destinationImage: conv1_img)
            let pool1_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool1_id)
            maxpooling1.encode(commandBuffer: commandBuffer, sourceImage: conv1_img, destinationImage: pool1_img)
            
            let conv2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv2_id)
            conv2.encode(commandBuffer: commandBuffer, sourceImage: pool1_img, destinationImage: conv2_img)
            let pool2_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool2_id)
            maxpooling2.encode(commandBuffer: commandBuffer, sourceImage: conv2_img, destinationImage: pool2_img)
            
            let conv3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv3_id)
            conv3.encode(commandBuffer: commandBuffer, sourceImage: pool2_img, destinationImage: conv3_img)
            let pool3_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool3_id)
            maxpooling3.encode(commandBuffer: commandBuffer, sourceImage: conv3_img, destinationImage: pool3_img)
            
            let conv4_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv4_id)
            conv4.encode(commandBuffer: commandBuffer, sourceImage: pool3_img, destinationImage: conv4_img)
            let pool4_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool4_id)
            maxpooling4.encode(commandBuffer: commandBuffer, sourceImage: conv4_img, destinationImage: pool4_img)
            
            let conv5_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv5_id)
            conv5.encode(commandBuffer: commandBuffer, sourceImage: pool4_img, destinationImage: conv5_img)
            let pool5_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool5_id)
            maxpooling5.encode(commandBuffer: commandBuffer, sourceImage: conv5_img, destinationImage: pool5_img)
            
            let conv6_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv6_id)
            conv6.encode(commandBuffer: commandBuffer, sourceImage: pool5_img, destinationImage: conv6_img)
            let pool6_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: pool6_id)
            maxpooling6.encode(commandBuffer: commandBuffer, sourceImage: conv6_img, destinationImage: pool6_img)
            
            let conv7_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv7_id)
            conv7.encode(commandBuffer: commandBuffer, sourceImage: pool6_img, destinationImage: conv7_img)
            let conv8_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv8_id)
            conv8.encode(commandBuffer: commandBuffer, sourceImage: conv7_img, destinationImage: conv8_img)
            //let conv9_img = MPSTemporaryImage(commandBuffer: commandBuffer, imageDescriptor: conv9_id)
            conv9.encode(commandBuffer: commandBuffer, sourceImage: conv8_img, destinationImage: conv9_img)
            
            // Tell the GPU to start and wait until it's done.
            commandBuffer.commit()
            commandBuffer.waitUntilCompleted()
        }
        
        
        // Convert the 13x13x125 tensor to rectangle boxes
        let result = fetchResult(image: conv9_img)
        
        let endTime = CACurrentMediaTime()
        print("Elapsed time: \(endTime - startTime) sec")
        
       
        return result
    }

// Convert the 13x13x125 tensor to rectangle boxes
public func fetchResult(image outputImage: MPSImage) -> [Prediction] {
    let featuresImage = outputImage
    let features = featuresImage.toFloatArray()
    assert(features.count == 13*13*128) //because the 3rd of MPSImage must be a multiple of 4
    
    // We only run the convolutional part of YOLO on the GPU. The last part of
    // the process is done on the CPU. It should be possible to do this on the
    // GPU too, but it might not be worth the effort.
    
    var predictions = [Prediction]()
    
    let blockSize: Float = 32  //416/13=32
    let gridHeight = 13
    let gridWidth = 13
    let boxesPerCell = 5
    let numClasses = 20
    
    // This helper function finds the offset in the features array for a given
    // channel for a particular pixel. (See the comment below.)
    func offset(_ channel: Int, _ x: Int, _ y: Int) -> Int {
        let slice = channel / 4
        let indexInSlice = channel - slice*4
        let offset = slice*gridHeight*gridWidth*4 + y*gridWidth*4 + x*4 + indexInSlice
        return offset
    }
    
    // The 416x416 image is divided into a 13x13 grid. Each of these grid cells
    // will predict 5 bounding boxes (boxesPerCell). A bounding box consists of
    // five data items: x, y, width, height, and a confidence score. Each grid
    // cell also predicts which class each bounding box belongs to.
    //
    // The "features" array therefore contains (numClasses + 5)*boxesPerCell
    // values for each grid cell, i.e. 125 channels. The total features array
    // contains 13x13x125 elements (actually x128 instead of x125 because in
    // Metal the number of channels must be a multiple of 4).
    
    for cy in 0..<gridHeight {
        for cx in 0..<gridWidth {
            for b in 0..<boxesPerCell {
                
                // The 13x13x125 image is arranged in planes of 4 channels. First are
                // channels 0-3 for the entire image, then channels 4-7 for the whole
                // image, then channels 8-11, and so on. Since we have 128 channels,
                // there are 128/4 = 32 of these planes (a.k.a. texture slices).
                //
                //    0123 0123 0123 ... 0123    ^
                //    0123 0123 0123 ... 0123    |
                //    0123 0123 0123 ... 0123    13 rows
                //    ...                        |
                //    0123 0123 0123 ... 0123    v
                //    4567 4557 4567 ... 4567
                //    etc
                //    <----- 13 columns ---->
                //
                // For the first bounding box (b=0) we have to read channels 0-24,
                // for b=1 we have to read channels 25-49, and so on. Unfortunately,
                // these 25 channels are spread out over multiple slices. We use a
                // helper function to find the correct place in the features array.
                // (Note: It might be quicker / more convenient to transpose this
                // array so that all 125 channels are stored consecutively instead
                // of being scattered over multiple texture slices.)
                let channel = b*(numClasses + 5)
                let tx = features[offset(channel, cx, cy)]
                let ty = features[offset(channel + 1, cx, cy)]
                let tw = features[offset(channel + 2, cx, cy)]
                let th = features[offset(channel + 3, cx, cy)]
                let tc = features[offset(channel + 4, cx, cy)]
                
                // The predicted tx and ty coordinates are relative to the location
                // of the grid cell; we use the logistic sigmoid to constrain these
                // coordinates to the range 0 - 1. Then we add the cell coordinates
                // (0-12) and multiply by the number of pixels per grid cell (32).
                // Now x and y represent center of the bounding box in the original
                // 416x416 image space.
                let x = (Float(cx) + Math.sigmoid(tx)) * blockSize
                let y = (Float(cy) + Math.sigmoid(ty)) * blockSize
                
                // The size of the bounding box, tw and th, is predicted relative to
                // the size of an "anchor" box. Here we also transform the width and
                // height into the original 416x416 image space.
                let w = exp(tw) * anchors[2*b    ] * blockSize
                let h = exp(th) * anchors[2*b + 1] * blockSize
                
                // The confidence value for the bounding box is given by tc. We use
                // the logistic sigmoid to turn this into a percentage.
                let confidence = Math.sigmoid(tc)
                
                // Gather the predicted classes for this anchor box and softmax them,
                // so we can interpret these numbers as percentages.
                var classes = [Float](repeating: 0, count: numClasses)
                for c in 0..<numClasses {
                    classes[c] = features[offset(channel + 5 + c, cx, cy)]
                }
                classes = Math.softmax(classes)
                
                // Find the index of the class with the largest score.
                let (detectedClass, bestClassScore) = classes.argmax()
                
                // Combine the confidence score for the bounding box, which tells us
                // how likely it is that there is an object in this box (but not what
                // kind of object it is), with the largest class prediction, which
                // tells us what kind of object it detected (but not where).
                let confidenceInClass = bestClassScore * confidence
                
                // Since we compute 13x13x5 = 845 bounding boxes, we only want to
                // keep the ones whose combined score is over a certain threshold.
                if confidenceInClass > 0.3 {
                    let rect = CGRect(x: CGFloat(x - w/2), y: CGFloat(y - h/2),
                                      width: CGFloat(w), height: CGFloat(h))
                    
                    let prediction = Prediction(classIndex: detectedClass,
                                                score: confidenceInClass,
                                                rect: rect)
                    predictions.append(prediction)
                }
            }
        }
    }
    
    // We already filtered out any bounding boxes that have very low scores,
    // but there still may be boxes that overlap too much with others. We'll
    // use "non-maximum suppression" to prune those duplicate bounding boxes.
    let result = nonMaxSuppression(boxes: predictions, limit: 10, threshold: 0.5)
    return result
}
}



/**
 Removes bounding boxes that overlap too much with other boxes that have
 a higher score.
 
 Based on code from https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/kernels/non_max_suppression_op.cc
 
 - Parameters:
 - boxes: an array of bounding boxes and their scores
 - limit: the maximum number of boxes that will be selected
 - threshold: used to decide whether boxes overlap too much
 */
func nonMaxSuppression(boxes: [YOLO.Prediction], limit: Int, threshold: Float) -> [YOLO.Prediction] {
    
    // Do an argsort on the confidence scores, from high to low.
    let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }
    
    var selected: [YOLO.Prediction] = []
    var active = [Bool](repeating: true, count: boxes.count)
    var numActive = active.count
    
    // The algorithm is simple: Start with the box that has the highest score.
    // Remove any remaining boxes that overlap it more than the given threshold
    // amount. If there are any boxes left (i.e. these did not overlap with any
    // previous boxes), then repeat this procedure, until no more boxes remain
    // or the limit has been reached.
    outer: for i in 0..<boxes.count {
        if active[i] {
            let boxA = boxes[sortedIndices[i]]
            selected.append(boxA)
            if selected.count >= limit { break }
            
            for j in i+1..<boxes.count {
                if active[j] {
                    let boxB = boxes[sortedIndices[j]]
                    if IOU(a: boxA.rect, b: boxB.rect) > threshold {
                        active[j] = false
                        numActive -= 1
                        if numActive <= 0 { break outer }
                    }
                }
            }
        }
    }
    return selected
}
