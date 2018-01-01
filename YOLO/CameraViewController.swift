import UIKit
import Metal
import MetalPerformanceShaders
import AVFoundation
import CoreMedia

// The labels for the 20 classes.
let labels = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class CameraViewController: UIViewController {
    
    @IBOutlet weak var videoPreview: UIView!
    @IBOutlet weak var timeLabel: UILabel!
    
    var videoCapture: VideoCapture!
    var device: MTLDevice!
    var commandQueue: MTLCommandQueue!
    var network: YOLO!
    
    var startupGroup = DispatchGroup()
    
    var boundingBoxes = [BoundingBox]()
    var colors: [UIColor] = []
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        timeLabel.text = ""
        
        device = MTLCreateSystemDefaultDevice()
        if device == nil {
            print("Error: this device does not support Metal")
            return
        }
        
        commandQueue = device.makeCommandQueue()
        
        // The app can show up to 10 detections at a time. You can increase this
        // limit by allocating more BoundingBox objects, but there's only so much
        // room on the screen. (You also need to change the limit in YOLO.swift.)
        for _ in 0..<10 {
            boundingBoxes.append(BoundingBox())
        }
        
        // Make colors for the bounding boxes. There is one color for each class,
        // 20 classes in total.
        for r: CGFloat in [0.2, 0.4, 0.6, 0.8, 1.0] {
            for g: CGFloat in [0.3, 0.7] {
                for b: CGFloat in [0.4, 0.8] {
                    let color = UIColor(red: r, green: g, blue: b, alpha: 1)
                    colors.append(color)
                }
            }
        }
        
        videoCapture = VideoCapture(device: device)
        videoCapture.delegate = self
        videoCapture.fps = 5
        
        // Initialize the camera.
        startupGroup.enter()
        videoCapture.setUp(sessionPreset: AVCaptureSessionPreset640x480) { success in
            // Add the video preview into the UI.
            if let previewLayer = self.videoCapture.previewLayer {
                self.videoPreview.layer.addSublayer(previewLayer)
                self.resizePreviewLayer()
            }
            self.startupGroup.leave()
        }
        
        // Initialize the neural network.
        startupGroup.enter()
        createNeuralNetwork {
            self.startupGroup.leave()
        }
        
        startupGroup.notify(queue: .main) {
            // Add the bounding box layers to the UI, on top of the video preview.
            for box in self.boundingBoxes {
                box.addToLayer(self.videoPreview.layer)
            }
            
            // Once the NN is set up, we can start capturing live video.
            self.videoCapture.start()
        }
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        print(#function)
    }
    
    // MARK: - UI stuff
    
    override func viewWillLayoutSubviews() {
        super.viewWillLayoutSubviews()
        resizePreviewLayer()
    }
    
    override var preferredStatusBarStyle: UIStatusBarStyle {
        return .lightContent
    }
    
    func resizePreviewLayer() {
        videoCapture.previewLayer?.frame = videoPreview.bounds
    }
    
    // MARK: - Neural network
    
    func createNeuralNetwork(completion: @escaping () -> Void) {
        // Make sure the current device supports MetalPerformanceShaders.
        guard MPSSupportsMTLDevice(device) else {
            print("Error: this device does not support Metal Performance Shaders")
            return
        }
        
        // Because it may take a few seconds to load the network's parameters,
        // perform the construction of the neural network in the background.
        DispatchQueue.global().async {
            
            self.network = YOLO(device: self.device)
            
            DispatchQueue.main.async(execute: completion)
        }
    }
    
    func predict(texture: MTLTexture) {
        // Since we want to run in "realtime", every call to predict() results in
        // a UI update on the main thread. It would be a waste to make the neural
        // network do work and then immediately throw those results away, so the
        // network should not be called more often than the UI thread can handle.
        // It is up to VideoCapture to throttle how often the neural network runs.
        
        
        // It takes between 0.15-0.3 seconds to perform a forward pass of the net.
        // YOLO.predict() blocks until the GPU is ready, so to prevent the app's
        // UI from being blocked we call that method from a background thread.
        DispatchQueue.global().async {
            let startTime = CACurrentMediaTime()
            let inputImage = MPSImage(texture: texture, featureChannels: 3)
            let prediction = self.network.predict(image: inputImage, bgr: true)
            let endTime = CACurrentMediaTime()
            let elapsedTime = endTime-startTime
            
            DispatchQueue.main.async {
                self.show(predictions: prediction)
                self.timeLabel.text = String(format: "Time consuming: %.5f seconds (%.2f FPS)", elapsedTime, 1/elapsedTime)
            }
        }
        
    }
    
    private func show(predictions: [YOLO.Prediction]) {
        for i in 0..<boundingBoxes.count {
            if i < predictions.count {
                let prediction = predictions[i]
                
                // The predicted bounding box is in the coordinate space of the input
                // image, which is a square image of 416x416 pixels. We want to show it
                // on the video preview, which is as wide as the screen and has a 4:3
                // aspect ratio. The video preview also may be letterboxed at the top
                // and bottom.
                let width = view.bounds.width
                let height = width * 4 / 3
                let scaleX = width / 416
                let scaleY = height / 416
                let top = (view.bounds.height - height) / 2
                
                // Translate and scale the rectangle to our own coordinate system.
                var rect = prediction.rect
                rect.origin.x *= scaleX
                rect.origin.y *= scaleY
                rect.origin.y += top
                rect.size.width *= scaleX
                rect.size.height *= scaleY
                
                // Show the bounding box.
                let label = String(format: "%@ %.1f", labels[prediction.classIndex], prediction.score * 100)
                let color = colors[prediction.classIndex]
                boundingBoxes[i].show(frame: rect, label: label, color: color)
                
            } else {
                boundingBoxes[i].hide()
            }
        }
    }
}

extension CameraViewController: VideoCaptureDelegate {
    func videoCapture(_ capture: VideoCapture, didCaptureVideoTexture texture: MTLTexture?, timestamp: CMTime) {
        // Call the predict() method, which encodes the neural net's GPU commands,
        // on our own thread. Since NeuralNetwork.predict() can block, so can our
        // thread. That is OK, since any new frames will be automatically dropped
        // while the serial dispatch queue is blocked.
        if let texture = texture {
            predict(texture: texture)
        }
    }
    
    func videoCapture(_ capture: VideoCapture, didCapturePhotoTexture texture: MTLTexture?, previewImage: UIImage?) {
        // not implemented
    }
}
