import ArgumentParser

import SwiftFusion
import BeeDataset
import BeeTracking
import TensorFlow
import PythonKit
import Foundation

/// Fan10: RP Tracker, using the new tracking model
struct Fan10: ParsableCommand {
  
  typealias LikelihoodModel = TrackingLikelihoodModel<PretrainedDenseRAE, MultivariateGaussian, MultivariateGaussian>

  @Option(help: "Run on track number x")
  var trackId: Int = 3

  @Option(help: "Run for number of frames")
  var trackLength: Int = 80

  @Option(help: "Size of feature space")
  var featureSize: Int = 20

  @Flag(help: "Training mode")
  var training: Bool = false

  func getTrainingDataEM(
    from dataset: OISTBeeVideo,
    numberForeground: Int = 300,
    numberBackground: Int = 300
  ) -> [LikelihoodModel.Datum] {
    let bgBoxes = dataset.makeBackgroundBoundingBoxes(patchSize: (40, 70), batchSize: numberBackground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.bg, obb: $0.obb)
    }
    let fgBoxes = dataset.makeForegroundBoundingBoxes(patchSize: (40, 70), batchSize: numberForeground).map {
      (frame: $0.frame, type: LikelihoodModel.PatchType.fg, obb: $0.obb)
    }
    
    return fgBoxes + bgBoxes
  }
  
  // Just runs an RP tracker and saves image to file
  // Make sure you have a folder `Results/fan10` before running
  func run() {
    let kHiddenDimension = 100
    let dataDir = URL(fileURLWithPath: "./OIST_Data")

    let generator = ARC4RandomNumberGenerator(seed: 42)
    var em = MonteCarloEM<LikelihoodModel>(sourceOfEntropy: generator)
    
    let trainingDataset = OISTBeeVideo(directory: dataDir, length: 30)!
    
    let trainingData = getTrainingDataEM(from: trainingDataset)
    
    let trackingModel = em.run(
      with: trainingData,
      iterationCount: 3,
      hook: { i, _, _ in
        print("EM run iteration \(i)")
      },
      given: LikelihoodModel.HyperParameters(
        encoder: PretrainedDenseRAE.HyperParameters(hiddenDimension: kHiddenDimension, latentDimension: featureSize, weightFile: "./oist_rae_weight_\(featureSize).npy")
      )
    )
    
    let imagesPath = URL(fileURLWithPath: "Results/fan10/fan10_ae_mg_mg")
    if !FileManager.default.fileExists(atPath: imagesPath.absoluteString) {
        do {
            try FileManager.default.createDirectory(atPath: imagesPath.absoluteString, withIntermediateDirectories: true, attributes: nil)
        } catch {
            print(error.localizedDescription);
        }
    }
    
    let (fig, track, gt) = runProbabilisticTracker(
      directory: dataDir,
      likelihoodModel: trackingModel,
      onTrack: trackId, forFrames: trackLength, withSampling: true,
      withFeatureSize: featureSize,
      savePatchesIn: "Results/fan10/fan10_ae_mg_mg"
    )

    /// Actual track v.s. ground truth track
    fig.savefig("Results/fan10/fan10_ae_mg_mg_track\(trackId)_\(featureSize).pdf", bbox_inches: "tight")
    fig.savefig("Results/fan10/fan10_ae_mg_mg_track\(trackId)_\(featureSize).png", bbox_inches: "tight")

    let json = JSONEncoder()
    json.outputFormatting = .prettyPrinted

    let track_data = try! json.encode(track)
    try! track_data.write(to: URL(fileURLWithPath: "Results/fan10/fan10_ae_mg_mg_track_\(trackId)_\(featureSize).json"))

    let gt_data = try! json.encode(gt)
    try! gt_data.write(to: URL(fileURLWithPath: "Results/fan10/fan10_ae_mg_mg_gt_\(trackId)_\(featureSize).json"))
  }
}
