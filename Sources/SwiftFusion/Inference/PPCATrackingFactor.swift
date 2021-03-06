import PenguinStructures
import TensorFlow

/// A factor over a target's pose and appearance in an image.
public struct PPCATrackingFactor: LinearizableFactor2 {
  /// The first adjacent variable, the pose of the target in the image.
  ///
  /// This explicitly specifies `LinearizableFactor2`'s `associatedtype V0`.
  public typealias V0 = Pose2

  /// The second adjacent variable, the PPCA latent code for the appearance of the target.
  ///
  /// This explicitly specifies `LinearizableFactor2`'s `associatedtype V1`.
  public typealias V1 = Vector5

  /// A region cropped from the image.
  public typealias Patch = Tensor28x62x1

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// The image containing the target.
  public let measurement: Tensor<Double>

  /// The PPCA appearance model weight matrix.
  ///
  /// The shape is `Patch.shape + [V1.dimension]`.
  public var W: Tensor<Double>

  /// The PPCA appearance model mean.
  public var mu: Patch

  /// Creates an instance.
  ///
  /// - Requires `W.shape == Patch.shape + [V1.dimension]`.
  public init(
    _ poseId: TypedID<Pose2>,
    _ latentId: TypedID<Vector5>,
    measurement: Tensor<Double>,
    W: Tensor<Double>,
    mu: Patch
  ) {
    precondition(W.shape == Patch.shape + [V1.dimension])
    self.edges = Tuple2(poseId, latentId)
    self.measurement = measurement
    self.W = W
    self.mu = mu
  }

  /// Returns the difference between the PPCA generated `Patch` and the `Patch` cropped from
  /// `measurement`.
  @differentiable
  public func errorVector(_ pose: Pose2, _ latent: Vector5) -> Patch.TangentVector {
    return generatedAppearance(latent) - Patch(measurement.patch(at: region(pose)))
  }

  /// Returns a linear approximation to `self` at `x`.
  public func linearized(at x: Variables) -> LinearizedPPCATrackingFactor {
    let pose = x.head
    let latent = x.tail.head
    let actualAppearance = measurement.patchWithJacobian(at: region(pose))
    return LinearizedPPCATrackingFactor(
      error: Patch(actualAppearance.patch) - generatedAppearance(latent),
      errorVector_H_pose_latent: Tensor(concatenating: [-actualAppearance.jacobian, W], alongAxis: -1),
      edges: Variables.linearized(edges))
  }

  /// Returns the appearance generated by the generative (PPCA) model with parameters `latent`.
  private func generatedAppearance(_ latent: Vector5) -> Patch {
    return mu + Patch(matmul(W, latent.flatTensor.expandingShape(at: 1)).squeezingShape(at: 2))
  }

  /// Returns the oriented region of the image at `center`, with the size of the patch we are
  /// tracking.
  private func region(_ center: Pose2) -> OrientedBoundingBox {
    OrientedBoundingBox(center: center, rows: Patch.shape[0], cols: Patch.shape[1])
  }
}

/// A linear approximation to `PPCATrackingFactor`, at a certain linearization point.
public struct LinearizedPPCATrackingFactor: GaussianFactor {

  /// The tangent vectors of the `PPCATrackingFactor`'s "pose" and "latent" variables.
  public typealias Variables = Tuple2<Pose2.TangentVector, Vector5>

  /// The error vector at the linearization point.
  public let error: PPCATrackingFactor.Patch

  /// The linear transformation mapping small changes in input variables to small changes in error
  /// around the linearization point.
  ///
  /// The shape is `PPCATrackingFactor.Patch.shape + [Variables.dimension]`.
  public let errorVector_H_pose_latent: Tensor<Double>

  /// The IDs of the variables adjacent to this factor.
  public let edges: Variables.Indices

  /// Creates an instance with the given `errorVector` and `errorVector_H_pose_latent`.
  ///
  /// - Requires: `errorVector_H_pose_latent.shape
  //    == PPCATrackingFactor.Patch.shape + [Variables.dimension]`.
  public init(
    error: PPCATrackingFactor.Patch,
    errorVector_H_pose_latent: Tensor<Double>,
    edges: Variables.Indices
  ) {
    precondition(
      errorVector_H_pose_latent.shape == PPCATrackingFactor.Patch.shape + [Variables.dimension])
    self.error = error
    self.errorVector_H_pose_latent = errorVector_H_pose_latent
    self.edges = edges
  }

  public func error(at x: Variables) -> Double {
    return 0.5 * errorVector(at: x).squaredNorm
  }

  @differentiable
  public func errorVector(at x: Variables) -> PPCATrackingFactor.Patch {
    errorVector_linearComponent(x) - error
  }

  public func errorVector_linearComponent(_ x: Variables) -> PPCATrackingFactor.Patch {
    PPCATrackingFactor.Patch(
      matmul(errorVector_H_pose_latent, x.flatTensor.expandingShape(at: 1)).squeezingShape(at: 2))
  }

  public func errorVector_linearComponent_adjoint(_ y: PPCATrackingFactor.Patch) -> Variables {
    Variables(
      flatTensor: matmul(
        errorVector_H_pose_latent.reshaped(to: [PPCATrackingFactor.Patch.dimension, Variables.dimension]),
        transposed: true,
        y.tensor.reshaped(to: [PPCATrackingFactor.Patch.dimension, 1])
      ).squeezingShape(at: 1)
    )
  }
}

extension PPCATrackingFactor {
  /// Returns an example `PPCATrackingFactor` on variables at `poseID` and `latentID`, for use
  /// during tests.
  ///
  /// - Parameter seed: Seeds the randomness used while generating the factor.
  public static func testFixture(
    _ poseID: TypedID<Pose2>,
    _ latentID: TypedID<Vector5>,
    seed: TensorFlowSeed = Context.local.randomSeed
  ) -> Self {
    let image = Tensor<Double>(randomNormal: [500, 500, 1], seed: seed)
    let W = Tensor<Double>(
      randomNormal: PPCATrackingFactor.Patch.shape + [PPCATrackingFactor.V1.dimension],
      seed: seed)
    let mu = PPCATrackingFactor.Patch(
      Tensor<Double>(randomNormal: PPCATrackingFactor.Patch.shape, seed: seed))
    return PPCATrackingFactor(poseID, latentID, measurement: image, W: W, mu: mu)
  }
}
