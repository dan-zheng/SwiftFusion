// Copyright 2020 The SwiftFusion Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

import Foundation
import TensorFlow
import XCTest

import PenguinStructures
@testable import SwiftFusion

class PPCATrackingFactorTests: XCTestCase {
  /// Tests that the custom `linearized(at:)` method produces the same results at autodiff.
  func testLinearizedValue() {
    let factor = PPCATrackingFactor.testFixture(TypedID<Pose2>(0), TypedID<Vector5>(0), seed: (4, 4))

    for _ in 0..<2 {
      let linearizationPoint = Tuple2(
        Pose2(randomWithCovariance: eye(rowCount: 3), seed: (5, 5)),
        Vector5(flatTensor: Tensor(randomNormal: [5], seed: (6, 6))))

      typealias Variables = PPCATrackingFactor.Variables.TangentVector
      typealias ErrorVector = PPCATrackingFactor.ErrorVector

      // Below we compare custom and much faster differentiation with the autodiff version.

      let forwardAutodiff = ForwardJacobianFactor<Array<ErrorVector>, Variables>(
        linearizing: factor, at: linearizationPoint)
      let autodiff = JacobianFactor<Array<Variables>, ErrorVector>(
        linearizing: factor, at: linearizationPoint)
      let custom = factor.linearized(at: linearizationPoint)

      // Notebook: takes 10-20 seconds instead of <1 second
      // Compare the linearizations at zero (the error vector).
      XCTAssertEqual(
        custom.errorVector(at: Variables.zero), autodiff.errorVector(at: Variables.zero))
      XCTAssertEqual(
        autodiff.errorVector(at: Variables.zero),
        forwardAutodiff.errorVector(at: Variables.zero))

      // Compare the Jacobian-vector-products (forward derivative).
      for _ in 0..<10 {
        let v = Variables(flatTensor: Tensor(randomNormal: [Variables.dimension], seed: (7, 7)))
        assertEqual(
          custom.errorVector_linearComponent(v).tensor,
          autodiff.errorVector_linearComponent(v).tensor,
          accuracy: 1e-6)
        assertEqual(
          custom.errorVector_linearComponent(v).tensor,
          forwardAutodiff.errorVector_linearComponent(v).tensor,
          accuracy: 1e-6)
      }

      // Compare the vector-Jacobian-products (reverse derivative).
      for _ in 0..<10 {
        let e = ErrorVector(Tensor(randomNormal: ErrorVector.shape, seed: (8, 8)))
        assertEqual(
          custom.errorVector_linearComponent_adjoint(e).flatTensor,
          autodiff.errorVector_linearComponent_adjoint(e).flatTensor,
          accuracy: 1e-6)
        assertEqual(
          custom.errorVector_linearComponent_adjoint(e).flatTensor,
          forwardAutodiff.errorVector_linearComponent_adjoint(e).flatTensor,
          accuracy: 1e-6)
      }
    }
  }

  /// Tests that factor graphs with a `PPCATrackingFactor`s linearize them using the custom
  /// linearization.
  func testFactorGraphUsesCustomLinearized() {
    var x = VariableAssignments()
    let poseId = x.store(Pose2(100, 100, 0))
    let latentId = x.store(Vector5.zero)

    var fg = FactorGraph()
    fg.store(PPCATrackingFactor.testFixture(poseId, latentId, seed: (9, 9)))

    let gfg = fg.linearized(at: x)

    // Assert that the linearized graph has the custom `LinearizedPPCATrackingFactor` that uses our
    // faster custom Jacobian calculate the default `JacobianFactor` that calculates the Jacobian
    // using autodiff.
    //
    // This works by checking that the first (only) element of the graph's storage can be cast to
    // an array of `LinearizedPPCATrackingFactor`.
    XCTAssertNotNil(gfg.storage.first!.value[elementType: Type<LinearizedPPCATrackingFactor>()])
  }
}
