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

import PenguinStructures
import TensorFlow

/// A BetweenFactor alternative that uses the Chordal (Frobenious) norm on rotation for Rot3
/// Please refer to Carlone15icra (Initialization Techniques for 3D SLAM: a Survey on Rotation Estimation and its Use in Pose Graph Optimization)
/// for explanation.
public struct RelaxedRotationFactorRot3: LinearizableFactor
{
  public typealias Variables = Tuple2<Vector9, Vector9>
  public typealias JacobianRows = Array9<Tuple2<Vector9.TangentVector, Vector9.TangentVector>>
  
  public let edges: Variables.Indices
  public let difference: Vector9
  
  public init(_ id1: TypedID<Vector9, Int>, _ id2: TypedID<Vector9, Int>, _ difference: Vector9) {
    self.edges = Tuple2(id1, id2)
    self.difference = difference
  }
  
  public typealias ErrorVector = Vector9
  
  @differentiable
  public func errorVector(_ start: Vector9, _ end: Vector9) -> ErrorVector {
    let R2 = Matrix3(end.s0, end.s1, end.s2, end.s3, end.s4, end.s5, end.s6, end.s7, end.s8)
    let R12 = Matrix3(difference.s0, difference.s1, difference.s2, difference.s3, difference.s4, difference.s5, difference.s6, difference.s7, difference.s8)
    let R = matmul(R12, R2.transposed()).transposed()
    return R.vec - start
  }
  
  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }
  
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head, x.tail.head)
  }
  
  public typealias Linearization = JacobianFactor<JacobianRows, ErrorVector>
  public func linearized(at x: Variables) -> Linearization {
    Linearization(linearizing: errorVector, at: x, edges: edges)
  }
}

/// A factor for the anchor in chordal initialization
public struct RelaxedAnchorFactorRot3: LinearizableFactor
{
  public typealias Variables = Tuple1<Vector9>
  public typealias JacobianRows = Array9<Tuple1<Vector9.TangentVector>>
  
  public let edges: Variables.Indices
  public let prior: Vector9
  
  public init(_ id: TypedID<Vector9, Int>, _ val: Vector9) {
    self.edges = Tuple1(id)
    self.prior = val
  }
  
  public typealias ErrorVector = Vector9
  public func errorVector(_ val: Vector9) -> ErrorVector {
    val + prior
  }
  
  // Note: All the remaining code in this factor is boilerplate that we can eventually eliminate
  // with sugar.
  
  public func error(at x: Variables) -> Double {
    return errorVector(at: x).squaredNorm
  }
  
  public func errorVector(at x: Variables) -> ErrorVector {
    return errorVector(x.head)
  }
  
  public typealias Linearization = JacobianFactor<JacobianRows, ErrorVector>
  public func linearized(at x: Variables) -> Linearization {
    Linearization(linearizing: errorVector, at: x, edges: edges)
  }
}

/// Type shorthands used in the relaxed pose graph
/// NOTE: Specializations are added in `FactorsStorage.swift`
public typealias Jacobian9x9_1 = Array9<Tuple1<Vector9>>
public typealias Jacobian9x9_2 = Array9<Tuple2<Vector9, Vector9>>
public typealias JacobianFactor9x9_1 = JacobianFactor<Jacobian9x9_1, Vector9>
public typealias JacobianFactor9x9_2 = JacobianFactor<Jacobian9x9_2, Vector9>

/// Chordal Initialization for Pose3s
public struct ChordalInitialization {
  /// ID of the anchor used in chordal initialization, should only be used if not using `GetInitializations`.
  public var anchorId: TypedID<Pose3, Int>
  
  public init() {
    anchorId = TypedID<Pose3, Int>(0)
  }
  
  /// Extract a subgraph of the original graph with only Pose3s.
  public func buildPose3graph(graph: FactorGraph) -> FactorGraph {
    var pose3Graph = FactorGraph()
    
    for factor in graph.factors(type: BetweenFactor3.self) {
      pose3Graph.store(factor)
    }
    
    for factor in graph.factors(type: PriorFactor3.self) {
      pose3Graph.store(BetweenFactor3(anchorId, factor.edges.head, factor.prior))
    }
    
    return pose3Graph
  }
  
  /// Solves the unconstrained chordal graph given the original Pose3 graph.
  /// - Parameters:
  ///   - g: The factor graph with only `BetweenFactor<Pose3>` and `PriorFactor<Pose3>`
  ///   - v: the current pose priors
  ///   - ids: the `TypedID`s of the poses
  public func solveOrientationGraph(
    g: FactorGraph,
    v: VariableAssignments,
    ids: Array<TypedID<Pose3, Int>>
  ) -> VariableAssignments {
    /// The orientation graph, with only unconstrained rotation factors
    var orientationGraph = FactorGraph()
    /// orientation storage
    var orientations = VariableAssignments()
    /// association to lookup the vector-based storage from the pose3 ID
    var associations = Dictionary<Int, TypedID<Vector9, Int>>()
    
    // allocate the space for solved rotations, and memory the assocation
    for i in ids {
      associations[i.perTypeID] = orientations.store(Vector9(0, 0, 0, 0, 0, 0, 0, 0, 0))
    }
    
    // allocate the space for anchor
    associations[anchorId.perTypeID] = orientations.store(Vector9(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    
    // iterate the pose3 graph and make corresponding relaxed factors
    for factor in g.factors(type: BetweenFactor3.self) {
      let R = factor.difference.rot.coordinate.R
      let frob_factor = RelaxedRotationFactorRot3(associations[factor.edges.head.perTypeID]!, associations[factor.edges.tail.head.perTypeID]!, R.vec)
      orientationGraph.store(frob_factor)
    }
    
    // make the anchor factor
    orientationGraph.store(RelaxedAnchorFactorRot3(associations[anchorId.perTypeID]!, Vector9(1.0, 0.0, 0.0, /*  */ 0.0, 1.0, 0.0, /*  */ 0.0, 0.0, 1.0)))
    
    // optimize
    var optimizer = GenericCGLS()
    let linearGraph = orientationGraph.linearized(at: orientations)
    optimizer.optimize(gfg: linearGraph, initial: &orientations)
    
    return normalizeRelaxedRotations(orientations, associations: associations, ids: ids)
  }
  
  /// This function finds the closest Rot3 to the unconstrained 3x3 matrix with SVD.
  /// - Parameters:
  ///   - relaxedRot3: the results of the unconstrained chordal optimization
  ///   - associations: mapping from the index of the pose to the index of the corresponding rotation
  ///   - ids: the IDs of the poses
  ///
  /// TODO(fan): replace this with a 3x3 specialized SVD instead of this generic SVD (slow)
  public func normalizeRelaxedRotations(
    _ relaxedRot3: VariableAssignments,
    associations: Dictionary<Int, TypedID<Vector9, Int>>,
    ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var validRot3 = VariableAssignments()
    
    for v in ids {
      let M_v: Vector9 = relaxedRot3[associations[v.perTypeID]!]
      
      let M = Matrix3(M_v.s0, M_v.s1, M_v.s2, M_v.s3, M_v.s4, M_v.s5, M_v.s6, M_v.s7, M_v.s8)
      
      let initRot = Rot3.ClosestTo(mat: M)
      
      // TODO(fan): relies on the assumption of continuous and ordered allocation
      let _ = validRot3.store(initRot)
    }
    
    let M_v_anchor: Vector9 = relaxedRot3[associations[anchorId.perTypeID]!]
    
    let M_anchor = Matrix3(M_v_anchor.s0, M_v_anchor.s1, M_v_anchor.s2,
                           M_v_anchor.s3, M_v_anchor.s4, M_v_anchor.s5,
                           M_v_anchor.s6, M_v_anchor.s7, M_v_anchor.s8)
    
    let initRot_anchor = Rot3.ClosestTo(mat: M_anchor)
    
    // TODO(fan): relies on the assumption of continous and ordered allocation
    let _ = validRot3.store(initRot_anchor)
    
    return validRot3;
  }
  
  /// This function computes the inital poses given the chordal initialized rotations.
  /// - Parameters:
  ///   - graph: The factor graph with only `BetweenFactor<Pose3>` and `PriorFactor<Pose3>`
  ///   - orientations: The orientations returned by the chordal initialization for `Rot3`s
  ///   - ids: the `TypedID`s of the poses
  public func computePoses(graph: FactorGraph, orientations: VariableAssignments, ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var val = VariableAssignments()
    for v in ids {
      let _ = val.store(Pose3(orientations[TypedID<Rot3, Int>(v.perTypeID)], Vector3(0,0,0)))
    }
    
    let _ = val.store(Pose3(orientations[TypedID<Rot3, Int>(anchorId.perTypeID)], Vector3(0,0,0)))
    // optimize for 1 G-N iteration
    for _ in 0..<1 {
      let gfg = graph.linearized(at: val)
      var dx = val.tangentVectorZeros
      var optimizer = GenericCGLS(precision: 1e-1, max_iteration: 100)
      optimizer.optimize(gfg: gfg, initial: &dx)
      val.move(along: (-1) * dx)
    }
    return val
  }
  
  /// This function computes the chordal initialization. Normally this is what the user needs to call.
  /// - Parameters:
  ///   - graph: The factor graph with only `BetweenFactor<Pose3>` and `PriorFactor<Pose3>`
  ///   - ids: the `TypedID`s of the poses
  ///
  /// NOTE: This function builds upon the assumption that all variables stored are Pose3s, will fail if that is not the case.
  public static func GetInitializations(graph: FactorGraph, ids: Array<TypedID<Pose3, Int>>) -> VariableAssignments {
    var ci = ChordalInitialization()
    var val = VariableAssignments()
    for _ in ids {
      let _ = val.store(Pose3())
    }
    ci.anchorId = val.store(Pose3())
    // We "extract" the Pose3 subgraph of the original graph: this
    // is done to properly model priors and avoiding operating on a larger graph
    // TODO(fan): This does not work yet as we have not yet reached concensus on how should we
    // handle associations
    let pose3Graph = ci.buildPose3graph(graph: graph)
    
    // Get orientations from relative orientation measurements
    let orientations = ci.solveOrientationGraph(g: pose3Graph, v: val, ids: ids)
    
    print("Pose3Graph = \(pose3Graph.factors(type: BetweenFactor3.self).map { ($0.edges.head.perTypeID, $0.edges.tail.head.perTypeID) })")
    // Compute the full poses (1 GN iteration on full poses)
    return ci.computePoses(graph: pose3Graph, orientations: orientations, ids: ids)
  }
}
