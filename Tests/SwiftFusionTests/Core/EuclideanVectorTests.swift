import Foundation
import XCTest

import SwiftFusion

/// Conform to this and call `runAllEuclideanVectorTests()` to test all the `EuclideanVector`
/// requirements on a concrete type.
///
/// Or call `runAllEuclideanVectorNTests()` to test all the `EuclideanVectorN` requirements.
protocol EuclideanVectorTests {
  /// The concrete type that we are testing.
  associatedtype Testee: EuclideanVector

  /// The dimension of the vector we are testing.
  static var dimension: Int { get }
}

extension EuclideanVectorTests {
  /// A set of basis vectors.
  fileprivate var basisVectors: [Testee] {
    return (0..<Self.dimension).map { index in
      var v = Array(repeating: Double(0), count: Self.dimension)
      v[index] = 1
      return Testee(Vector(v))
    }
  }

  /// Make a vector whose first element is `start` and whose subsequent elements increment by
  /// `stride`.
  fileprivate func makeVector(from start: Double, stride: Double) -> Testee {
    return Testee(Vector(Array((0..<Self.dimension).map { start + Double($0) * stride })))
  }

  /// Tests all `EuclideanVector` requirements.
  func runAllEuclideanVectorTests() {
    testEquality()
    testMove()
    testAddMutating()
    testAddFunctional()
    testSubtractMutating()
    testSubtractFunctional()
    testScalarProductMutating()
    testScalarProductFunctional()
    testDot()
    testNegate()
    testSquaredNorm()
    testNorm()
  }

  /// Tests ==.
  func testEquality() {
    let v1 = makeVector(from: 0, stride: 0)
    let v2 = makeVector(from: 1, stride: 0)
    XCTAssertTrue(v1 == v1)
    XCTAssertFalse(v1 == v2)
  }

  /// Tests move (exponential map).
  func testMove() {
    var v = makeVector(from: 0, stride: 1)
    let t = makeVector(from: 1, stride: 1)
    v.move(along: t)
    XCTAssertEqual(v, makeVector(from: 1, stride: 2))
  }

  /// Tests the value and derivative of the mutating +=.
  func testAddMutating() {
    func doMutatingAdd(_ v1: Testee, _ v2: Testee) -> Testee {
      var result = v1
      result += v2
      return result
    }

    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 2, stride: 1)
    XCTAssertEqual(doMutatingAdd(v1, v2), makeVector(from: 3, stride: 2))

    let (value, pb) = valueWithPullback(at: v1, v2, in: doMutatingAdd)
    XCTAssertEqual(value, makeVector(from: 3, stride: 2))
    for v in basisVectors {
      XCTAssertEqual(pb(v).0, v)
      XCTAssertEqual(pb(v).1, v)
    }
  }

  /// Tests the value and derivative of the functional +.
  func testAddFunctional() {
    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 2, stride: 1)
    XCTAssertEqual(v1 + v2, makeVector(from: 3, stride: 2))

    let (value, pb) = valueWithPullback(at: v1, v2, in: +)
    XCTAssertEqual(value, makeVector(from: 3, stride: 2))
    for v in basisVectors {
      XCTAssertEqual(pb(v).0, v)
      XCTAssertEqual(pb(v).1, v)
    }
  }

  /// Tests the value and derivative of the mutating -=.
  func testSubtractMutating() {
    func doMutatingSubtract(_ v1: Testee, _ v2: Testee) -> Testee {
      var result = v1
      result -= v2
      return result
    }

    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 1, stride: -1)
    XCTAssertEqual(doMutatingSubtract(v1, v2), makeVector(from: 0, stride: 2))

    let (value, pb) = valueWithPullback(at: v1, v2, in: doMutatingSubtract)
    XCTAssertEqual(value, makeVector(from: 0, stride: 2))
    for v in basisVectors {
      XCTAssertEqual(pb(v).0, v)
      XCTAssertEqual(pb(v).1, -v)
    }
  }

  /// Tests the value and derivative of the functional -.
  func testSubtractFunctional() {
    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 1, stride: -1)
    XCTAssertEqual(v1 - v2, makeVector(from: 0, stride: 2))

    let (value, pb) = valueWithPullback(at: v1, v2, in: -)
    XCTAssertEqual(value, makeVector(from: 0, stride: 2))
    for v in basisVectors {
      XCTAssertEqual(pb(v).0, v)
      XCTAssertEqual(pb(v).1, -v)
    }
  }

  /// Tests the value and derivative of the mutating *=.
  func testScalarProductMutating() {
    func doMutatingScalarProduct(_ s: Double, _ v1: Testee) -> Testee {
      var result = v1
      result *= s
      return result
    }

    let s = Double(2)
    let v1 = makeVector(from: 1, stride: 1)
    XCTAssertEqual(doMutatingScalarProduct(s, v1), makeVector(from: 2, stride: 2))

    let (value, pb) = valueWithPullback(at: s, v1, in: doMutatingScalarProduct)
    XCTAssertEqual(value, makeVector(from: 2, stride: 2))
    for (index, v) in basisVectors.enumerated() {
      XCTAssertEqual(pb(v).0, Double(index + 1))
      XCTAssertEqual(pb(v).1, 2 * v)
    }
  }

  /// Tests the value and derivative of the functional *.
  func testScalarProductFunctional() {
    let s = Double(2)
    let v1 = makeVector(from: 1, stride: 1)
    XCTAssertEqual(s * v1, makeVector(from: 2, stride: 2))

    let (value, pb) = valueWithPullback(at: s, v1, in: *)
    XCTAssertEqual(value, makeVector(from: 2, stride: 2))
    for (index, v) in basisVectors.enumerated() {
      XCTAssertEqual(pb(v).0, Double(index + 1))
      XCTAssertEqual(pb(v).1, 2 * v)
    }
  }

  /// Tests the value and derivative of `dot`.
  func testDot() {
    let v1 = makeVector(from: 1, stride: 1)
    let v2 = makeVector(from: 2, stride: 1)
    let expectedDot = (0..<Self.dimension).reduce(into: 0) { (r: inout Double, i: Int) in
      let x = Double(i) + 1
      let y = Double(i) + 2
      r += x * y
    }
    XCTAssertEqual(v1.dot(v2), expectedDot)

    let (value, g) = valueWithGradient(at: v1, v2) { $0.dot($1) }
    XCTAssertEqual(value, expectedDot)
    XCTAssertEqual(g.0, v2)
    XCTAssertEqual(g.1, v1)
  }

  /// Tests the value and derivative of the prefix unary -.
  func testNegate() {
    let v1 = makeVector(from: 1, stride: 1)
    XCTAssertEqual(-v1, makeVector(from: -1, stride: -1))

    let (value, pb) = valueWithPullback(at: v1) { -$0 }
    XCTAssertEqual(value, makeVector(from: -1, stride: -1))
    for v in basisVectors {
      XCTAssertEqual(pb(v), -v)
    }
  }

  /// Tests the value and derivative of `squaredNorm`.
  func testSquaredNorm() {
    let v1 = makeVector(from: 1, stride: 1)
    let expectedSquaredNorm = (0..<Self.dimension).reduce(into: 0) { (r: inout Double, i: Int) in
      let x = Double(i) + 1
      r += x * x
    }
    XCTAssertEqual(v1.squaredNorm, expectedSquaredNorm)

    let (value, g) = valueWithGradient(at: v1) { $0.squaredNorm }
    XCTAssertEqual(value, expectedSquaredNorm)
    XCTAssertEqual(g, 2 * v1)
  }

  /// Tests the value and derivative of `norm`.
  func testNorm() {
    let v1 = makeVector(from: 1, stride: 1)
    let expectedNorm = (0..<Self.dimension).reduce(into: 0) { (r: inout Double, i: Int) in
      let x = Double(i) + 1
      r += x * x
    }.squareRoot()
    XCTAssertEqual(v1.norm, expectedNorm)

    let (value, g) = valueWithGradient(at: v1) { $0.norm }
    XCTAssertEqual(value, expectedNorm)
    XCTAssertEqual(g, (1 / v1.norm) * v1)
  }
}

extension EuclideanVectorTests where Testee: EuclideanVectorN {
  /// Tests all `EuclideanVectorN` requirements, including those inherited from `EuclideanVector`.
  func runAllEuclideanVectorNTests() {
    runAllEuclideanVectorTests()
    testDimension()
    testStandardBasis()
  }

  /// Tests that the dimension is correct.
  func testDimension() {
    XCTAssertEqual(Testee.dimension, Self.dimension)
  }

  /// Tests that the standard basis is correct.
  func testStandardBasis() {
    let actualStandardBasis = Array(Testee.standardBasis)
    XCTAssertEqual(actualStandardBasis, self.basisVectors)
  }
}

/// Tests methods that involve multiple distinct euclidean vector types.
class MultipleEuclideanVectorTests: XCTestCase {
  /// Tests converting from one type to another type with the same number of elements.
  func testConversion() {
    let v = Vector9(0, 1, 2, 3, 4, 5, 6, 7, 8)
    let m = Matrix3(0, 1, 2, 3, 4, 5, 6, 7, 8)
    XCTAssertEqual(Vector9(m), v)

    let (value, pb) = valueWithPullback(at: m) { Vector9($0) }
    XCTAssertEqual(value, v)
    for (bV, bM) in zip(Vector9.standardBasis, Matrix3.standardBasis) {
      XCTAssertEqual(pb(bV), bM)
    }
  }

  /// Tests concatenating two vectors.
  func testConcatenate() {
    let v1 = Vector2(0, 1)
    let v2 = Vector3(2, 3, 4)
    let expected = Vector5(0, 1, 2, 3, 4)
    XCTAssertEqual(Vector5(concatenating: v1, v2), expected)

    let (value, pb) = valueWithPullback(at: v1, v2) { Vector5(concatenating: $0, $1) }
    XCTAssertEqual(value, expected)

    XCTAssertEqual(pb(Vector5(1, 0, 0, 0, 0)).0, Vector2(1, 0))
    XCTAssertEqual(pb(Vector5(0, 1, 0, 0, 0)).0, Vector2(0, 1))
    XCTAssertEqual(pb(Vector5(0, 0, 1, 0, 0)).0, Vector2(0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 1, 0)).0, Vector2(0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 0, 1)).0, Vector2(0, 0))

    XCTAssertEqual(pb(Vector5(1, 0, 0, 0, 0)).1, Vector3(0, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 1, 0, 0, 0)).1, Vector3(0, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 1, 0, 0)).1, Vector3(1, 0, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 1, 0)).1, Vector3(0, 1, 0))
    XCTAssertEqual(pb(Vector5(0, 0, 0, 0, 1)).1, Vector3(0, 0, 1))
  }
}