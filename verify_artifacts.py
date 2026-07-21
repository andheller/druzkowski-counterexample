#!/usr/bin/env python3
"""Verify the published counterexample without importing the generator.

This verifier treats the TSV/JSON files as its inputs.  It independently
replays the saved block reductions, reconstructs the homogeneous map, checks
both cube decompositions, rebuilds A from B and D, and evaluates the published
collision.  All arithmetic is exact.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent


def load_json(name: str):
    with (HERE / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def rational(value) -> sp.Rational:
    return sp.Rational(str(value))


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def verify_hashes() -> None:
    with (HERE / "SHA256SUMS").open(encoding="utf-8") as handle:
        for line in handle:
            expected, filename = line.strip().split("  ", 1)
            assert digest(HERE / filename) == expected, filename


def read_matrix(name: str) -> list[dict[int, sp.Integer]]:
    dimension = None
    rows = None
    with (HERE / name).open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith("# dimension"):
                dimension = int(line.split("\t", 1)[1])
                rows = [{} for _ in range(dimension)]
                continue
            if line.startswith("#"):
                continue
            row, column, value = map(int, line.split("\t"))
            assert rows is not None
            rows[row - 1][column - 1] = sp.Integer(value)
    assert dimension is not None and rows is not None
    return rows


def sparse_payload_rows(payload, width: int) -> sp.MutableSparseMatrix:
    entries = {}
    for row, sparse_row in enumerate(payload):
        for column, coefficient in sparse_row:
            entries[row, column - 1] = rational(coefficient)
    return sp.SparseMatrix(len(payload), width, entries)


def sparse_payload_columns(payload, height: int) -> sp.MutableSparseMatrix:
    entries = {}
    for column, sparse_column in enumerate(payload):
        for row, coefficient in sparse_column:
            entries[row - 1, column] = rational(coefficient)
    return sp.SparseMatrix(height, len(payload), entries)


def polynomial_from_terms(terms, variables):
    result = sp.Integer(0)
    for term in terms:
        monomial = rational(term["coefficient"])
        for index, exponent in term["monomial"]:
            monomial *= variables[index - 1] ** exponent
        result += monomial
    return sp.expand(result)


def vector_at(expressions, variables, point):
    substitution = dict(zip(variables, point))
    return [sp.cancel(expression.subs(substitution)) for expression in expressions]


def assert_polynomial_vectors_equal(left, right) -> None:
    assert len(left) == len(right)
    for index, (first, second) in enumerate(zip(left, right), start=1):
        assert sp.expand(first - second) == 0, f"polynomial coordinate {index}"


def split_degrees(expression, variables):
    pieces = {0: sp.Integer(0), 1: sp.Integer(0), 2: sp.Integer(0), 3: sp.Integer(0)}
    polynomial = sp.Poly(sp.expand(expression), *variables)
    for exponents, coefficient in polynomial.terms():
        degree = sum(exponents)
        assert degree in pieces, f"unexpected degree {degree}"
        monomial = coefficient * sp.prod(
            variable**exponent for variable, exponent in zip(variables, exponents)
        )
        pieces[degree] += monomial
    return {degree: sp.expand(piece) for degree, piece in pieces.items()}


def replay_reduction(certificate, blocks):
    variables = list(sp.symbols("x0:3"))
    namespace = {str(variable): variable for variable in variables}
    current = [sp.sympify(text, locals=namespace) for text in certificate["seed_map"]]
    points = [[rational(value) for value in point] for point in certificate["seed_points"]]
    target = [rational(value) for value in certificate["seed_target"]]

    assert sp.expand(sp.Matrix(current).jacobian(variables).det() - 1) == 0
    assert len({tuple(point) for point in points}) == 3
    for point in points:
        assert vector_at(current, variables, point) == target

    for expected_move, step in enumerate(blocks["steps"], start=1):
        assert step["move"] == expected_move
        assert step["input_dimension"] == len(variables)
        assert step["output_dimension"] == len(variables) + 2
        assert step["source_automorphism_jacobian"] == "1"
        assert step["target_automorphism_jacobian"] == "1"

        p = polynomial_from_terms(step["P_terms"], variables)
        q = polynomial_from_terms(step["Q_terms"], variables)
        expanded_product = sp.sympify(
            step["expanded_product"],
            locals={str(variable): variable for variable in variables},
        )
        assert sp.expand(p * q - expanded_product) == 0

        r, s = sp.symbols(f"x{len(variables)}:{len(variables) + 2}")
        coordinate = step["coordinate"] - 1
        next_map = list(current)
        next_map[coordinate] = sp.expand(next_map[coordinate] - (r + p) * (s + q))
        next_map.extend([sp.expand(r + p), sp.expand(s + q)])

        lifted_points = []
        for point in points:
            substitution = dict(zip(variables, point))
            lifted_points.append(
                point
                + [
                    -sp.cancel(p.subs(substitution)),
                    -sp.cancel(q.subs(substitution)),
                ]
            )
        recorded = [
            [rational(value) for value in point]
            for point in step["lifted_collision_points"]
        ]
        assert lifted_points == recorded

        variables.extend([r, s])
        target.extend([sp.Integer(0), sp.Integer(0)])
        for point in lifted_points:
            assert vector_at(next_map, variables, point) == target
        current, points = next_map, lifted_points

    assert len(variables) == blocks["reduced_dimension"] == 23
    assert max(sp.Poly(expression, *variables).total_degree() for expression in current) <= 3

    linear_part = sp.Matrix(current).jacobian(variables).subs(
        {variable: 0 for variable in variables}
    )
    recorded_linear = sparse_payload_rows(blocks["linear_normalization"]["L_rows"], 23)
    recorded_inverse = sparse_payload_rows(
        blocks["linear_normalization"]["L_inverse_rows"], 23
    )
    assert linear_part == recorded_linear
    assert linear_part.det() == 1
    assert recorded_inverse * linear_part == sp.eye(23)

    normalized = [
        sp.expand(value)
        for value in list(recorded_inverse * sp.Matrix(current))
    ]
    assert sp.Matrix(normalized).jacobian(variables).subs(
        {variable: 0 for variable in variables}
    ) == sp.eye(23)

    normalized_target = list(recorded_inverse * sp.Matrix(target))
    recorded_target = [rational(value) for value in blocks["normalized_common_target"]]
    assert normalized_target == recorded_target
    recorded_points = [
        [rational(value) for value in point]
        for point in blocks["normalized_collision_points"]
    ]
    assert points == recorded_points
    for point in points:
        assert vector_at(normalized, variables, point) == normalized_target

    quadratic = []
    cubic = []
    for coordinate, expression in enumerate(normalized):
        pieces = split_degrees(expression, variables)
        assert pieces[0] == 0
        assert sp.expand(pieces[1] - variables[coordinate]) == 0
        quadratic.append(pieces[2])
        cubic.append(pieces[3])
    return variables, normalized, quadratic, cubic, points, normalized_target


def homogeneous_lift(quadratic, cubic):
    z = list(sp.symbols("z0:47"))
    substitution = {sp.Symbol(f"x{i}"): z[i] for i in range(23)}
    qz = [sp.expand(value.subs(substitution)) for value in quadratic]
    cz = [sp.expand(value.subs(substitution)) for value in cubic]
    t = z[46]
    h = [sp.expand(t * qz[i] - t**2 * z[23 + i]) for i in range(23)]
    h.extend(cz)
    h.append(sp.Integer(0))
    for expression in h:
        if expression:
            assert sp.Poly(expression, *z).is_homogeneous
            assert sp.Poly(expression, *z).total_degree() == 3
    return z, h, cz


def cube_decomposition(z, h, pairing, basis):
    d0 = sparse_payload_rows(pairing["D0_rows"], 47)
    b0 = sparse_payload_columns(pairing["B0_columns"], 47)
    assert d0.shape == (211, 47)
    assert b0.shape == (47, 211)

    forms = list(d0 * sp.Matrix(z))
    reconstructed = list(b0 * sp.Matrix([sp.expand(form**3) for form in forms]))
    assert_polynomial_vectors_equal(reconstructed, h)

    pre_d0 = sparse_payload_rows(basis["prebasis_D0_rows"], 47)
    pre_b0 = sparse_payload_columns(basis["prebasis_B0_columns"], 47)
    assert pre_d0.shape == (221, 47)
    assert pre_b0.shape == (47, 221)
    pre_forms = list(pre_d0 * sp.Matrix(z))
    pre_cubes = [sp.expand(form**3) for form in pre_forms]
    pre_reconstructed = list(pre_b0 * sp.Matrix(pre_cubes))
    assert_polynomial_vectors_equal(pre_reconstructed, h)

    pivots = [column - 1 for column in basis["pivot_columns"]]
    assert len(pivots) == len(set(pivots)) == basis["basis_count"] == 211
    assert d0 == pre_d0[pivots, :]
    for relation in basis["nonpivot_relations"]:
        old_cube = pre_cubes[relation["old_column"] - 1]
        combination = sp.Integer(0)
        for basis_row, coefficient in relation["basis_coefficients"]:
            combination += rational(coefficient) * pre_cubes[pivots[basis_row - 1]]
        assert sp.expand(old_cube - combination) == 0
    assert len(basis["nonpivot_relations"]) == 10
    return b0, d0


def check_pairing_and_matrix(z, h, b0, d0, pairing, matrix_rows):
    tau = sp.Matrix([rational(value) for value in pairing["common_target"]])
    b = b0.row_join(tau)
    d = d0.col_join(sp.zeros(1, 47))
    assert b.shape == (47, 212)
    assert d.shape == (212, 47)
    assert b * d == sp.zeros(47, 47)

    a0 = d * b
    published = sp.SparseMatrix(
        212,
        212,
        {
            (row, column): value
            for row, entries in enumerate(matrix_rows)
            for column, value in entries.items()
        },
    )
    assert 144 * a0 == published
    assert published * published == sp.zeros(212, 212)
    assert d.rank() == 47
    assert b.rank() == published.rank() == 32

    # The two determinant-preserving identities used by the certificate are
    # now exact matrix identities over Q:
    #   JG = I + 3 diag((DBW)^2) DB,
    #   det(I+UV) = det(I+VU) = det JPhi(BW).
    # The preceding replay proved det JPhi=1 from the seed via determinant-one
    # stabilizations and the displayed homogenization.
    assert_polynomial_vectors_equal(
        [z[i] + h[i] for i in range(47)],
        list(sp.Matrix(z) + b0 * sp.Matrix([(d0 * sp.Matrix(z))[i] ** 3 for i in range(211)])),
    )


def check_homogeneous_collision(z, h, cubic, reduced_points, reduced_target, pairing):
    tau = [rational(value) for value in pairing["common_target"]]
    lifted = []
    for point in reduced_points:
        substitution = {sp.Symbol(f"x{i}"): point[i] for i in range(23)}
        y = [-sp.cancel(expression.subs(substitution)) for expression in cubic]
        source = point + y + [sp.Integer(1)]
        lifted.append(source)
        assert vector_at([z[i] + h[i] for i in range(47)], z, source) == tau
    expected_tau = reduced_target + [sp.Integer(0)] * 23 + [sp.Integer(1)]
    assert tau == expected_tau
    assert len({tuple(point) for point in lifted}) == 3


def matvec(rows, vector):
    return [
        sum((coefficient * vector[column] for column, coefficient in row.items()), sp.Integer(0))
        for row in rows
    ]


def check_published_collision(matrix_rows, points_payload):
    points = {
        name: [rational(value) for value in coordinates]
        for name, coordinates in points_payload["preimages"].items()
    }
    image = [rational(value) for value in points_payload["common_image"]]
    assert len({tuple(point) for point in points.values()}) == 3
    for name, point in points.items():
        linear = matvec(matrix_rows, point)
        actual = [coordinate + value**3 for coordinate, value in zip(point, linear)]
        assert actual == image, name


def main() -> None:
    verify_hashes()
    certificate = load_json("certificate.json")
    blocks = load_json("block_reduction.json")
    pairing = load_json("pairing.json")
    basis = load_json("basis_reduction.json")
    points_payload = load_json("points.json")
    matrix_rows = read_matrix("A_integer.coo.tsv")

    variables, _, quadratic, cubic, reduced_points, reduced_target = replay_reduction(
        certificate, blocks
    )
    assert len(variables) == 23
    z, h, _ = homogeneous_lift(quadratic, cubic)
    b0, d0 = cube_decomposition(z, h, pairing, basis)
    check_homogeneous_collision(
        z, h, cubic, reduced_points, reduced_target, pairing
    )
    check_pairing_and_matrix(z, h, b0, d0, pairing, matrix_rows)
    check_published_collision(matrix_rows, points_payload)

    print("PASS: seed determinant and triple collision")
    print("PASS: 10 determinant-one stable reductions and 23D normalization")
    print("PASS: 47D cubic homogenization and both saved cube decompositions")
    print("PASS: A = 144*D*B, rank(A)=32, and A^2=0")
    print("PASS: three distinct 212D rational points have the saved common image")
    print("all artifact-first exact checks passed")


if __name__ == "__main__":
    main()
