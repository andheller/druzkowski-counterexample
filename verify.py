#!/usr/bin/env python3
"""Regenerate and exactly check the published sparse counterexample."""

from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import generate


HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_matrix(path: Path):
    dimension = None
    rows = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith("# dimension"):
                dimension = int(line.split("\t", 1)[1])
                rows = [{} for _ in range(dimension)]
                continue
            if line.startswith("#"):
                continue
            row, column, value = map(int, line.split("\t"))
            rows[row - 1][column - 1] = value
    if dimension is None or rows is None:
        raise ValueError("matrix dimension header is missing")
    return rows


def matvec(rows, vector):
    return [
        sum(Fraction(coefficient) * vector[column] for column, coefficient in row.items())
        for row in rows
    ]


def druzkowski_map(rows, point):
    product = matvec(rows, point)
    return [coordinate + linear_value**3 for coordinate, linear_value in zip(point, product)]


def sparse_square(rows):
    result = []
    for row in rows:
        output = {}
        for middle, first in row.items():
            for column, second in rows[middle].items():
                output[column] = output.get(column, 0) + first * second
        result.append({column: value for column, value in output.items() if value})
    return result


def sympy_sparse_rows(matrix):
    return [
        [
            [column + 1, generate.qstr(matrix[row, column])]
            for column in range(matrix.cols)
            if matrix[row, column] != 0
        ]
        for row in range(matrix.rows)
    ]


def main():
    # Byte-level integrity of all generated artifacts.
    with (HERE / "SHA256SUMS").open(encoding="utf-8") as handle:
        for line in handle:
            expected, filename = line.strip().split("  ", 1)
            actual = digest(HERE / filename)
            assert actual == expected, f"hash mismatch for {filename}"

    matrix_rows = read_matrix(HERE / "A_integer.coo.tsv")
    with (HERE / "points.json").open(encoding="utf-8") as handle:
        points_payload = json.load(handle)
    points = {
        name: [Fraction(value) for value in coordinates]
        for name, coordinates in points_payload["preimages"].items()
    }
    common_image = [Fraction(value) for value in points_payload["common_image"]]

    assert len(matrix_rows) == points_payload["dimension"] == 212
    assert len({tuple(point) for point in points.values()}) == 3
    for name, point in points.items():
        assert druzkowski_map(matrix_rows, point) == common_image, name
    assert sparse_square(matrix_rows) == [{} for _ in matrix_rows]
    print("direct check: three distinct rational points map to the recorded common image")
    print("direct matrix check: A^2=0")

    # Rebuild the structural certificate from the seed map.  This checks the
    # determinant-one seed, all stable reductions, cubic decomposition, A=D*B,
    # scalar integralization, and exact collision, then compares every entry.
    rebuilt = generate.build_counterexample(verbose=False)
    rebuilt_rows = [
        {column: int(value) for column, value in row.items()}
        for row in rebuilt["integer_rows"]
    ]
    assert rebuilt_rows == matrix_rows
    rebuilt_points = {
        name: [Fraction(int(value.p), int(value.q)) for value in point]
        for name, point in zip(["u", "v", "w"], rebuilt["integer_preimages"])
    }
    assert rebuilt_points == points
    rebuilt_image = [
        Fraction(int(value.p), int(value.q)) for value in rebuilt["integer_image"]
    ]
    assert rebuilt_image == common_image
    assert rebuilt["stats"]["rank_A"] == 32
    assert rebuilt["stats"]["rank_BD"] == 0
    assert rebuilt["stats"]["nilpotency_index_A"] == 2
    assert rebuilt["stats"]["A_squared_zero"]
    assert rebuilt["stats"]["A_cubed_zero"]
    assert not rebuilt["stats"]["A_squared_nonzero"]

    with (HERE / "block_reduction.json").open(encoding="utf-8") as handle:
        block_payload = json.load(handle)
    assert block_payload["step_count"] == len(rebuilt["reduction_steps"]) == 10
    assert block_payload["reduced_dimension"] == rebuilt["stats"]["reduced_dimension"] == 23
    assert block_payload["steps"] == rebuilt["reduction_steps"]
    normalization = block_payload["linear_normalization"]
    assert normalization["L_determinant"] == "1"
    assert normalization["L_rows"] == sympy_sparse_rows(rebuilt["reduced_linear_part"])
    assert normalization["L_inverse_rows"] == sympy_sparse_rows(
        rebuilt["reduced_linear_part_inverse"]
    )
    assert block_payload["normalized_collision_points"] == [
        [generate.qstr(value) for value in point]
        for point in rebuilt["reduced_points"]
    ]
    assert block_payload["normalized_common_target"] == [
        generate.qstr(value) for value in rebuilt["reduced_target"]
    ]

    with (HERE / "basis_reduction.json").open(encoding="utf-8") as handle:
        basis_payload = json.load(handle)
    basis = rebuilt["basis_reduction"]
    assert basis_payload["prebasis_count"] == len(basis["prebasis_columns"]) == 221
    assert basis_payload["basis_count"] == basis["rank"] == 211
    assert basis_payload["prebasis_D0_rows"] == [
        [[index + 1, generate.qstr(coefficient)] for index, coefficient in form]
        for form, _ in basis["prebasis_columns"]
    ]
    assert basis_payload["prebasis_B0_columns"] == [
        [
            [coordinate + 1, generate.qstr(coefficient)]
            for coordinate, coefficient in sorted(output_vector.items())
        ]
        for _, output_vector in basis["prebasis_columns"]
    ]
    assert basis_payload["pivot_columns"] == [value + 1 for value in basis["pivot_columns"]]
    expected_relations = [
        {
            "old_column": relation["old_column"] + 1,
            "basis_coefficients": [
                [basis_row + 1, generate.qstr(coefficient)]
                for basis_row, coefficient in relation["coefficients"]
            ],
        }
        for relation in basis["nonpivot_relations"]
    ]
    assert basis_payload["nonpivot_relations"] == expected_relations
    assert len(expected_relations) == 10

    with (HERE / "pairing.json").open(encoding="utf-8") as handle:
        pairing_payload = json.load(handle)
    assert pairing_payload["homogeneous_dimension"] == rebuilt["stats"][
        "homogeneous_dimension"
    ]
    assert pairing_payload["number_of_cube_forms"] == rebuilt["stats"]["cube_forms"]
    assert pairing_payload["D0_rows"] == [
        [[index + 1, generate.qstr(coefficient)] for index, coefficient in form]
        for form, _ in rebuilt["columns"]
    ]
    assert pairing_payload["B0_columns"] == [
        [
            [coordinate + 1, generate.qstr(coefficient)]
            for coordinate, coefficient in sorted(output_vector.items())
        ]
        for _, output_vector in rebuilt["columns"]
    ]
    assert pairing_payload["common_target"] == [
        generate.qstr(value) for value in rebuilt["homogeneous_target"]
    ]

    with (HERE / "certificate.json").open(encoding="utf-8") as handle:
        certificate = json.load(handle)
    assert certificate["stats"] == rebuilt["stats"]
    print(
        "structural check: det J(D_A) = 1 via BCW homogenization and "
        "the A=D*B determinant identity"
    )
    print(
        f"matrix check: {len(matrix_rows)}x{len(matrix_rows)}, "
        f"nnz={sum(len(row) for row in matrix_rows)}, "
        f"max|a_ij|={max(abs(value) for row in matrix_rows for value in row.values())}"
    )
    print("block check: 10 determinant-one reductions and linear normalization verified")
    print("basis check: 221 merged scalar cubes have exact rank 211")
    print("structure check: rank(A)=32 and A^2=0")
    print("all exact checks passed")


if __name__ == "__main__":
    main()
