#!/usr/bin/env python3
"""Generate and exactly verify an explicit 212-dimensional Druzkowski counterexample.

The construction starts with the July 2026 three-variable Keller map, performs
the Bass--Connell--Wright degree reduction, homogenizes to a cubic map, and
then applies a collision-tailored Gorni--Zampieri pairing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from math import lcm
from pathlib import Path

import sympy as sp


HERE = Path(__file__).resolve().parent


def qstr(value: sp.Expr) -> str:
    value = sp.Rational(sp.cancel(value))
    return str(value.p) if value.q == 1 else f"{value.p}/{value.q}"


def nonzero_terms(expr: sp.Expr) -> list[sp.Expr]:
    if expr == 0:
        return []
    return list(sp.Add.make_args(sp.expand(expr)))


def term_data(term: sp.Expr, variables: list[sp.Symbol]):
    powers = term.as_powers_dict()
    exponents = [int(powers.get(variable, 0)) for variable in variables]
    monomial = sp.prod(
        variable**exponent for variable, exponent in zip(variables, exponents)
    )
    coefficient = sp.cancel(term / monomial)
    if coefficient.free_symbols:
        raise ValueError(f"coefficient is not scalar: {term}")
    return sp.Rational(coefficient), exponents


def evaluate(expr: sp.Expr, variables, point):
    return sp.cancel(expr.subs(dict(zip(variables, point))))


def evaluate_map(polynomials, variables, point):
    return [evaluate(polynomial, variables, point) for polynomial in polynomials]


FIXED_REDUCTION_SCHEDULE = [
    # (zero-based output coordinate, terms of P, terms of Q).
    # A term is (coefficient, {variable_index: exponent}).  The first two
    # moves cancel whole factored high-degree blocks, which is why this
    # schedule needs only ten moves rather than eighteen monomial moves.
    (
        2,
        [(sp.Rational(1), {0: 1, 1: 2})],
        [
            (sp.Rational(1), {0: 2, 1: 1, 2: 1}),
            (sp.Rational(3), {0: 1, 1: 2}),
            (sp.Rational(3), {0: 1, 2: 1}),
            (sp.Rational(7), {1: 1}),
        ],
    ),
    (
        1,
        [(sp.Rational(3), {0: 2, 1: 1})],
        [
            (sp.Rational(1), {0: 1, 1: 1, 2: 1}),
            (sp.Rational(3), {1: 2}),
            (sp.Rational(2), {2: 1}),
        ],
    ),
    (0, [(sp.Rational(-1, 2), {0: 2})], [(sp.Rational(1), {0: 1, 2: 1})]),
    (2, [(sp.Rational(-1), {0: 2})], [(sp.Rational(1), {1: 1, 2: 1, 3: 1})]),
    (1, [(sp.Rational(-3), {0: 2})], [(sp.Rational(1), {1: 1, 6: 1})]),
    (1, [(sp.Rational(-1), {0: 1, 1: 1})], [(sp.Rational(1), {2: 1, 5: 1})]),
    (2, [(sp.Rational(-1), {0: 1, 1: 1})], [(sp.Rational(1), {1: 1, 4: 1})]),
    (2, [(sp.Rational(-3), {0: 1, 1: 1})], [(sp.Rational(1), {1: 1, 3: 1})]),
    (2, [(sp.Rational(-1), {1: 1, 2: 1})], [(sp.Rational(1), {3: 1, 9: 1})]),
    (4, [(sp.Rational(1), {0: 2})], [(sp.Rational(1), {1: 1, 2: 1})]),
]


def schedule_polynomial(terms, variables):
    return sp.expand(
        sum(
            coefficient
            * sp.prod(
                variables[index] ** exponent for index, exponent in powers.items()
            )
            for coefficient, powers in terms
        )
    )


def reduce_to_cubic(polynomials, variables, points):
    """Apply the fixed, audited 10-step determinant-one block reduction."""
    steps = []
    for move, (coordinate, p_terms, q_terms) in enumerate(
        FIXED_REDUCTION_SCHEDULE, start=1
    ):
        p_term = schedule_polynomial(p_terms, variables)
        q_term = schedule_polynomial(q_terms, variables)
        product = sp.expand(p_term * q_term)
        current_poly = sp.Poly(polynomials[coordinate], *variables)
        for product_term in nonzero_terms(product):
            product_coefficient, product_exponents = term_data(product_term, variables)
            monomial = sp.prod(
                variable**exponent
                for variable, exponent in zip(variables, product_exponents)
            )
            assert current_poly.coeff_monomial(monomial) == product_coefficient, (
                move,
                coordinate,
                product_term,
            )
        degree = sp.Poly(product, *variables).total_degree()
        old_dimension = len(variables)
        r_var, s_var = sp.symbols(f"x{old_dimension} x{old_dimension + 1}")

        # Explicit elementary source and target automorphisms. Their Jacobians
        # are triangular with diagonal one; assert that fact symbolically for
        # every concrete move.
        all_variables = list(variables) + [r_var, s_var]
        source_automorphism = list(variables) + [r_var + p_term, s_var + q_term]
        assert sp.factor(sp.Matrix(source_automorphism).jacobian(all_variables).det()) == 1
        target_variables = list(sp.symbols(f"u0:{old_dimension + 2}"))
        target_automorphism = list(target_variables)
        target_automorphism[coordinate] -= target_variables[-2] * target_variables[-1]
        assert sp.factor(
            sp.Matrix(target_automorphism).jacobian(target_variables).det()
        ) == 1

        lifted_points = []
        for point in points:
            lifted_points.append(
                point
                + [
                    -evaluate(p_term, variables, point),
                    -evaluate(q_term, variables, point),
                ]
            )

        polynomials = list(polynomials)
        polynomials[coordinate] = sp.expand(
            polynomials[coordinate] - (r_var + p_term) * (s_var + q_term)
        )
        polynomials.extend([r_var + p_term, s_var + q_term])
        variables = list(variables) + [r_var, s_var]
        points = lifted_points
        steps.append(
            {
                "move": move,
                "degree": degree,
                "coordinate": coordinate + 1,
                "input_dimension": old_dimension,
                "output_dimension": old_dimension + 2,
                "P": str(p_term),
                "Q": str(q_term),
                "P_terms": [
                    {
                        "coefficient": qstr(coefficient),
                        "monomial": [
                            [index + 1, exponent]
                            for index, exponent in sorted(powers.items())
                        ],
                    }
                    for coefficient, powers in p_terms
                ],
                "Q_terms": [
                    {
                        "coefficient": qstr(coefficient),
                        "monomial": [
                            [index + 1, exponent]
                            for index, exponent in sorted(powers.items())
                        ],
                    }
                    for coefficient, powers in q_terms
                ],
                "expanded_product": str(product),
                "source_automorphism_jacobian": "1",
                "target_automorphism_jacobian": "1",
                "lifted_collision_points": [
                    [qstr(value) for value in point] for point in lifted_points
                ],
            }
        )
    assert all(sp.Poly(polynomial, *variables).total_degree() <= 3 for polynomial in polynomials)
    return polynomials, variables, points, steps


def canonical_form(form):
    form = {index: sp.Rational(value) for index, value in form.items() if value}
    first = min(form)
    sign = -1 if form[first] < 0 else 1
    key = tuple(
        sorted((index, sp.Rational(sign) * value) for index, value in form.items())
    )
    # original form = sign * canonical form
    return key, sp.Rational(sign)


def linear_value(form, point):
    return sp.cancel(sum(coefficient * point[index] for index, coefficient in form))


def sparse_matvec(rows, vector):
    return [
        sp.cancel(sum(coefficient * vector[column] for column, coefficient in row.items()))
        for row in rows
    ]


def scalar_cube_coefficients(form):
    """Return the exact scalar monomial expansion of one sparse linear-form cube."""
    result = defaultdict(lambda: sp.Rational(0))
    entries = list(form)
    for first_index, first_coefficient in entries:
        for second_index, second_coefficient in entries:
            for third_index, third_coefficient in entries:
                counts = defaultdict(int)
                counts[first_index] += 1
                counts[second_index] += 1
                counts[third_index] += 1
                monomial = tuple(sorted(counts.items()))
                result[monomial] += (
                    first_coefficient * second_coefficient * third_coefficient
                )
    return {
        monomial: sp.cancel(coefficient)
        for monomial, coefficient in result.items()
        if sp.cancel(coefficient) != 0
    }


def basis_reduce_cube_columns(columns):
    """Replace dependent scalar cubes by an exact RREF-selected cube basis.

    Returns the reduced columns and a complete certificate containing the
    pre-basis columns, pivot indices, and each eliminated cube relation.
    """
    expansions = [scalar_cube_coefficients(form) for form, _ in columns]
    monomials = sorted({monomial for expansion in expansions for monomial in expansion})
    row_index = {monomial: index for index, monomial in enumerate(monomials)}
    matrix = sp.MutableSparseMatrix(
        len(monomials),
        len(columns),
        {
            (row_index[monomial], column): coefficient
            for column, expansion in enumerate(expansions)
            for monomial, coefficient in expansion.items()
        },
    )
    rref, pivots = matrix.rref()
    rank = len(pivots)

    reduced = []
    for basis_row, pivot_column in enumerate(pivots):
        output_vector = defaultdict(lambda: sp.Rational(0))
        for old_column, (_, old_output) in enumerate(columns):
            scalar = rref[basis_row, old_column]
            if scalar:
                for coordinate, coefficient in old_output.items():
                    output_vector[coordinate] += scalar * coefficient
        reduced.append(
            (
                columns[pivot_column][0],
                {
                    coordinate: sp.cancel(coefficient)
                    for coordinate, coefficient in output_vector.items()
                    if sp.cancel(coefficient) != 0
                },
            )
        )

    pivot_position = {column: index for index, column in enumerate(pivots)}
    nonpivot_relations = []
    for old_column in range(len(columns)):
        if old_column in pivot_position:
            continue
        nonpivot_relations.append(
            {
                "old_column": old_column,
                "coefficients": [
                    (basis_row, sp.cancel(rref[basis_row, old_column]))
                    for basis_row in range(rank)
                    if rref[basis_row, old_column]
                ],
            }
        )
    certificate = {
        "prebasis_columns": columns,
        "pivot_columns": list(pivots),
        "nonpivot_relations": nonpivot_relations,
        "scalar_cube_support_monomials": len(monomials),
        "rank": rank,
    }
    return reduced, certificate


def build_counterexample(verbose=False):
    def log(*items):
        if verbose:
            print(*items)

    # Rational tangent-to-identity normalization of the announced map:
    # (F_3/2,F_2,F_1).  This has determinant one and keeps the short
    # three-point rational collision.
    x, y, z = sp.symbols("x0 x1 x2")
    variables = [x, y, z]
    u = 1 + x * y
    seed = [
        sp.expand(x - sp.Rational(3, 2) * x**2 * y - sp.Rational(1, 2) * x**3 * z),
        sp.expand(y + 3 * x * u**2 * z + 3 * x * y**2 * (4 + 3 * x * y)),
        sp.expand(u**3 * z + y**2 * u * (4 + 3 * x * y)),
    ]
    original_points = [
        [sp.Rational(0), sp.Rational(0), sp.Rational(-1, 4)],
        [sp.Rational(1), sp.Rational(-3, 2), sp.Rational(13, 2)],
        [sp.Rational(-1), sp.Rational(3, 2), sp.Rational(13, 2)],
    ]
    original_target = [sp.Rational(0), sp.Rational(0), sp.Rational(-1, 4)]

    seed_det = sp.factor(sp.Matrix(seed).jacobian(variables).det())
    assert seed_det == 1
    assert all(
        evaluate_map(seed, variables, point) == original_target
        for point in original_points
    )
    log("seed: det=1 and three-point collision verified")

    reduced, reduced_variables, reduced_points, reduction_steps = reduce_to_cubic(
        seed, variables, original_points
    )
    reduced_dimension = len(reduced_variables)
    assert reduced_dimension == 23
    assert len(reduction_steps) == 10

    # Polynomial block factors may have linear terms. Normalize the resulting
    # cubic map by its exact linear part. Its determinant is one, so this output
    # linear change preserves the determinant identity and all source collisions.
    origin = {variable: 0 for variable in reduced_variables}
    linear_part = sp.Matrix(reduced).jacobian(reduced_variables).subs(origin)
    assert linear_part.det() == 1
    linear_part_inverse = linear_part.inv()
    assert linear_part_inverse * linear_part == sp.eye(reduced_dimension)
    reduced = [sp.expand(value) for value in linear_part_inverse * sp.Matrix(reduced)]
    assert sp.Matrix(reduced).jacobian(reduced_variables).subs(origin) == sp.eye(
        reduced_dimension
    )
    reduced_values = [
        evaluate_map(reduced, reduced_variables, point) for point in reduced_points
    ]
    assert all(value == reduced_values[0] for value in reduced_values)
    assert all(
        sp.Poly(polynomial, *reduced_variables).total_degree() <= 3
        for polynomial in reduced
    )
    log("degree reduction: 10 fixed block steps, normalization, dimension 23")

    quadratic = []
    cubic = []
    for polynomial, variable in zip(reduced, reduced_variables):
        q_part = sp.Rational(0)
        c_part = sp.Rational(0)
        for term in nonzero_terms(sp.expand(polynomial - variable)):
            _, exponents = term_data(term, reduced_variables)
            degree = sum(exponents)
            if degree == 2:
                q_part += term
            elif degree == 3:
                c_part += term
            else:
                raise AssertionError(f"unexpected degree {degree}: {term}")
        quadratic.append(sp.expand(q_part))
        cubic.append(sp.expand(c_part))

    # Phi(X,Y,T)=(X+T*R2(X)-T^2*Y, Y+R3(X), T)=Id+H.
    homogeneous_dimension = 2 * reduced_dimension + 1
    homogenizer = 2 * reduced_dimension
    homogeneous_terms = []  # (output coordinate, coefficient, exponent dictionary)
    for coordinate in range(reduced_dimension):
        for term in nonzero_terms(quadratic[coordinate]):
            coefficient, exponents = term_data(term, reduced_variables)
            exponent_dict = {
                index: exponent for index, exponent in enumerate(exponents) if exponent
            }
            exponent_dict[homogenizer] = 1
            homogeneous_terms.append((coordinate, coefficient, exponent_dict))
        homogeneous_terms.append(
            (coordinate, sp.Rational(-1), {reduced_dimension + coordinate: 1, homogenizer: 2})
        )
        for term in nonzero_terms(cubic[coordinate]):
            coefficient, exponents = term_data(term, reduced_variables)
            exponent_dict = {
                index: exponent for index, exponent in enumerate(exponents) if exponent
            }
            homogeneous_terms.append(
                (reduced_dimension + coordinate, coefficient, exponent_dict)
            )
    assert len(homogeneous_terms) == 102
    assert all(sum(exponents.values()) == 3 for _, _, exponents in homogeneous_terms)

    homogeneous_sources = []
    for point in reduced_points:
        cubic_value = [evaluate(part, reduced_variables, point) for part in cubic]
        homogeneous_sources.append(point + [-value for value in cubic_value] + [sp.Rational(1)])

    def homogeneous_h(point):
        value = [sp.Rational(0)] * homogeneous_dimension
        for coordinate, coefficient, exponents in homogeneous_terms:
            value[coordinate] += coefficient * sp.prod(
                point[index] ** exponent for index, exponent in exponents.items()
            )
        return [sp.cancel(entry) for entry in value]

    homogeneous_images = []
    for point in homogeneous_sources:
        h_value = homogeneous_h(point)
        homogeneous_images.append(
            [sp.cancel(point[index] + h_value[index]) for index in range(homogeneous_dimension)]
        )
    target = homogeneous_images[0]
    assert all(image == target for image in homogeneous_images)
    log("homogenization: cubic homogeneous Keller collision in dimension 47")

    # Check the two polarization identities used below once, symbolically.
    aa, bb, cc = sp.symbols("aa bb cc")
    assert sp.expand(
        ((bb + aa) ** 3 + (bb - aa) ** 3 - 2 * bb**3) / 6 - aa**2 * bb
    ) == 0
    assert sp.expand(
        (
            (aa + bb + cc) ** 3
            + (aa - bb - cc) ** 3
            - (aa + bb - cc) ** 3
            - (aa - bb + cc) ** 3
        )
        / 24
        - aa * bb * cc
    ) == 0

    cube_columns = defaultdict(lambda: defaultdict(lambda: sp.Rational(0)))

    def add_cube(output_coordinate, coefficient, form):
        key, sign = canonical_form(form)
        cube_columns[key][output_coordinate] += sp.cancel(coefficient * sign)

    for output_coordinate, coefficient, exponents in homogeneous_terms:
        pattern = sorted(exponents.values(), reverse=True)
        if pattern == [3]:
            index = next(iter(exponents))
            add_cube(output_coordinate, coefficient, {index: 1})
        elif pattern == [2, 1]:
            repeated = next(index for index, exponent in exponents.items() if exponent == 2)
            singleton = next(index for index, exponent in exponents.items() if exponent == 1)
            add_cube(output_coordinate, coefficient / 6, {singleton: 1, repeated: 1})
            add_cube(output_coordinate, coefficient / 6, {singleton: 1, repeated: -1})
            add_cube(output_coordinate, -coefficient / 3, {singleton: 1})
        elif pattern == [1, 1, 1]:
            first, second, third = sorted(exponents)
            add_cube(
                output_coordinate,
                coefficient / 24,
                {first: 1, second: 1, third: 1},
            )
            add_cube(
                output_coordinate,
                coefficient / 24,
                {first: 1, second: -1, third: -1},
            )
            add_cube(
                output_coordinate,
                -coefficient / 24,
                {first: 1, second: 1, third: -1},
            )
            add_cube(
                output_coordinate,
                -coefficient / 24,
                {first: 1, second: -1, third: 1},
            )
        else:
            raise AssertionError(f"unexpected cubic exponent pattern: {exponents}")

    columns = []
    for form, output_vector in cube_columns.items():
        cleaned = {
            coordinate: sp.cancel(value)
            for coordinate, value in output_vector.items()
            if sp.cancel(value) != 0
        }
        if cleaned:
            columns.append((form, cleaned))
    # Preserve first-occurrence order.  RREF pivot selection depends on column
    # order; this deterministic order gives a materially sparser 211-cube basis.
    merged_number_of_forms = len(columns)
    assert merged_number_of_forms == 221

    # Coefficientwise polynomial check of H=B0*(D0*Z)^{*3}.  Ordered triples
    # automatically produce the multinomial coefficients in a cube expansion.
    expected_coefficients = defaultdict(lambda: sp.Rational(0))
    for output_coordinate, coefficient, exponents in homogeneous_terms:
        key = (output_coordinate, tuple(sorted(exponents.items())))
        expected_coefficients[key] += coefficient
    actual_coefficients = defaultdict(lambda: sp.Rational(0))
    for form, output_vector in columns:
        for first_index, first_coefficient in form:
            for second_index, second_coefficient in form:
                for third_index, third_coefficient in form:
                    exponent_counts = defaultdict(int)
                    exponent_counts[first_index] += 1
                    exponent_counts[second_index] += 1
                    exponent_counts[third_index] += 1
                    monomial = tuple(sorted(exponent_counts.items()))
                    cube_coefficient = (
                        first_coefficient * second_coefficient * third_coefficient
                    )
                    for output_coordinate, output_coefficient in output_vector.items():
                        actual_coefficients[(output_coordinate, monomial)] += (
                            output_coefficient * cube_coefficient
                        )
    expected_coefficients = {
        key: sp.cancel(value)
        for key, value in expected_coefficients.items()
        if sp.cancel(value) != 0
    }
    actual_coefficients = {
        key: sp.cancel(value)
        for key, value in actual_coefficients.items()
        if sp.cancel(value) != 0
    }
    assert actual_coefficients == expected_coefficients

    # Ten scalar cube polynomials are linearly dependent on the others. Select
    # an exact RREF basis and push the basis-change coefficients into B0.
    columns, basis_reduction = basis_reduce_cube_columns(columns)
    number_of_forms = len(columns)
    assert basis_reduction["rank"] == number_of_forms == 211
    assert len(basis_reduction["nonpivot_relations"]) == 10

    # Independent coefficientwise reconstruction after basis reduction.
    basis_actual_coefficients = defaultdict(lambda: sp.Rational(0))
    for form, output_vector in columns:
        expansion = scalar_cube_coefficients(form)
        for monomial, cube_coefficient in expansion.items():
            for output_coordinate, output_coefficient in output_vector.items():
                basis_actual_coefficients[(output_coordinate, monomial)] += (
                    output_coefficient * cube_coefficient
                )
    basis_actual_coefficients = {
        key: sp.cancel(value)
        for key, value in basis_actual_coefficients.items()
        if sp.cancel(value) != 0
    }
    assert basis_actual_coefficients == expected_coefficients

    def cubes_h(point):
        value = [sp.Rational(0)] * homogeneous_dimension
        for form, output_vector in columns:
            cube = linear_value(form, point) ** 3
            for coordinate, coefficient in output_vector.items():
                value[coordinate] += coefficient * cube
        return [sp.cancel(entry) for entry in value]

    assert all(cubes_h(point) == homogeneous_h(point) for point in homogeneous_sources)
    probe = [sp.Rational((index % 7) - 3, (index % 3) + 1) for index in range(homogeneous_dimension)]
    assert cubes_h(probe) == homogeneous_h(probe)
    log("cube decomposition: 221 merged forms, exact scalar-cube rank 211")

    # Collision-tailored pairing: B=[B0|target], D=[D0;0], A0=D*B.
    rational_dimension = number_of_forms + 1
    rational_rows = []
    for form, _ in columns:
        row = defaultdict(lambda: sp.Rational(0))
        for homogeneous_coordinate, d_coefficient in form:
            for column_index, (_, output_vector) in enumerate(columns):
                if homogeneous_coordinate in output_vector:
                    row[column_index] += (
                        d_coefficient * output_vector[homogeneous_coordinate]
                    )
            row[number_of_forms] += d_coefficient * target[homogeneous_coordinate]
        rational_rows.append(
            {
                column: sp.cancel(value)
                for column, value in row.items()
                if sp.cancel(value) != 0
            }
        )
    rational_rows.append({})
    assert len(rational_rows) == rational_dimension == 212

    # Exact structural ranks and nilpotency.  With B=[B0|target] and
    # D=[D0;0], one has BD=B0*D0.  Since D has full column rank,
    # rank(DB)=rank(B).  Also A^k=D*(BD)^(k-1)*B for k>=2.
    d0_matrix = sp.MutableSparseMatrix(
        number_of_forms,
        homogeneous_dimension,
        {
            (row, coordinate): coefficient
            for row, (form, _) in enumerate(columns)
            for coordinate, coefficient in form
        },
    )
    b0_matrix = sp.MutableSparseMatrix(
        homogeneous_dimension,
        number_of_forms,
        {
            (coordinate, column): coefficient
            for column, (_, output_vector) in enumerate(columns)
            for coordinate, coefficient in output_vector.items()
        },
    )
    target_column = sp.MutableSparseMatrix(
        homogeneous_dimension,
        1,
        {
            (coordinate, 0): coefficient
            for coordinate, coefficient in enumerate(target)
            if coefficient
        },
    )
    b_matrix = b0_matrix.row_join(target_column)
    bd_matrix = b0_matrix * d0_matrix
    rank_d = int(d0_matrix.rank())
    rank_b = int(b_matrix.rank())
    rank_bd = int(bd_matrix.rank())
    assert rank_d == 47
    assert rank_b == 32
    assert rank_bd == 0
    assert bd_matrix == sp.zeros(homogeneous_dimension)
    rank_a = rank_b  # D is injective, so rank(D*B)=rank(B).
    assert rank_a == 32

    rational_preimages = []
    for source in homogeneous_sources:
        d0_source = [linear_value(form, source) for form, _ in columns]
        rational_preimages.append([-value**3 for value in d0_source] + [sp.Rational(1)])

    rational_image = [sp.Rational(0)] * number_of_forms + [sp.Rational(1)]
    for preimage in rational_preimages:
        product = sparse_matvec(rational_rows, preimage)
        image = [
            sp.cancel(preimage[index] + product[index] ** 3)
            for index in range(rational_dimension)
        ]
        assert image == rational_image
    assert len({tuple(point) for point in rational_preimages}) == 3

    denominators = [
        int(sp.denom(value)) for row in rational_rows for value in row.values()
    ]
    denominator_lcm = 1
    for denominator in denominators:
        denominator_lcm = lcm(denominator_lcm, denominator)
    assert denominator_lcm == 24
    scalar = 1
    while scalar * scalar % denominator_lcm:
        scalar += 1
    assert scalar == 12
    coordinate_scale = scalar**3
    matrix_scale = scalar**2

    integer_rows = [
        {
            column: sp.Rational(matrix_scale * value)
            for column, value in row.items()
        }
        for row in rational_rows
    ]
    assert all(value.q == 1 for row in integer_rows for value in row.values())
    integer_preimages = [
        [sp.cancel(value / coordinate_scale) for value in point]
        for point in rational_preimages
    ]
    integer_image = [sp.cancel(value / coordinate_scale) for value in rational_image]

    for preimage in integer_preimages:
        product = sparse_matvec(integer_rows, preimage)
        image = [
            sp.cancel(preimage[index] + product[index] ** 3)
            for index in range(rational_dimension)
        ]
        assert image == integer_image
    assert len({tuple(point) for point in integer_preimages}) == 3

    matrix_values = [int(value) for row in integer_rows for value in row.values()]
    point_values = [value for point in integer_preimages for value in point]
    stats = {
        "dimension": rational_dimension,
        "nonzero_matrix_entries": sum(len(row) for row in integer_rows),
        "max_abs_matrix_entry": max(abs(value) for value in matrix_values),
        "rational_matrix_denominator_lcm": denominator_lcm,
        "matrix_scaling": matrix_scale,
        "coordinate_scaling": coordinate_scale,
        "max_abs_point_numerator": max(abs(int(sp.numer(value))) for value in point_values),
        "max_point_denominator": max(int(sp.denom(value)) for value in point_values),
        "nonzero_point_entries": [
            sum(value != 0 for value in point) for point in integer_preimages
        ],
        "reduction_steps": len(reduction_steps),
        "reduced_dimension": reduced_dimension,
        "homogeneous_dimension": homogeneous_dimension,
        "cube_forms": number_of_forms,
        "merged_cube_forms_before_basis_reduction": merged_number_of_forms,
        "cube_polynomial_rank": basis_reduction["rank"],
        "eliminated_dependent_cube_forms": len(basis_reduction["nonpivot_relations"]),
        "cube_coefficients_nonzero": sum(len(vector) for _, vector in columns),
        "linear_form_coefficients_nonzero": sum(len(form) for form, _ in columns),
        "rank_D": rank_d,
        "rank_B": rank_b,
        "rank_BD": rank_bd,
        "rank_A": rank_a,
        "BD_zero": True,
        "BD_squared_zero": True,
        "A_squared_zero": True,
        "A_squared_nonzero": False,
        "A_cubed_zero": True,
        "nilpotency_index_A": 2,
    }
    assert stats["nonzero_matrix_entries"] == 5415
    assert stats["max_abs_matrix_entry"] == 3888
    log(
        "Druzkowski matrix:",
        f"{stats['dimension']}x{stats['dimension']},",
        f"nnz={stats['nonzero_matrix_entries']},",
        f"max|a_ij|={stats['max_abs_matrix_entry']}",
    )

    return {
        "seed": seed,
        "original_points": original_points,
        "original_target": original_target,
        "reduction_steps": reduction_steps,
        "reduced_linear_part": linear_part,
        "reduced_linear_part_inverse": linear_part_inverse,
        "reduced_points": reduced_points,
        "reduced_target": reduced_values[0],
        "homogeneous_terms": homogeneous_terms,
        "homogeneous_sources": homogeneous_sources,
        "homogeneous_target": target,
        "columns": columns,
        "basis_reduction": basis_reduction,
        "rational_rows": rational_rows,
        "integer_rows": integer_rows,
        "integer_preimages": integer_preimages,
        "integer_image": integer_image,
        "stats": stats,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_artifacts(result, output_directory=HERE):
    output_directory.mkdir(parents=True, exist_ok=True)
    stats = result["stats"]
    dimension = stats["dimension"]

    def sparse_matrix_rows(matrix):
        return [
            [
                [column + 1, qstr(matrix[row, column])]
                for column in range(matrix.cols)
                if matrix[row, column] != 0
            ]
            for row in range(matrix.rows)
        ]

    matrix_path = output_directory / "A_integer.coo.tsv"
    matrix_lines = [
        "# Explicit integer Druzkowski matrix A (one-based sparse COO)",
        f"# dimension\t{dimension}",
        f"# nonzeros\t{stats['nonzero_matrix_entries']}",
        "# row\tcolumn\tvalue",
    ]
    for row_index, row in enumerate(result["integer_rows"], start=1):
        for column_index, value in sorted(row.items()):
            matrix_lines.append(f"{row_index}\t{column_index + 1}\t{int(value)}")
    matrix_path.write_text("\n".join(matrix_lines) + "\n", encoding="utf-8")

    point_names = ["u", "v", "w"]
    points_payload = {
        "dimension": dimension,
        "map": "D_A(X) = X + (A X)^{*3}",
        "preimages": {
            name: [qstr(value) for value in point]
            for name, point in zip(point_names, result["integer_preimages"])
        },
        "common_image": [qstr(value) for value in result["integer_image"]],
    }
    points_path = output_directory / "points.json"
    points_path.write_text(
        json.dumps(points_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    block_reduction_payload = {
        "indexing": "one-based",
        "formula": (
            "replace coordinate F_i by F_i-(r+P)(s+Q), append r+P and s+Q, "
            "and lift a to (a,-P(a),-Q(a))"
        ),
        "seed_dimension": 3,
        "step_count": len(result["reduction_steps"]),
        "reduced_dimension": stats["reduced_dimension"],
        "steps": result["reduction_steps"],
        "linear_normalization": {
            "operation": "postcompose the reduced map with L_inverse",
            "L_determinant": "1",
            "L_rows": sparse_matrix_rows(result["reduced_linear_part"]),
            "L_inverse_rows": sparse_matrix_rows(
                result["reduced_linear_part_inverse"]
            ),
            "normalized_linear_part": "identity",
        },
        "normalized_collision_points": [
            [qstr(value) for value in point] for point in result["reduced_points"]
        ],
        "normalized_common_target": [
            qstr(value) for value in result["reduced_target"]
        ],
    }
    block_reduction_path = output_directory / "block_reduction.json"
    block_reduction_path.write_text(
        json.dumps(block_reduction_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    pairing_payload = {
        "homogeneous_dimension": stats["homogeneous_dimension"],
        "number_of_cube_forms": stats["cube_forms"],
        "indexing": "one-based",
        "D0_rows": [
            [[index + 1, qstr(coefficient)] for index, coefficient in form]
            for form, _ in result["columns"]
        ],
        "B0_columns": [
            [
                [coordinate + 1, qstr(coefficient)]
                for coordinate, coefficient in sorted(output_vector.items())
            ]
            for _, output_vector in result["columns"]
        ],
        "common_target": [qstr(value) for value in result["homogeneous_target"]],
    }
    pairing_path = output_directory / "pairing.json"
    pairing_path.write_text(
        json.dumps(pairing_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    basis = result["basis_reduction"]
    basis_payload = {
        "indexing": "one-based",
        "claim": "the 221 merged scalar cubes have rank 211",
        "prebasis_count": len(basis["prebasis_columns"]),
        "basis_count": basis["rank"],
        "scalar_cube_support_monomials": basis["scalar_cube_support_monomials"],
        "pivot_columns": [column + 1 for column in basis["pivot_columns"]],
        "prebasis_D0_rows": [
            [[index + 1, qstr(coefficient)] for index, coefficient in form]
            for form, _ in basis["prebasis_columns"]
        ],
        "prebasis_B0_columns": [
            [
                [coordinate + 1, qstr(coefficient)]
                for coordinate, coefficient in sorted(output_vector.items())
            ]
            for _, output_vector in basis["prebasis_columns"]
        ],
        "nonpivot_relations": [
            {
                "old_column": relation["old_column"] + 1,
                "basis_coefficients": [
                    [basis_row + 1, qstr(coefficient)]
                    for basis_row, coefficient in relation["coefficients"]
                ],
            }
            for relation in basis["nonpivot_relations"]
        ],
    }
    basis_path = output_directory / "basis_reduction.json"
    basis_path.write_text(
        json.dumps(basis_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    generated_hashes = {
        path.name: sha256(path)
        for path in [
            matrix_path,
            points_path,
            block_reduction_path,
            pairing_path,
            basis_path,
        ]
    }
    certificate_payload = {
        "claim": {
            "matrix": "A_integer.coo.tsv",
            "points": "points.json",
            "determinant_identity": "det J(D_A) == 1",
            "collision": "D_A(u) == D_A(v) == D_A(w)",
        },
        "seed_map": [str(polynomial) for polynomial in result["seed"]],
        "seed_points": [
            [qstr(value) for value in point] for point in result["original_points"]
        ],
        "seed_target": [qstr(value) for value in result["original_target"]],
        "construction": {
            "degree_reduction": (
                "fixed audited 10-step determinant-one polynomial-block BCW "
                "schedule, followed by exact determinant-one linear normalization"
            ),
            "homogenization": "Phi(X,Y,T)=(X+T*R2(X)-T^2*Y,Y+R3(X),T)",
            "cube_basis_reduction": "221 merged cubes, exact RREF rank 211",
            "pairing": "B=[B0|target], D=[D0;0], A0=D*B",
            "integer_scaling": "A=12^2*A0; preimages=preimages0/12^3",
        },
        "stats": stats,
        "sha256": generated_hashes,
    }
    certificate_path = output_directory / "certificate.json"
    certificate_path.write_text(
        json.dumps(certificate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    all_hashes = dict(generated_hashes)
    all_hashes[certificate_path.name] = sha256(certificate_path)
    sums_path = output_directory / "SHA256SUMS"
    sums_path.write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(all_hashes.items())),
        encoding="utf-8",
    )
    return [
        matrix_path,
        points_path,
        block_reduction_path,
        pairing_path,
        basis_path,
        certificate_path,
        sums_path,
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args()
    result = build_counterexample(verbose=True)
    if not args.no_write:
        paths = write_artifacts(result, args.output)
        print("wrote:")
        for path in paths:
            print(f"  {path}")
    print("all exact construction checks passed")


if __name__ == "__main__":
    main()
