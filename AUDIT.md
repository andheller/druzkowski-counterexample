# Audit guide

The claim in this repository has several layers. They can be checked separately.

## Fastest check: the explicit collision

`verify_artifacts.py` reads `A_integer.coo.tsv` and `points.json`, evaluates

\[
D_A(x)=x+(Ax)^{*3}
\]

at all three saved points using exact rationals, and compares the resulting
212 coordinates. It also multiplies the sparse matrix by itself to check
`A^2 = 0`.

## Structural determinant certificate

The determinant claim is not established by attempting to expand a raw
212-by-212 symbolic determinant. The certificate instead records the smaller
maps and exact identities from which it follows:

1. Verify the three-variable seed has determinant one.
2. Replay each of the ten polynomial-block stabilizations in
   `block_reduction.json`. Each is a source shear followed by a target shear,
   both with determinant one.
3. Check the saved linear normalization has determinant one.
4. Reconstruct the 47-dimensional cubic homogeneous map and its identity
   \(H=B_0(D_0z)^{*3}\) from `pairing.json`.
5. Form \(B=[B_0\mid\tau]\), \(D=[D_0;0]\), and check the published integer
   matrix is exactly \(A=144DB\).
6. Apply Sylvester's determinant identity to transfer
   \(\det J\Phi=1\) to \(\det JD_A=1\).

`verify_artifacts.py` performs these checks without importing `generate.py`.
It treats the committed TSV and JSON files as its inputs and independently
reimplements the certificate identities.

## Regeneration check

`verify.py` follows a different path: it runs the deterministic construction
in `generate.py`, compares every generated matrix, point, block, basis, and
pairing entry with the committed artifacts, then checks their hashes.

Run both:

```bash
python verify_artifacts.py
python verify.py
sha256sum -c SHA256SUMS
```

All three commands use exact integer or rational arithmetic. No randomized or
floating-point test is part of the certificate.

## Useful review targets

The most valuable independent review is the chain

\[
F_3\rightsquigarrow F_{23}\rightsquigarrow\Phi_{47}.
\]

In particular, reviewers should check the ten stable-equivalence identities,
the determinant-one linear normalization, the homogenization block determinant,
and the coefficientwise cube decomposition. The final \(47\to212\) pairing is
short enough to verify directly from Section 4 of the README.

Bug reports should include the verifier command, Python and SymPy versions,
and the first failed assertion or differing artifact entry.
