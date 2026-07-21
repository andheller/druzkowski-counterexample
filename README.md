# An Explicit 212-Dimensional Square-Zero Drużkowski Counterexample

[![Exact verification](https://github.com/andheller/druzkowski-counterexample/actions/workflows/verify.yml/badge.svg)](https://github.com/andheller/druzkowski-counterexample/actions/workflows/verify.yml)

This repository gives a concrete cubic-linear normal form derived from the
Alpöge–Fable three-variable Keller map: an explicit integer matrix

$$
A\in M_{212}(\mathbb Z)
$$

and three distinct rational points $u,v,w$ such that, for
$D_A(X)=X+(AX)^{*3}$,

$$
\det J\bigl(X+(AX)^{*3}\bigr)\equiv 1,
\qquad
D_A(u)=D_A(v)=D_A(w).
$$

In fact, their common image is $e_{212}/1728$. The matrix satisfies the
additional square-zero reduction condition

$$
A^2=0,\qquad \operatorname{rank}A=32.
$$

The primary object is [`A_integer.coo.tsv`](A_integer.coo.tsv), a one-based
sparse COO representation with 5,415 nonzero entries and
$\max |a_{ij}|=3888$. The complete rational vectors are in
[`points.json`](points.json).

## What this claim is—and is not

This is an explicit computational realization of the classical
Bass–Connell–Wright, Drużkowski, and Gorni–Zampieri reductions. It makes the
resulting “bad matrix” inspectable and reusable; it is not a second independent
disproof of the Jacobian conjecture. The input is the newly announced
three-variable counterexample. Its short exact identities are described by
[MathWorld](https://mathworld.wolfram.com/JacobianConjecture.html) and in an
open [Formal Conjectures pull request](https://github.com/google-deepmind/formal-conjectures/pull/4474).
The announcement should not be confused with a peer-reviewed publication.

Targeted searches found no earlier public matrix derived from this seed. That
is context, not a claim of priority.

## Verify it

Python 3.11 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Artifact-first checker: does not import the construction program.
python verify_artifacts.py

# Deterministically regenerate and compare the complete certificate.
python verify.py
sha256sum -c SHA256SUMS
```

Both verification paths use exact integer and rational arithmetic. No
floating-point or randomized test is used. See [`AUDIT.md`](AUDIT.md) for the
claim boundaries and a suggested independent-review order.

## Construction

### Seed map

Put $q=1+xy$ and define the tangent-to-identity map

$$
\begin{aligned}
F_1&=x-\frac32x^2y-\frac12x^3z,\\
F_2&=y+3xq^2z+3xy^2(4+3xy),\\
F_3&=q^3z+y^2q(4+3xy).
\end{aligned}
$$

Exact symbolic expansion gives $\det JF=1$, and

$$
\begin{aligned}
F(0,0,-1/4)
&=F(1,-3/2,13/2)\\
&=F(-1,3/2,13/2)
=(0,0,-1/4).
\end{aligned}
$$

The generator proves both identities before applying any reduction.

### Constructive reduction

#### 1. Ten polynomial-block reductions

If a current coordinate contains a factorized block $PQ$, introduce fresh
variables $r,s$ and replace the map by

$$
\bigl(F_1,\ldots,F_i-(r+P)(s+Q),\ldots,F_n,r+P,s+Q\bigr).
$$

This is a stabilization of $F$, composed on the source and target with
elementary polynomial automorphisms of Jacobian determinant one. A collision
point $a$ lifts to $(a,-P(a),-Q(a))$. The fixed schedule is:

| Move | Coordinate | $P$ | $Q$ |
|---:|---:|---|---|
| 1 | 3 | $xy^2$ | $x^2yz+3xy^2+3xz+7y$ |
| 2 | 2 | $3x^2y$ | $xyz+3y^2+2z$ |
| 3 | 1 | $-x^2/2$ | $xz$ |
| 4 | 3 | $-x^2$ | $yzx_3$ |
| 5 | 2 | $-3x^2$ | $yx_6$ |
| 6 | 2 | $-xy$ | $zx_5$ |
| 7 | 3 | $-xy$ | $yx_4$ |
| 8 | 3 | $-3xy$ | $yx_3$ |
| 9 | 3 | $-yz$ | $x_3x_9$ |
| 10 | 5 | $x^2$ | $yz$ |

Here $x=x_0,y=x_1,z=x_2$, and each move appends its two fresh variables
in order. The result is a map $\widetilde R$ of degree at most three in 23
variables.

Some block factors contain linear terms, so $\widetilde R$ need not yet be
tangent to the identity. Let $L=J\widetilde R(0)$. The generator verifies
$\det L=1$ exactly and defines

$$
R=L^{-1}\widetilde R=X+R_2+R_3.
$$

This determinant-one output change preserves the determinant identity and
the three source points. `block_reduction.json` records every $P,Q$, every
lifted point, and sparse rows of both $L$ and $L^{-1}$.

#### 2. Cubic homogenization

In 47 variables define

$$
\Phi(X,Y,T)
=\bigl(X+TR_2(X)-T^2Y,\;Y+R_3(X),\;T\bigr)
=(X,Y,T)+H(X,Y,T).
$$

The nonlinear part $H$ is cubic homogeneous. The block determinant is

$$
\det
\begin{pmatrix}
I+TJR_2(X)&-T^2I\\
JR_3(X)&I
\end{pmatrix}
=\det\bigl(I+TJR_2(X)+T^2JR_3(X)\bigr)
=\det JR(TX)=1.
$$

A collision point $a$ of $R$ lifts explicitly to
$(a,-R_3(a),1)$. The three lifted points have a common target $\tau$.

#### 3. Cubes and exact basis reduction

The generator polarizes every cubic monomial using

$$
a^2b=\frac{(b+a)^3+(b-a)^3-2b^3}{6}
$$

and

$$
abc=\frac{(a+b+c)^3+(a-b-c)^3-(a+b-c)^3-(a-b+c)^3}{24}.
$$

After identical and sign-equivalent linear forms are merged, there are 221
scalar cubes. Their exact coefficient matrix has rank 211. An RREF-selected
cube basis gives

$$
H(Z)=B_0(D_0Z)^{*3},
\qquad D_0\in M_{211\times47}(\mathbb Q),
\quad B_0\in M_{47\times211}(\mathbb Q).
$$

`basis_reduction.json` records all 221 pre-basis columns, the 211 pivot
columns, and ten exact dependence relations. `pairing.json` records the
reduced $B_0,D_0,\tau$. The generator reconstructs every coefficient of
the polynomial identity both before and after the basis change.

#### 4. Collision-target cubic-linear embedding

Set

$$
B=[B_0\mid\tau],\qquad
D=\begin{bmatrix}D_0\\0\end{bmatrix},\qquad
A_0=DB\in M_{212}(\mathbb Q).
$$

For $G_{A_0}(W)=W+(A_0W)^{*3}$, put $Z=BW$. Sylvester's determinant
identity gives

$$
\begin{aligned}
\det JG_{A_0}(W)
&=\det\left(I+3\operatorname{diag}((DZ)^{*2})DB\right)\\
&=\det\left(I+3B\operatorname{diag}((DZ)^{*2})D\right)\\
&=\det J\Phi(Z)=1.
\end{aligned}
$$

For each of the three sources $p$ of $\Phi$, define

$$
U_p=\bigl(-(D_0p)^{*3},1\bigr).
$$

Since $\Phi(p)=p+H(p)=\tau$, one has

$$
BU_p=p,\qquad A_0U_p=Dp,\qquad G_{A_0}(U_p)=e_{212}.
$$

The structural certificate verifies

$$
\operatorname{rank}D=47,\quad
\operatorname{rank}B=32,\quad
BD=0.
$$

Consequently $\operatorname{rank}A_0=32$ and
$A_0^2=D(BD)B=0$.

#### 5. Integer matrix

The entries of $A_0$ have denominator LCM 24. Scalar cubic conjugacy with
$s=12$ gives

$$
A=12^2A_0=144A_0\in M_{212}(\mathbb Z),
\qquad u_p=U_p/12^3=U_p/1728.
$$

Thus $D_A=S^{-1}\circ G_{A_0}\circ S$, where $S=1728I$. The three
points in `points.json` map to $e_{212}/1728$. The relation $A^2=0$
is unchanged by scalar multiplication.

## Files and verification

- `A_integer.coo.tsv`: explicit 212-by-212 integer matrix, one-based sparse COO.
- `points.json`: three rational preimages and their common image.
- `block_reduction.json`: ten block reductions, collision lifts, and linear normalization.
- `pairing.json`: reduced sparse $B_0,D_0,\tau$ certificate.
- `basis_reduction.json`: pre-basis cubes, pivots, and exact dependencies.
- `certificate.json`: dimensions, statistics, construction metadata, and hashes.
- `verify_artifacts.py`: artifact-first checker that does not import the generator.
- `generate.py`: deterministic construction from the three-variable seed.
- `verify.py`: regeneration-based comparison of every committed artifact.
- `AUDIT.md`: review guide and explanation of the two verification paths.

The pairing payload records $B_0,D_0,\tau$. The full matrices in the proof
are recovered without ambiguity as $B=[B_0\mid\tau]$ and $D=[D_0;0]$.

## Classical ingredients

- Bass, Connell, and Wright, [*The Jacobian Conjecture: Reduction of Degree
  and Formal Expansion of the Inverse*](https://doi.org/10.1090/S0273-0979-1982-15032-7)
  (1982).
- Drużkowski, [*An Effective Approach to Keller's Jacobian
  Conjecture*](https://doi.org/10.1007/BF01459126) (1983).
- Gorni and Zampieri, [*On Cubic-Linear Polynomial
  Mappings*](https://doi.org/10.1016/S0019-3577(97)81552-2) (1997),
  especially the explicit $A=DB$ pairing viewpoint.
- Meisters, [*Wanted: A Bad Matrix*](https://www.jstor.org/stable/2974772)
  (1995), for the historical “bad matrix” framing.

## Citation and feedback

Repository citation metadata is provided in [`CITATION.cff`](CITATION.cff).
Mathematical review, independent implementations, and minimal-dimension or
rank improvements are especially welcome. Reproducibility reports should
include the failing command and first differing certificate entry.
