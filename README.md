# TensorPrimitives.jl

Julia bindings for [`tensorprimitives-rs`](https://github.com/lkdvos/tensorprimitives-rs),
a native-Rust, transpose-free dense tensor contraction engine, reached through its
TAPP C ABI.

**Status: prerelease (0.1.0), not registered.** See [Confidence](#confidence) before
quoting a performance number.

## Installing

`tensorprimitives_tapp_jll` is in the General registry, so the binary resolves by
itself; only this package needs `develop`-ing until it is registered too:

```julia
using Pkg
Pkg.develop(path = "path/to/TensorPrimitives.jl")
```

### Against a locally built engine

The JLL comes from a BinaryBuilder recipe in
[Yggdrasil](https://github.com/JuliaPackaging/Yggdrasil/tree/master/T/tensorprimitives_tapp),
pinned to a released commit. To test against an engine that is *not* released — or to
regenerate the bindings for a changed header — build a JLL locally and `develop` that
too:

```bash
export BBROOT=/tmp/$USER/bb                   # local disk: overlayfs cannot
mkdir -p $BBROOT/{env,tmp,storage,depot}      # use an NFS upperdir
export JULIA_DEPOT_PATH=$BBROOT/depot TMPDIR=$BBROOT/tmp
export BINARYBUILDER_STORAGE_DIR=$BBROOT/storage
julia --project=$BBROOT/env -e 'using Pkg; Pkg.add("BinaryBuilder")'

cd <yggdrasil-fork>/T/tensorprimitives_tapp
julia --project=$BBROOT/env build_tarballs.jl x86_64-linux-gnu \
    --verbose --deploy=local
# -> $BBROOT/depot/dev/tensorprimitives_tapp_jll
```

The recipe builds from the commit it pins, so a local build is of a released commit
and not of your working tree. `gen/generator.jl` also takes `TAPP_INCLUDE_DIR`, which
points it at an `install.sh --prefix=` layout instead of the JLL, for the case where the
header has moved ahead of any published artifact.

## Using it

### Through `@tensor`

The interesting entry point is a `TensorOperations.jl` backend. It is **a package
extension of `TensorOperations` over this package**, so loading both is what turns it
on — this package itself does not depend on `TensorOperations`:

```julia
using TensorOperations, TensorPrimitives

A = randn(ComplexF64, 8, 12, 4)
B = randn(ComplexF64, 4, 12, 5)

@tensor backend = TensorOperations.TAPPBackend() C[i, j] := conj(A[i, k, l]) * B[l, k, j]
```

Conjugation is free — the engine folds it into packing as a negation of the imaginary
plane — and a contraction permutes nothing, copies nothing and allocates no temporary:
arbitrary extents, arbitrary element strides and per-operand conjugation all go
straight to the ABI. That is the whole point of the engine, and it is why the mapping
from `TensorOperations`' `pA`/`pB`/`pAB` is a relabelling rather than a permutation.

The backend is **opt-in**: it is deliberately not registered with `select_backend`, so
loading these packages changes nothing about code that does not ask for it.
`tensoradd!` and `tensortrace!` are delegated to `TensorOperations`' own backends
rather than computed here, so any `@tensor` expression works, with the contractions
going through this engine.

Supported element types are `Float32`, `Float64`, `ComplexF32` and `ComplexF64`, and
all operands must share one. Anything else is refused with an error naming the type.

### Directly

Use this package on its own to plan a contraction once and execute it many times,
which the backend does not do — it builds and destroys a plan per call.

```julia
using TensorPrimitives

info = TensorPrimitives.TensorInfo(Float64, Int64[4, 6], Int64[1, 4])  # strides in elements
plan = TensorPrimitives.Product(TensorPrimitives.handle(), ...)
TensorPrimitives.execute!(plan, TensorPrimitives.executor(), ...)
```

`handle()` and `executor()` are process-wide and created on first use. Handles are
distinct Julia types with finalizers, so lifetimes take care of themselves;
`destroy!` is available and idempotent when you want to be explicit.

`TensorPrimitives.LibTAPP` is the raw ABI, one function per exported symbol,
**generated** from the header the JLL ships. Every enumerator is there, including the
two datatypes the ABI declares and this library refuses.

### Which library am I actually running?

```julia
TensorPrimitives.implementation_name()      # "tensorprimitives-rs: tensorcontract (...)"
TensorPrimitives.implementation_version()   # v"0.1.0", from the loaded .so
TensorPrimitives.header_version()           # v"0.1.0", from the generated bindings
```

`implementation_version` reports the version of the **shared library**, not of this
package — they can differ when the JLL and the wrapper move independently, and this is
how you find out. It is also the version to name when quoting a performance number.
The test suite asserts the two agree.

## Threading

**The engine is single-threaded by default, and you cannot currently change that from
Julia.** `TAPP_execute_product` ignores its executor argument, so the only control is
the `TENSORCONTRACT_THREADS` environment variable — and the engine caches it on first
use, so setting `ENV[...]` after the first contraction is silently ignored.

```julia
ENV["TENSORCONTRACT_THREADS"] = Sys.CPU_THREADS ÷ 2   # before the first contraction
using TensorOperations, TensorPrimitives
```

or export it before starting Julia. `TensorPrimitives.thread_setting()` reports what
the process will do.

Results are **bitwise identical at every thread count** — no reduction is
parallelised, every output element has one owning thread accumulating over the full
contracted extent in the original order — so this is never a correctness or accuracy
decision.

## Confidence

Read from the upstream `CHANGELOG.md`, which states this per component and is the only
place to take it from. The distinction matters more than the numbers.

| | |
|---|---|
| Correctness, in every element type, complex method, instruction set and thread count | **Tested** against a brute-force oracle and against TBLIS and OpenBLAS-TTGT: randomised extents and axis orders, diagonals, reductions, negative strides, all 16 conjugation masks, shapes crossing every cache-blocking level |
| Register blocks — AVX-512, AVX2, NEON | **Measured, each on exactly one microarchitecture.** They are a property of the microarchitecture rather than of the instruction set, so none of the three transfers. Only the NEON menu was measured with an error bar, and six of its eight winners came out as *ties* |
| Cache blocking (`MC`/`KC`/`NC`) | **Swept on two machines and closed with a negative result:** no few-percent win, and the shipped constants are fitted to one Cascade Lake workstation |
| Threading | **Scaling measured on seven nodes**, off by default for a measured reason: below ~1 MiB at 64 threads it is slower than serial, and the optimal thread count walks 4 → 64 across the size range. Not reachable from Julia anyway |
| Absolute throughput on your machine | **Unknown** |
| Anything measured *through this wrapper* | **Nothing.** No number upstream was taken through Julia |

**Off x86 there is a hand-written NEON kernel**, measured end to end on an M3 Max at
**1.84x** the portable path in `f64` and reaching **1.00x TBLIS 2.0-dev** there — not a
pinned measurement, since Darwin has no CPU affinity API, and nothing about it
transfers to another aarch64 core. An earlier version of this file said there was no
vectorised kernel off x86; that stopped being true.

## Developing

The raw bindings in `src/lib/` are generated and committed. Regenerate them with
[`gen/generator.jl`](gen/README.md) after any header change; CI regenerates and fails
on a diff, so they cannot drift from the header. Formatting is
[Runic](https://github.com/fredrikekre/Runic.jl), checked in CI.

## License

MIT OR Apache-2.0, matching the engine.
