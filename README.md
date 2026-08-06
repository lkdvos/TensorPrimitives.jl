# TensorPrimitives.jl

Julia bindings for [`tensorprimitives-rs`](https://github.com/lkdvos/tensorprimitives-rs),
a native-Rust, transpose-free dense tensor contraction engine, reached through its
TAPP C ABI.

**Status: prerelease (0.1.0), not registered.** See [Confidence](#confidence)
before quoting a performance number — this is a correctness-complete engine whose
performance has been measured on exactly one machine.

## Installing

Neither this package nor its JLL is in the General registry yet. Both are
`develop`-able from a checkout:

```julia
using Pkg
Pkg.develop(path = "path/to/tensorprimitives-rs/julia/TensorPrimitives")
Pkg.develop(path = "path/to/tensorprimitives_tapp_jll")   # see below
```

The JLL comes from the BinaryBuilder recipe in the same repository. To build one
locally:

```bash
# In a checkout of tensorprimitives-rs
export BBROOT=/tmp/$USER/bb                       # local disk: overlayfs cannot
mkdir -p $BBROOT/{env,tmp,storage,depot,src}      # use an NFS upperdir
export JULIA_DEPOT_PATH=$BBROOT/depot TMPDIR=$BBROOT/tmp
export BINARYBUILDER_STORAGE_DIR=$BBROOT/storage
julia --project=$BBROOT/env -e 'using Pkg; Pkg.add("BinaryBuilder")'

mkdir -p $BBROOT/src/tensorprimitives-rs-0.1.0
git archive --format=tar HEAD | tar -x -C $BBROOT/src/tensorprimitives-rs-0.1.0

# the recipe lives in a Yggdrasil fork, not in this repository -- see RELEASING.md step 4
cd <yggdrasil-fork>/T/tensorprimitives_tapp
TAPP_LOCAL_SRC=$BBROOT/src julia --project=$BBROOT/env build_tarballs.jl \
    x86_64-linux-gnu --verbose --deploy=local
# -> $BBROOT/depot/dev/tensorprimitives_tapp_jll
```

## Using it

The interesting entry point is a `TensorOperations.jl` backend, so an existing
`@tensor` expression is routed through this engine by adding one keyword:

```julia
using TensorOperations, TensorPrimitives

A = randn(ComplexF64, 8, 12, 4)
B = randn(ComplexF64, 4, 12, 5)

@tensor backend = TAPPBackend() C[i, j] := conj(A[i, k, l]) * B[l, k, j]
```

Conjugation is free — the engine folds it into packing as a negation of the
imaginary plane — and nothing here permutes, copies or allocates a temporary:
arbitrary extents, arbitrary element strides and per-operand conjugation all go
straight to the ABI. That is the whole point of the engine.

The backend is **opt-in**. It is deliberately not registered with
`select_backend`, so loading this package changes nothing about code that does not
ask for it.

Only `tensorcontract!` is implemented. `tensoradd!` and `tensortrace!` fall
through to `TensorOperations`' own backends: TAPP can express both — a trace is a
repeated label, an add is a contraction against a rank-0 operand — but each is a
correctness surface to add with tests rather than acquire for free.

Supported element types are `Float32`, `Float64`, `ComplexF32` and `ComplexF64`,
and all operands must share one. Anything else is refused with an error naming the
type.

### The raw ABI

`TensorPrimitives.LibTAPP` is a thin, complete wrapper over
`crates/tensorprimitives-tapp/include/tapp.h`: one function per exported symbol,
handles as distinct Julia types with finalizers, errors as `TAPPError`. Reach for
it when you want to plan a contraction once and execute it many times, which the
backend above does not do — it builds and destroys a plan per call.

```julia
using TensorPrimitives: LibTAPP

handle = LibTAPP.Handle()
exec = LibTAPP.Executor()
infoA = LibTAPP.TensorInfo(Float64, Int64[4, 6], Int64[1, 4])   # strides in elements
```

### Which library am I actually running?

```julia
TensorPrimitives.implementation_name()      # "tensorprimitives-rs: tensorcontract (...)"
TensorPrimitives.implementation_version()   # v"0.1.0", from the loaded .so
```

`implementation_version` reports the version of the **shared library**, not of this
package. They can differ when the JLL and the wrapper move independently, and this
is how you find out.

## Threading

**The engine is single-threaded by default, and you cannot currently change that
from Julia.** `TAPP_execute_product` ignores its executor argument, so the only
control is the `TENSORCONTRACT_THREADS` environment variable — and the engine
caches it on first use, so setting `ENV[...]` after the first contraction is
silently ignored.

```julia
ENV["TENSORCONTRACT_THREADS"] = Sys.CPU_THREADS ÷ 2   # before the first contraction
using TensorOperations, TensorPrimitives
```

or export it before starting Julia. `TensorPrimitives.thread_setting()` reports
what the process will do.

Results are **bitwise identical at every thread count** — no reduction is
parallelised, every output element has one owning thread accumulating over the full
contracted extent in the original order — so this is never a correctness or
accuracy decision. Thread *scaling*, on the other hand, has not been measured.

## Confidence

Taken from the upstream `CHANGELOG.md`, which states this per component. The
distinction matters more than the numbers.

| | |
|---|---|
| Correctness, in every element type, complex method, instruction set and thread count | **Tested** against a brute-force oracle and against TBLIS and OpenBLAS-TTGT: randomised extents and axis orders, diagonals, reductions, negative strides, all 16 conjugation masks, shapes crossing every cache-blocking level |
| AVX-512 register blocks | **Measured**, per method and element type |
| AVX2 register blocks | **Provisional and unmeasured** |
| Cache blocking (`MC`/`KC`/`NC`) | **Untuned** off the one Cascade Lake machine it was fitted to. Nothing about the constants transfers to another cache hierarchy |
| Threading | Implemented, correct, **scaling unmeasured**, off by default |
| Absolute throughput on your machine | **Unknown** |

Off x86 there is **no hand-written vectorised kernel** — the portable scalar path
is what runs, and a slower aarch64 run is documented behaviour, not a bug.

**How much slower is now measured, and it is less than this note used to imply.**
On an Apple M3 Max, single-threaded over 12 shapes at 64 MiB, the scalar path
reaches a geomean 19.8 GF/s in `f64` and 24.2 in `c64`, against an OpenBLAS-TTGT
baseline at 30.7 and 35.0 — about **0.65x**, not a tenth.

LLVM vectorises the micro-kernel to NEON unprompted; what it cannot do is fuse
`acc += a * b` into a single `fmla`, because that would drop a rounding. So the
path spends two instructions per multiply-accumulate where the hardware offers
one, which halves the ceiling available to it — and on its best shape it then
reaches 80.5% of that halved ceiling, against OpenBLAS's 87.1% of the full one.
Expect roughly a factor of two from a hand-written NEON kernel, not ten.

Measured on one Apple part only, and **not a pinned measurement** — macOS has no
CPU affinity API. Nothing here transfers to another aarch64 core.

## License

MIT OR Apache-2.0, matching the engine.
