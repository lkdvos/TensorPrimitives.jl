"""
    TensorPrimitives

Julia bindings for [`tensorprimitives-rs`](https://github.com/lkdvos/tensorprimitives-rs):
a native-Rust, transpose-free dense tensor contraction engine, reached through its
TAPP C ABI.

Two layers:

  * [`LibTAPP`](@ref) — the raw ABI, one function per exported symbol, with handle
    lifetimes and error checking. Use this if you want to plan a contraction once
    and execute it many times.
  * [`TAPPBackend`](@ref) — a `TensorOperations.jl` backend, so an existing
    `@tensor` expression can be routed through this engine by adding
    `backend = TAPPBackend()`.

```julia
using TensorOperations, TensorPrimitives
A = randn(ComplexF64, 8, 12, 4); B = randn(ComplexF64, 4, 12, 5)
@tensor backend = TAPPBackend() C[i, j] := conj(A[i, k, l]) * B[l, k, j]
```

!!! warning "Prerelease"
    Version 0.1.0. The engine is correct in every element type, method, instruction
    set and thread count it offers, and its *performance* is measured on exactly
    one machine. Cache blocking is untuned off Cascade Lake, the AVX2 register
    blocks are unmeasured, threading is off by default, and off x86 there is no
    vectorised kernel at all. See the upstream `CHANGELOG.md`, which states this
    per component.
"""
module TensorPrimitives

include("LibTAPP.jl")
using .LibTAPP: LibTAPP, TAPPError

export TAPPBackend, TAPPError

# ------------------------------------------------------------- process context
#
# `TAPP_handle` and `TAPP_executor` carry no state that varies per call -- the
# handle is a context object and the executor is inert in this implementation --
# so one of each per process is right, created on first use rather than in
# `__init__` so that merely loading the package touches no library state.
#
# Not thread-safe to *initialise* concurrently, which is why each is behind a
# lock; the library itself synchronises nothing, and the header says so.

const _CONTEXT_LOCK = ReentrantLock()
const _HANDLE = Ref{Union{Nothing, LibTAPP.Handle}}(nothing)
const _EXECUTOR = Ref{Union{Nothing, LibTAPP.Executor}}(nothing)

function _handle()
    h = _HANDLE[]
    h === nothing || return h
    return lock(_CONTEXT_LOCK) do
        h = _HANDLE[]
        h === nothing || return h
        _HANDLE[] = LibTAPP.Handle()
    end
end

function _executor()
    e = _EXECUTOR[]
    e === nothing || return e
    return lock(_CONTEXT_LOCK) do
        e = _EXECUTOR[]
        e === nothing || return e
        _EXECUTOR[] = LibTAPP.Executor()
    end
end

# --------------------------------------------------------------- introspection

"""
    implementation_name() -> String

Which TAPP provider is linked, as the library itself reports it.
"""
implementation_name() = LibTAPP.implementation_name()

"""
    implementation_version() -> VersionNumber

The version of the loaded shared library, as the library itself reports it — not
the version of this Julia package, and not the version of any header. They can
differ when the JLL and the wrapper are updated separately, and this is how you
find out.
"""
implementation_version() = LibTAPP.implementation_version()

"""
    thread_setting() -> Int

How many threads the engine will use, read the way the engine reads it.

**The engine is single-threaded by default and this is not currently settable from
Julia.** `TAPP_execute_product` ignores its executor argument entirely, so
`TENSORCONTRACT_THREADS` is the only control, and the engine caches it in a
`OnceLock` on first use — setting `ENV["TENSORCONTRACT_THREADS"]` after the first
contraction has no effect, silently.

To use more than one thread, set it before the first contraction:

```julia
ENV["TENSORCONTRACT_THREADS"] = Sys.CPU_THREADS ÷ 2
using TensorOperations, TensorPrimitives
```

or export it before starting Julia. Results are **bitwise identical** at every
thread count — no reduction is parallelised — so this is never a correctness or
accuracy decision. Thread *scaling*, on the other hand, has not been measured on
the machine you are reading this on, or on any machine.
"""
function thread_setting()
    v = get(ENV, "TENSORCONTRACT_THREADS", "1")
    n = tryparse(Int, v)
    n === nothing && return 1
    return max(n, 1)
end

include("backend.jl")

end # module TensorPrimitives
