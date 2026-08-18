"""
    TensorPrimitives

Julia bindings for [`tensorprimitives-rs`](https://github.com/lkdvos/tensorprimitives-rs):
a native-Rust, transpose-free dense tensor contraction engine, reached through its
TAPP C ABI.

Two layers:

  * [`LibTAPP`](@ref) — the raw ABI, generated from the shipped header.
  * this module — handles with lifetimes, errors as exceptions, and the element-type
    mapping. Use it to plan a contraction once and execute it many times.

For `@tensor` expressions, load `TensorOperations` as well and pass its
`TAPPBackend`, which is a package extension of `TensorOperations` over this package:

```julia
using TensorOperations, TensorPrimitives
A = randn(ComplexF64, 8, 12, 4); B = randn(ComplexF64, 4, 12, 5)
@tensor backend = TensorOperations.TAPPBackend() C[i, j] := conj(A[i, k, l]) * B[l, k, j]
```

!!! warning "Prerelease"
    The engine is correct in every element type, complex method, instruction set and
    thread count it offers. Its *performance* is measured per component and unevenly:
    the AVX-512, AVX2 and NEON register blocks are each measured on one
    microarchitecture, cache blocking is tuned on one machine only, and threading is
    off by default with its scaling measured but topology-dependent. The upstream
    `CHANGELOG.md` states this per component and is the only place to read it from.
    Nothing here has been measured *through Julia*.
"""
module TensorPrimitives

include("LibTAPP.jl")
using .LibTAPP: LibTAPP
using CEnum: CEnum

export TAPPError, TAPPElement

"""
    TensorPrimitives.ABICode

Anything the ABI accepts where it declares an `int`: a Julia integer, or one of the
generated enumerators.

The enumerators are CEnum values, and CEnum values are deliberately **not**
`<: Integer` — they convert and compare like integers but do not subtype them — so a
signature of `::Integer` would reject `LibTAPP.TAPP_IDENTITY`, which is exactly the
value a caller has in hand.
"""
const ABICode = Union{Integer, CEnum.Cenum}

# ------------------------------------------------------------------ element types

"""
    TAPPElement

The element types this implementation supports. `TAPP_F16` and `TAPP_BF16` are
declared by the ABI and refused by the library, so they are absent here.
"""
const TAPPElement = Union{Float32, Float64, ComplexF32, ComplexF64}

"""
    datatype(T) -> LibTAPP.TAPP_datatype

The ABI's tag for a Julia element type.

The tags are not interchangeable across implementations by value: TBLIS swapped two
of its own between releases, which produces plausible wrong numbers and no error at
all, so they are read from the generated header rather than written down here.
"""
datatype(::Type{Float32}) = LibTAPP.TAPP_F32
datatype(::Type{Float64}) = LibTAPP.TAPP_F64
datatype(::Type{ComplexF32}) = LibTAPP.TAPP_C32
datatype(::Type{ComplexF64}) = LibTAPP.TAPP_C64

# ------------------------------------------------------------------------ errors

"""
    TAPPError <: Exception

A non-zero `TAPP_error`, carrying the library's own explanation of it.
"""
struct TAPPError <: Exception
    code::Cint
    message::String
end

Base.showerror(io::IO, e::TAPPError) = print(io, "TAPPError($(e.code)): ", e.message)

check_success(code::ABICode) = LibTAPP.TAPP_check_success(code)

"""
    explain_error(code) -> String

The library's explanation of an error code. `TAPP_explain_error` returns the length
it *would* have written, so a truncated first attempt is retried at the reported
size rather than silently cut.
"""
function explain_error(code::ABICode)
    buf = Vector{UInt8}(undef, 256)
    n = GC.@preserve buf LibTAPP.TAPP_explain_error(code, length(buf), pointer(buf))
    if n >= length(buf)
        buf = Vector{UInt8}(undef, n + 1)
        GC.@preserve buf LibTAPP.TAPP_explain_error(code, length(buf), pointer(buf))
    end
    return GC.@preserve buf unsafe_string(pointer(buf))
end

"""
    check(code)

Return `nothing` on success, throw a [`TAPPError`](@ref) otherwise.

Success is tested with `TAPP_check_success` rather than `code == 0`: upstream
`error.h` fixes only `TAPP_SUCCESS` and leaves every other value to the
implementation, so comparing against the enumerators is exactly the thing the header
tells portable callers not to do.
"""
function check(code::ABICode)
    check_success(code) && return nothing
    throw(TAPPError(Cint(code), explain_error(code)))
end

# ----------------------------------------------------------------------- handles
#
# Every TAPP handle is an `intptr_t`, so the C types provide no cross-handle safety
# whatsoever: a `TAPP_tensor_info` and a `TAPP_handle` are the same type to a C
# compiler, and passing one where the other is expected is undefined behaviour
# rather than a diagnostic. Each gets a distinct Julia type here, which puts that
# safety back.
#
# `mutable struct` plus a `finalizer` makes the header's lifetime rules hold without
# the caller thinking about them. Zeroing the field on destruction is what makes a
# second destroy -- a finalizer running after an explicit `destroy!` -- a no-op
# rather than a double free.

for (T, alias, destroy, doc) in (
        (
            :Handle, :TAPP_handle, :TAPP_destroy_handle,
            "A `TAPP_handle`, the library-level context. Create one and reuse it; [`handle`](@ref) keeps one per process.",
        ),
        (
            :Executor, :TAPP_executor, :TAPP_destroy_executor,
            "A `TAPP_executor`. **Inert in this implementation**: `TAPP_execute_product` ignores it entirely, and thread count is set with the `TENSORCONTRACT_THREADS` environment variable instead. See [`thread_setting`](@ref).",
        ),
        (
            :TensorInfo, :TAPP_tensor_info, :TAPP_destroy_tensor_info,
            "A `TAPP_tensor_info`: element type, extents and strides. Owns no data. May be destroyed while products built from it are still live -- the plan copies what it needs.",
        ),
        (
            :Product, :TAPP_tensor_product, :TAPP_destroy_tensor_product,
            "A `TAPP_tensor_product`: a planned contraction. Build once, execute many times.",
        ),
    )
    @eval begin
        @doc $doc mutable struct $T
            ptr::LibTAPP.$alias
            function $T(ptr::Integer)
                h = new(convert(LibTAPP.$alias, ptr))
                finalizer(_finalize!, h)
                return h
            end
        end
        # Destroy by raw value, so `destroy!` can clear the field *before* the call
        # and never leave a window where a live handle is reachable twice.
        _destroy(::Type{$T}, ptr) = LibTAPP.$destroy(ptr)
    end
end

const AnyHandle = Union{Handle, Executor, TensorInfo, Product}

Base.isvalid(h::AnyHandle) = h.ptr != 0

"""
    destroy!(h)

Release a handle. Idempotent, so an explicit call followed by the finalizer is safe.
"""
function destroy!(h::T) where {T <: AnyHandle}
    ptr = h.ptr
    ptr == 0 && return nothing
    h.ptr = 0
    check(_destroy(T, ptr))
    return nothing
end

# A finalizer must not throw -- there is no caller to catch it and the exception
# would surface at an unrelated point in the program. A destructor here can only
# fail on a zero handle, which the guard above already excludes, so swallowing is
# not hiding a plausible failure.
function _finalize!(h::T) where {T <: AnyHandle}
    ptr = h.ptr
    ptr == 0 && return nothing
    h.ptr = 0
    _destroy(T, ptr)
    return nothing
end

function Handle()
    out = Ref{LibTAPP.TAPP_handle}(0)
    check(LibTAPP.TAPP_create_handle(out))
    return Handle(out[])
end

function Executor()
    out = Ref{LibTAPP.TAPP_executor}(0)
    check(LibTAPP.TAPP_create_executor(out))
    return Executor(out[])
end

"""
    destroy_status(status)

Destroy a `TAPP_status`. This implementation never produces one -- upstream defines
no semantics for the object beyond its destructor -- so this exists for completeness
and for callers handed a status by some other provider.
"""
destroy_status(status::ABICode) = check(LibTAPP.TAPP_destroy_status(status))

# -------------------------------------------------------------- process context
#
# `TAPP_handle` and `TAPP_executor` carry no state that varies per call -- the handle
# is a context object and the executor is inert in this implementation -- so one of
# each per process is right, created on first use rather than in `__init__` so that
# merely loading the package touches no library state.
#
# Not thread-safe to *initialise* concurrently, which is why each is behind a lock;
# the library itself synchronises nothing, and the header says so.
#
# Public, not underscore-private, because the `TensorOperations` extension lives in
# another package and an extension cannot reach into another package's internals.

const _CONTEXT_LOCK = ReentrantLock()
const _HANDLE = Ref{Union{Nothing, Handle}}(nothing)
const _EXECUTOR = Ref{Union{Nothing, Executor}}(nothing)

"""
    handle() -> Handle

The process-wide [`Handle`](@ref), created on first use. Reuse this rather than
creating one per contraction.
"""
function handle()
    h = _HANDLE[]
    h === nothing || return h
    return lock(_CONTEXT_LOCK) do
        h = _HANDLE[]
        h === nothing || return h
        _HANDLE[] = Handle()
    end
end

"""
    executor() -> Executor

The process-wide [`Executor`](@ref), created on first use. Inert in this
implementation — see [`thread_setting`](@ref).
"""
function executor()
    e = _EXECUTOR[]
    e === nothing || return e
    return lock(_CONTEXT_LOCK) do
        e = _EXECUTOR[]
        e === nothing || return e
        _EXECUTOR[] = Executor()
    end
end

# --------------------------------------------------------------------- attributes
#
# Upstream declares these and specifies no keys, so the library exports them and
# refuses every key with `TAPP_ERROR_UNSUPPORTED`. Wrapped, and called by the test
# suite, because "declared in the header but not exported by the library" was a real
# bug here once: calling one was a link failure, the least diagnosable failure
# available.

attr_set(attr::ABICode, key::ABICode, value::Ptr{Nothing}) =
    LibTAPP.TAPP_attr_set(attr, key, value)

"""
    attr_get(attr, key) -> (code, value)

Returns the raw code rather than throwing, because every key is expected to be
refused. `TAPP_attr_get` always stores a defined value (`C_NULL`) first, so the
second element is never an uninitialised read.
"""
function attr_get(attr::ABICode, key::ABICode)
    out = Ref{Ptr{Nothing}}(Ptr{Nothing}(UInt(1)))
    code = LibTAPP.TAPP_attr_get(attr, key, out)
    return code, out[]
end

attr_clear(attr::ABICode, key::ABICode) = LibTAPP.TAPP_attr_clear(attr, key)

# ------------------------------------------------------------------ tensor infos

"""
    TensorInfo(T, extents, strides) -> TensorInfo

Describe one tensor of element type `T`.

**Strides are in elements, not bytes** -- which is what `Base.strides` returns, so a
Julia array's own strides pass through unconverted. Negative and zero strides are
both legal; a zero stride on an input is a reduction.
"""
function TensorInfo(
        ::Type{T}, extents::AbstractVector{<:Integer},
        strides::AbstractVector{<:Integer}
    ) where {T <: TAPPElement}
    length(extents) == length(strides) || throw(
        ArgumentError(
            "extents and strides disagree: $(length(extents)) vs $(length(strides))"
        )
    )
    ext = collect(Int64, extents)
    str = collect(Int64, strides)
    out = Ref{LibTAPP.TAPP_tensor_info}(0)
    GC.@preserve ext str begin
        check(
            LibTAPP.TAPP_create_tensor_info(
                out, datatype(T), length(ext), pointer(ext), pointer(str)
            )
        )
    end
    return TensorInfo(out[])
end

get_nmodes(info::TensorInfo) = Int(LibTAPP.TAPP_get_nmodes(info.ptr))

set_nmodes!(info::TensorInfo, n::Integer) = check(LibTAPP.TAPP_set_nmodes(info.ptr, n))

function get_extents(info::TensorInfo)
    out = Vector{Int64}(undef, get_nmodes(info))
    GC.@preserve out LibTAPP.TAPP_get_extents(info.ptr, pointer(out))
    return out
end

function set_extents!(info::TensorInfo, ext::AbstractVector{<:Integer})
    v = collect(Int64, ext)
    return GC.@preserve v check(LibTAPP.TAPP_set_extents(info.ptr, pointer(v)))
end

function get_strides(info::TensorInfo)
    out = Vector{Int64}(undef, get_nmodes(info))
    GC.@preserve out LibTAPP.TAPP_get_strides(info.ptr, pointer(out))
    return out
end

function set_strides!(info::TensorInfo, str::AbstractVector{<:Integer})
    v = collect(Int64, str)
    return GC.@preserve v check(LibTAPP.TAPP_set_strides(info.ptr, pointer(v)))
end

# ----------------------------------------------------------------------- products

"""
    Product(handle, opA, infoA, idxA, opB, infoB, idxB,
                    opC, infoC, idxC, opD, infoD, idxD) -> Product

Plan `D = alpha * op_A(A) * op_B(B) + beta * op_C(C)`.

Labels are per *mode* of each tensor; modes sharing a label are contracted or batched
according to where the label appears. All four infos must agree on element type --
this engine plans one element type per product.

`prec` is accepted and ignored: computation happens at storage precision.
"""
function Product(
        h::Handle,
        opA::ABICode, infoA::TensorInfo, idxA::AbstractVector{<:Integer},
        opB::ABICode, infoB::TensorInfo, idxB::AbstractVector{<:Integer},
        opC::ABICode, infoC::TensorInfo, idxC::AbstractVector{<:Integer},
        opD::ABICode, infoD::TensorInfo, idxD::AbstractVector{<:Integer};
        prec::ABICode = LibTAPP.TAPP_DEFAULT_PREC
    )
    a = collect(Int64, idxA)
    b = collect(Int64, idxB)
    c = collect(Int64, idxC)
    d = collect(Int64, idxD)
    out = Ref{LibTAPP.TAPP_tensor_product}(0)
    GC.@preserve a b c d begin
        check(
            LibTAPP.TAPP_create_tensor_product(
                out, h.ptr,
                opA, infoA.ptr, pointer(a),
                opB, infoB.ptr, pointer(b),
                opC, infoC.ptr, pointer(c),
                opD, infoD.ptr, pointer(d),
                prec
            )
        )
    end
    return Product(out[])
end

"""
    execute!(plan, exec, alpha, A, B, beta, C, D)

Run a planned product over raw pointers.

`alpha` and `beta` are `Ref`s to one scalar of the product's element type -- for
`TAPP_C64` a `ComplexF64`, not a `Float64`. `C` may be a null pointer to mean "no C
term", which the library accepts **only** with `beta == 0`; pass `D`'s own pointer to
accumulate in place.

This is the raw entry point and it validates nothing about the pointers. The caller
must keep every referent alive across the call.
"""
function execute!(
        plan::Product, exec::Executor,
        alpha::Ref{T}, A::Ptr{T}, B::Ptr{T},
        beta::Ref{T}, C::Ptr{T}, D::Ptr{T}
    ) where {T <: TAPPElement}
    # A null `status` is what the header recommends: upstream defines no semantics
    # for the object beyond its destructor, so nothing is produced to destroy.
    return GC.@preserve alpha beta check(
        LibTAPP.TAPP_execute_product(
            plan.ptr, exec.ptr, C_NULL, alpha, A, B, beta, C, D
        )
    )
end

"""
    execute_batched!(plan, exec, alpha, As, Bs, beta, Cs, Ds)

The same product over `length(As)` independent buffer sets. `alpha` and `beta` are
single scalars shared by every batch.
"""
function execute_batched!(
        plan::Product, exec::Executor,
        alpha::Ref{T}, As::Vector{Ptr{T}}, Bs::Vector{Ptr{T}},
        beta::Ref{T}, Cs::Vector{Ptr{T}}, Ds::Vector{Ptr{T}}
    ) where {T <: TAPPElement}
    n = length(As)
    (length(Bs) == n && length(Cs) == n && length(Ds) == n) ||
        throw(ArgumentError("batch pointer arrays have different lengths"))
    return GC.@preserve alpha beta As Bs Cs Ds begin
        check(
            LibTAPP.TAPP_execute_batched_product(
                plan.ptr, exec.ptr, C_NULL, n,
                alpha, pointer(As), pointer(Bs),
                beta, pointer(Cs), pointer(Ds)
            )
        )
    end
end

# ------------------------------------------------------------------ introspection

"""
    implementation_name() -> String

Which TAPP provider is linked, as the library itself reports it. Non-standard;
free-form, and not machine-parseable.
"""
implementation_name() = unsafe_string(LibTAPP.TAPP_implementation_name())

"""
    implementation_version() -> VersionNumber

The version of the loaded **shared library**, as the library itself reports it — not
the version of this package, and not the version of any header. They can differ when
the JLL and the wrapper are updated separately, and this is how you find out. It is
also the version to name when quoting a performance number.
"""
implementation_version() =
    VersionNumber(unsafe_string(LibTAPP.TAPP_implementation_version()))

"""
    header_version() -> VersionNumber

The version the *generated bindings* were produced from, out of
`TAPP_VERSION_MAJOR`/`MINOR`/`PATCH`. Disagreement with
[`implementation_version`](@ref) means the JLL moved without a regeneration, which is
exactly what `gen/generator.jl` and the `bindings` CI job exist to prevent.
"""
header_version() = VersionNumber(
    LibTAPP.TAPP_VERSION_MAJOR, LibTAPP.TAPP_VERSION_MINOR, LibTAPP.TAPP_VERSION_PATCH
)

"""
    thread_setting() -> Int

How many threads the engine will use, read the way the engine reads it.

**The engine is single-threaded by default and this is not currently settable from
Julia.** `TAPP_execute_product` ignores its executor argument, so
`TENSORCONTRACT_THREADS` is the only control, and the engine caches it in a
`OnceLock` on first use — setting `ENV["TENSORCONTRACT_THREADS"]` after the first
contraction has no effect, silently.

To use more than one thread, set it before the first contraction:

```julia
ENV["TENSORCONTRACT_THREADS"] = Sys.CPU_THREADS ÷ 2
using TensorOperations, TensorPrimitives
```

or export it before starting Julia. Results are **bitwise identical** at every thread
count — no reduction is parallelised — so this is never a correctness or accuracy
decision. Thread *scaling* is measured upstream and is strongly topology-dependent;
none of it was measured through Julia.
"""
function thread_setting()
    v = get(ENV, "TENSORCONTRACT_THREADS", "1")
    n = tryparse(Int, v)
    n === nothing && return 1
    return max(n, 1)
end

end # module TensorPrimitives
