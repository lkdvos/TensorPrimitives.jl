"""
    LibTAPP

The TAPP C ABI, one Julia function per exported symbol, transcribed from
`crates/tensorprimitives-tapp/include/tapp.h`.

This layer is deliberately thin and complete: it adds handle lifetimes and error
checking and nothing else. Anything that decides *what* to compute belongs a level
up, in `TensorPrimitives`.

The transcription is hand-written and is a third copy of the interface, after the
header and `tests/abi_layout.rs`. Three copies drift, so the test suite calls
every wrapper here and compares this package's version against the string the
loaded library reports.
"""
module LibTAPP

using tensorprimitives_tapp_jll: libtensorprimitives_tapp

export TAPPError, TAPPElement

# ---------------------------------------------------------------- enumerators
#
# Plain `Int32` constants rather than an `@enum`, matching the header: TAPP's own
# enums are anonymous and its typedefs are plain `int`, so an `@enum` here would
# invent a type the ABI does not have. Values pinned against
# `enumerator_values_match_the_upstream_headers` on the Rust side; the project has
# already been bitten once by a library that swapped two datatype tags between
# releases, which produces plausible wrong numbers and no error at all.

const TAPP_F32 = Int32(0)
const TAPP_F64 = Int32(1)
const TAPP_C32 = Int32(2)
const TAPP_C64 = Int32(3)
const TAPP_F16 = Int32(4)   # declared by the ABI, refused by this library
const TAPP_BF16 = Int32(5)  # likewise

const TAPP_IDENTITY = Int32(0)
const TAPP_CONJUGATE = Int32(1)

# Not a `pub const` on the Rust side -- it exists only in the header -- so this is
# the one value here with nothing upstream to be checked against.
const TAPP_DEFAULT_PREC = Int32(-1)

const TAPP_SUCCESS = Int32(0)
const TAPP_ERROR_NULL = Int32(1)
const TAPP_ERROR_DATATYPE = Int32(2)
const TAPP_ERROR_SHAPE = Int32(3)
const TAPP_ERROR_LABELS = Int32(4)
const TAPP_ERROR_UNSUPPORTED = Int32(5)
const TAPP_ERROR_INTERNAL = Int32(6)

"""
    TAPPElement

The element types this implementation supports. `TAPP_F16`/`TAPP_BF16` are
declared by the ABI and refused by the library, so they are absent here.
"""
const TAPPElement = Union{Float32, Float64, ComplexF32, ComplexF64}

datatype(::Type{Float32}) = TAPP_F32
datatype(::Type{Float64}) = TAPP_F64
datatype(::Type{ComplexF32}) = TAPP_C32
datatype(::Type{ComplexF64}) = TAPP_C64

# --------------------------------------------------------------------- errors

"""
    TAPPError <: Exception

A non-zero `TAPP_error`, carrying the library's own explanation of it.
"""
struct TAPPError <: Exception
    code::Int32
    message::String
end

Base.showerror(io::IO, e::TAPPError) = print(io, "TAPPError($(e.code)): ", e.message)

check_success(code::Integer) =
    @ccall libtensorprimitives_tapp.TAPP_check_success(Int32(code)::Int32)::Bool

"""
    explain_error(code) -> String

The library's explanation of an error code. `TAPP_explain_error` returns the
length it *would* have written, so a truncated first attempt is retried at the
reported size rather than silently cut.
"""
function explain_error(code::Integer)
    buf = Vector{UInt8}(undef, 256)
    n = GC.@preserve buf @ccall libtensorprimitives_tapp.TAPP_explain_error(
        Int32(code)::Int32, length(buf)::Csize_t, pointer(buf)::Ptr{UInt8}
    )::Csize_t
    if n >= length(buf)
        buf = Vector{UInt8}(undef, n + 1)
        GC.@preserve buf @ccall libtensorprimitives_tapp.TAPP_explain_error(
            Int32(code)::Int32, length(buf)::Csize_t, pointer(buf)::Ptr{UInt8}
        )::Csize_t
    end
    return GC.@preserve buf unsafe_string(pointer(buf))
end

"""
    check(code)

Return `nothing` on success, throw a [`TAPPError`](@ref) otherwise.

Success is tested with `TAPP_check_success` rather than `code == 0`: upstream
`error.h` fixes only `TAPP_SUCCESS` and leaves every other value to the
implementation, so comparing against the codes above is exactly the thing the
header tells portable callers not to do.
"""
function check(code::Integer)
    check_success(code) && return nothing
    throw(TAPPError(Int32(code), explain_error(code)))
end

# -------------------------------------------------------------------- handles
#
# Every TAPP handle is an `intptr_t`, so the C types provide no cross-handle
# safety whatsoever: a `TAPP_tensor_info` and a `TAPP_handle` are the same type to
# a C compiler, and passing one where the other is expected is undefined
# behaviour rather than a diagnostic. Each gets a distinct Julia type here, which
# puts that safety back.
#
# `mutable struct` plus a `finalizer` makes the header's lifetime rules hold
# without the caller thinking about them. Zeroing the field on destruction is what
# makes a second destroy -- a finalizer running after an explicit `destroy!` --
# a no-op rather than a double free.

for (T, destroy, doc) in (
        (:Handle, :TAPP_destroy_handle,
         "A `TAPP_handle`, the library-level context. Create one and reuse it."),
        (:Executor, :TAPP_destroy_executor,
         "A `TAPP_executor`. **Inert in this implementation**: `TAPP_execute_product` ignores it entirely, and thread count is set with the `TENSORCONTRACT_THREADS` environment variable instead."),
        (:TensorInfo, :TAPP_destroy_tensor_info,
         "A `TAPP_tensor_info`: element type, extents and strides. Owns no data. May be destroyed while products built from it are still live -- the plan copies what it needs."),
        (:Product, :TAPP_destroy_tensor_product,
         "A `TAPP_tensor_product`: a planned contraction. Build once, execute many times."),
    )
    @eval begin
        @doc $doc mutable struct $T
            ptr::Int
            function $T(ptr::Integer)
                h = new(Int(ptr))
                finalizer(_finalize!, h)
                return h
            end
        end
        # Destroy by raw value, so `destroy!` can clear the field *before* the
        # call and never leave a window where a live handle is reachable twice.
        _destroy(::Type{$T}, ptr::Int) =
            @ccall libtensorprimitives_tapp.$destroy(ptr::Int)::Int32
    end
end

const AnyHandle = Union{Handle, Executor, TensorInfo, Product}

Base.isvalid(h::AnyHandle) = h.ptr != 0

"""
    destroy!(h)

Release a handle. Idempotent, so an explicit call followed by the finalizer is
safe.
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

# ---------------------------------------------------------- creating handles

function Handle()
    out = Ref{Int}(0)
    check(@ccall libtensorprimitives_tapp.TAPP_create_handle(out::Ptr{Int})::Int32)
    return Handle(out[])
end

function Executor()
    out = Ref{Int}(0)
    check(@ccall libtensorprimitives_tapp.TAPP_create_executor(out::Ptr{Int})::Int32)
    return Executor(out[])
end

"""
    destroy_status(status)

Destroy a `TAPP_status`. This implementation never produces one -- upstream
defines no semantics for the object beyond its destructor -- so this exists for
completeness and for callers handed a status by some other provider.
"""
destroy_status(status::Integer) =
    check(@ccall libtensorprimitives_tapp.TAPP_destroy_status(Int(status)::Int)::Int32)

# ----------------------------------------------------------------- attributes
#
# Upstream declares these and specifies no keys, so the library exports them and
# refuses every key with `TAPP_ERROR_UNSUPPORTED`. Wrapped, and called by the test
# suite, because "declared in the header but not exported by the library" was a
# real bug here once: calling one was a link failure, the least diagnosable
# failure available.

attr_set(attr::Integer, key::Integer, value::Ptr{Nothing}) =
    @ccall libtensorprimitives_tapp.TAPP_attr_set(
        Int(attr)::Int, Int32(key)::Int32, value::Ptr{Nothing}
    )::Int32

"""
    attr_get(attr, key) -> (code, value)

Returns the raw code rather than throwing, because every key is expected to be
refused. `TAPP_attr_get` always stores a defined value (`C_NULL`) first, so the
second element is never an uninitialised read.
"""
function attr_get(attr::Integer, key::Integer)
    out = Ref{Ptr{Nothing}}(Ptr{Nothing}(UInt(1)))
    code = @ccall libtensorprimitives_tapp.TAPP_attr_get(
        Int(attr)::Int, Int32(key)::Int32, out::Ptr{Ptr{Nothing}}
    )::Int32
    return code, out[]
end

attr_clear(attr::Integer, key::Integer) =
    @ccall libtensorprimitives_tapp.TAPP_attr_clear(
        Int(attr)::Int, Int32(key)::Int32
    )::Int32

# ---------------------------------------------------------------- tensor infos

"""
    TensorInfo(T, extents, strides) -> TensorInfo

Describe one tensor of element type `T`.

**Strides are in elements, not bytes** -- which is what `Base.strides` returns, so
a Julia array's own strides pass through unconverted. Negative and zero strides
are both legal; a zero stride on an input is a reduction.
"""
function TensorInfo(::Type{T}, extents::AbstractVector{<:Integer},
                    strides::AbstractVector{<:Integer}) where {T <: TAPPElement}
    length(extents) == length(strides) || throw(ArgumentError(
        "extents and strides disagree: $(length(extents)) vs $(length(strides))"
    ))
    ext = collect(Int64, extents)
    str = collect(Int64, strides)
    out = Ref{Int}(0)
    GC.@preserve ext str begin
        check(@ccall libtensorprimitives_tapp.TAPP_create_tensor_info(
            out::Ptr{Int}, datatype(T)::Int32, length(ext)::Int32,
            pointer(ext)::Ptr{Int64}, pointer(str)::Ptr{Int64}
        )::Int32)
    end
    return TensorInfo(out[])
end

get_nmodes(info::TensorInfo) =
    Int(@ccall libtensorprimitives_tapp.TAPP_get_nmodes(info.ptr::Int)::Int32)

set_nmodes!(info::TensorInfo, n::Integer) = check(
    @ccall libtensorprimitives_tapp.TAPP_set_nmodes(info.ptr::Int, Int32(n)::Int32)::Int32
)

function get_extents(info::TensorInfo)
    out = Vector{Int64}(undef, get_nmodes(info))
    GC.@preserve out @ccall libtensorprimitives_tapp.TAPP_get_extents(
        info.ptr::Int, pointer(out)::Ptr{Int64}
    )::Nothing
    return out
end

function set_extents!(info::TensorInfo, ext::AbstractVector{<:Integer})
    v = collect(Int64, ext)
    GC.@preserve v check(@ccall libtensorprimitives_tapp.TAPP_set_extents(
        info.ptr::Int, pointer(v)::Ptr{Int64}
    )::Int32)
end

function get_strides(info::TensorInfo)
    out = Vector{Int64}(undef, get_nmodes(info))
    GC.@preserve out @ccall libtensorprimitives_tapp.TAPP_get_strides(
        info.ptr::Int, pointer(out)::Ptr{Int64}
    )::Nothing
    return out
end

function set_strides!(info::TensorInfo, str::AbstractVector{<:Integer})
    v = collect(Int64, str)
    GC.@preserve v check(@ccall libtensorprimitives_tapp.TAPP_set_strides(
        info.ptr::Int, pointer(v)::Ptr{Int64}
    )::Int32)
end

# -------------------------------------------------------------------- products

"""
    Product(handle, opA, infoA, idxA, opB, infoB, idxB,
                    opC, infoC, idxC, opD, infoD, idxD) -> Product

Plan `D = alpha * op_A(A) * op_B(B) + beta * op_C(C)`.

Labels are per *mode* of each tensor; modes sharing a label are contracted or
batched according to where the label appears. All four infos must agree on element
type -- this engine plans one element type per product.

`prec` is accepted and ignored: computation happens at storage precision.
"""
function Product(handle::Handle,
                 opA::Integer, infoA::TensorInfo, idxA::AbstractVector{<:Integer},
                 opB::Integer, infoB::TensorInfo, idxB::AbstractVector{<:Integer},
                 opC::Integer, infoC::TensorInfo, idxC::AbstractVector{<:Integer},
                 opD::Integer, infoD::TensorInfo, idxD::AbstractVector{<:Integer};
                 prec::Integer = TAPP_DEFAULT_PREC)
    a = collect(Int64, idxA)
    b = collect(Int64, idxB)
    c = collect(Int64, idxC)
    d = collect(Int64, idxD)
    out = Ref{Int}(0)
    GC.@preserve a b c d begin
        check(@ccall libtensorprimitives_tapp.TAPP_create_tensor_product(
            out::Ptr{Int}, handle.ptr::Int,
            Int32(opA)::Int32, infoA.ptr::Int, pointer(a)::Ptr{Int64},
            Int32(opB)::Int32, infoB.ptr::Int, pointer(b)::Ptr{Int64},
            Int32(opC)::Int32, infoC.ptr::Int, pointer(c)::Ptr{Int64},
            Int32(opD)::Int32, infoD.ptr::Int, pointer(d)::Ptr{Int64},
            Int32(prec)::Int32
        )::Int32)
    end
    return Product(out[])
end

"""
    execute!(plan, exec, alpha, A, B, beta, C, D)

Run a planned product over raw pointers.

`alpha` and `beta` are `Ref`s to one scalar of the product's element type -- for
`TAPP_C64` a `ComplexF64`, not a `Float64`. `C` may be a null pointer to mean "no
C term", which the library accepts **only** with `beta == 0`; pass `D`'s own
pointer to accumulate in place.

This is the raw entry point and it validates nothing about the pointers. The
caller must keep every referent alive across the call.
"""
function execute!(plan::Product, exec::Executor,
                  alpha::Ref{T}, A::Ptr{T}, B::Ptr{T},
                  beta::Ref{T}, C::Ptr{T}, D::Ptr{T}) where {T <: TAPPElement}
    # A null `status` is what the header recommends: upstream defines no semantics
    # for the object beyond its destructor, so nothing is produced to destroy.
    check(@ccall libtensorprimitives_tapp.TAPP_execute_product(
        plan.ptr::Int, exec.ptr::Int, C_NULL::Ptr{Int},
        alpha::Ptr{T}, A::Ptr{T}, B::Ptr{T},
        beta::Ptr{T}, C::Ptr{T}, D::Ptr{T}
    )::Int32)
end

"""
    execute_batched!(plan, exec, alpha, As, Bs, beta, Cs, Ds)

The same product over `length(As)` independent buffer sets. `alpha` and `beta` are
single scalars shared by every batch.
"""
function execute_batched!(plan::Product, exec::Executor,
                          alpha::Ref{T}, As::Vector{Ptr{T}}, Bs::Vector{Ptr{T}},
                          beta::Ref{T}, Cs::Vector{Ptr{T}},
                          Ds::Vector{Ptr{T}}) where {T <: TAPPElement}
    n = length(As)
    (length(Bs) == n && length(Cs) == n && length(Ds) == n) ||
        throw(ArgumentError("batch pointer arrays have different lengths"))
    GC.@preserve As Bs Cs Ds begin
        check(@ccall libtensorprimitives_tapp.TAPP_execute_batched_product(
            plan.ptr::Int, exec.ptr::Int, C_NULL::Ptr{Int}, Int32(n)::Int32,
            alpha::Ptr{T}, pointer(As)::Ptr{Ptr{T}}, pointer(Bs)::Ptr{Ptr{T}},
            beta::Ptr{T}, pointer(Cs)::Ptr{Ptr{T}}, pointer(Ds)::Ptr{Ptr{T}}
        )::Int32)
    end
end

# ------------------------------------------------------------------ extensions

"""
    implementation_name() -> String

Which TAPP provider is linked. Non-standard; free-form, and not machine-parseable.
"""
implementation_name() =
    unsafe_string(@ccall libtensorprimitives_tapp.TAPP_implementation_name()::Cstring)

"""
    implementation_version() -> VersionNumber

The version of the *loaded library*, which is not necessarily the version of the
header anything else was compiled against. Non-standard.
"""
implementation_version() = VersionNumber(
    unsafe_string(@ccall libtensorprimitives_tapp.TAPP_implementation_version()::Cstring)
)

end # module LibTAPP
