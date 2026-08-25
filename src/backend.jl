# A `TensorOperations.jl` backend over the TAPP surface.
#
# The mapping is pure index bookkeeping, and that is the point: TAPP takes arbitrary
# extents and element strides plus per-operand conjugation, and the engine underneath
# contracts them in place, so nothing here permutes, copies or allocates a temporary.
# A backend over a GEMM would have to.
#
# Several of the names imported below are `TensorOperations` internals rather than
# public API -- the argument checks, the label helpers, `select_backend`. That is what
# writing a backend takes, and it is the same set the in-tree TBLIS backend imports;
# the cost is that a `TensorOperations` refactor can break this file, which its own
# test suite would catch.

using TensorOperations: TensorOperations as TO
using TensorOperations: AbstractBackend, DefaultAllocator, Index2Tuple
using TensorOperations: StridedView, isstrided
using TensorOperations: argcheck_tensorcontract, dimcheck_tensorcontract
using TensorOperations: contract_labels
using TensorOperations: select_backend, NoBackend

const SV = StridedView

"""
    TAPPBackend()

A [`TensorOperations`](https://github.com/QuantumKitHub/TensorOperations.jl) backend
that computes contractions with this package's TAPP provider.

```julia
using TensorOperations, TensorPrimitives
A = randn(8, 12); B = randn(12, 5)
@tensor backend = TAPPBackend() C[i, j] := A[i, k] * B[k, j]
```

Contractions go to the engine, which contracts strided memory in place: extents,
element strides and per-operand conjugation are handed to the ABI as they are, so the
index specification is a *relabelling* and nothing permutes, copies or allocates a
temporary.

[`tensoradd!`](@ref TensorOperations.tensoradd!) and
[`tensortrace!`](@ref TensorOperations.tensortrace!) are **delegated** to whichever
backend `select_backend` picks for the arguments, rather than computed here. TAPP can
express both -- a trace is a repeated label, an add is a contraction against a rank-0
operand -- but each is a separate correctness surface, and this engine's reason to
exist is the contraction. Delegating is not a no-op: without it `TensorOperations`
throws `Unknown backend` for anything that is not a pure contraction.

All operands must share one element type out of `Float32`, `Float64`, `ComplexF32` and
`ComplexF64`. Anything else, a non-strided array, or a conjugated output view is
refused with an `ArgumentError` rather than being passed to another backend, so a
contraction that never reached the engine cannot pass for one that did.

This backend is opt-in and is deliberately **not** registered with
`select_backend`, so nothing starts using it because it was loaded.

!!! note "Threading is off by default"
    The engine runs single-threaded unless `TENSORCONTRACT_THREADS` is set, and that
    variable is read once per process. See [`thread_setting`](@ref).
"""
struct TAPPBackend <: AbstractBackend end

# ------------------------------------------------------------------ argument checks

@noinline function throw_eltype(f, tensors)
    msg = "$f with TAPPBackend requires a common element type in " *
        "(Float32, Float64, ComplexF32, ComplexF64), got " *
        join(eltype.(tensors), ", ")
    return throw(ArgumentError(msg))
end

@noinline function throw_strided(f, tensors)
    msg = "$f with TAPPBackend requires strided arrays, got " *
        join(typeof.(tensors), ", ")
    return throw(ArgumentError(msg))
end

@noinline function throw_conj_output(f)
    msg = "$f with TAPPBackend cannot write into a conjugated view of the output"
    return throw(ArgumentError(msg))
end

function check_arguments(f, C::AbstractArray, As::AbstractArray...)
    tensors = (C, As...)
    T = eltype(C)
    (T <: TAPPElement && all(A -> eltype(A) === T, As)) || throw_eltype(f, tensors)
    all(isstrided, tensors) || throw_strided(f, tensors)
    return nothing
end

# ------------------------------------------------------- describing operands to TAPP

# The caller's flag composed with whatever the view already carries -- exclusive-or,
# not replacement, because `TensorOperations` composes them the same way when it
# absorbs a flag into a `StridedView`.
isconj(A::StridedView{T}, conjA::Bool) where {T} = T <: Complex && (conjA ⊻ (A.op === conj))

element_op(conj::Bool) = conj ? LibTAPP.TAPP_CONJUGATE : LibTAPP.TAPP_IDENTITY

# Extents and strides as the ABI wants them. **Strides are in elements**, which is
# what `Base.strides` returns, so nothing is converted. Negative strides are legal and
# are deliberately not "fixed" here: `pointer` addresses the element at the all-zero
# index, which for a negative stride is not the lowest address in the buffer, and the
# engine handles exactly that.
tapp_dims(A::StridedView) = (collect(Int64, size(A)), collect(Int64, strides(A)))

# `contract_labels` hands back one label per mode of each tensor, in that tensor's own
# mode order, which is precisely TAPP's convention -- so the index specification is a
# relabelling. It also does the three consistency checks. The labels are `Char`s there
# and `int64_t` here; only equality between them matters.
labels(ein::Tuple{Vararg{Char}}) = collect(Int64, map(Int, ein))

# ----------------------------------------------------------------------- contraction

function TO.tensorcontract!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        backend::TAPPBackend, allocator = DefaultAllocator()
    )
    check_arguments(TO.tensorcontract!, C, A, B)
    argcheck_tensorcontract(C, A, pA, B, pB, pAB)
    dimcheck_tensorcontract(C, A, pA, B, pB, pAB)

    T = eltype(C)
    einA, einB, einC = contract_labels(pA, pB, pAB)
    Av, Bv, Cv = SV(A), SV(B), SV(C)
    isconj(Cv, false) && throw_conj_output(TO.tensorcontract!)

    lenA, strideA = tapp_dims(Av)
    lenB, strideB = tapp_dims(Bv)
    lenC, strideC = tapp_dims(Cv)

    infoA = TensorInfo(T, lenA, strideA)
    infoB = TensorInfo(T, lenB, strideB)
    # C and D are the same tensor: `TensorOperations` computes `C = β*C + α*op(A)*op(B)`
    # in place, and the plan only needs one description of it.
    infoD = TensorInfo(T, lenC, strideC)

    plan = Product(
        _handle(),
        element_op(isconj(Av, conjA)), infoA, labels(einA),
        element_op(isconj(Bv, conjB)), infoB, labels(einB),
        LibTAPP.TAPP_IDENTITY, infoD, labels(einC),
        LibTAPP.TAPP_IDENTITY, infoD, labels(einC),
    )

    alpha = Ref(convert(T, α))
    beta = Ref(convert(T, β))
    # `β == 0` means "overwrite, do not read C", which is not the same as multiplying C
    # by zero: zero times an uninitialised NaN is a NaN. TAPP spells the former as a
    # null C operand, and accepts it only with `beta == 0`.
    cptr = iszero(β) ? Ptr{T}(C_NULL) : pointer(Cv)

    GC.@preserve A B C alpha beta begin
        execute!(plan, _executor(), alpha, pointer(Av), pointer(Bv), beta, cptr, pointer(Cv))
    end

    # Destroy eagerly rather than waiting for finalizers: a contraction in a hot loop
    # would otherwise accumulate handles until the next GC.
    destroy!(plan)
    destroy!(infoA)
    destroy!(infoB)
    destroy!(infoD)

    return C
end

# ------------------------------------------------------- addition and trace, delegated

@noinline function throw_no_delegate(f, backend, tensors)
    msg = "$f with $backend found no backend to delegate to for tensor types " *
        join(typeof.(tensors), ", ")
    return throw(ArgumentError(msg))
end

function delegate(f::Function, backend::TAPPBackend, tensors...)
    newbackend = select_backend(f, tensors...)
    newbackend isa NoBackend && throw_no_delegate(f, backend, tensors)
    return newbackend
end

function TO.tensoradd!(
        C::AbstractArray,
        A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::TAPPBackend, allocator = DefaultAllocator()
    )
    newbackend = delegate(TO.tensoradd!, backend, C, A)
    return TO.tensoradd!(C, A, pA, conjA, α, β, newbackend, allocator)
end

function TO.tensortrace!(
        C::AbstractArray,
        A::AbstractArray, p::Index2Tuple, q::Index2Tuple, conjA::Bool,
        α::Number, β::Number,
        backend::TAPPBackend, allocator = DefaultAllocator()
    )
    newbackend = delegate(TO.tensortrace!, backend, C, A)
    return TO.tensortrace!(C, A, p, q, conjA, α, β, newbackend, allocator)
end
