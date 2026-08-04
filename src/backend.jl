# A TensorOperations.jl backend over the TAPP surface.
#
# The mapping is pure index bookkeeping, and that is the point: TAPP takes
# arbitrary extents and element strides plus per-operand conjugation, and the
# engine underneath contracts them in place, so nothing here permutes, copies or
# allocates a temporary. A backend over a GEMM would have to.

using TensorOperations: TensorOperations, Index2Tuple
using StridedViews: StridedViews, StridedView

"""
    TAPPBackend()

A [`TensorOperations`](https://github.com/QuantumKitHub/TensorOperations.jl)
backend that computes contractions with this package's TAPP provider.

```julia
using TensorOperations, TensorPrimitives
A = randn(8, 12); B = randn(12, 5)
@tensor backend = TAPPBackend() C[i, j] := A[i, k] * B[k, j]
```

Only [`tensorcontract!`](@ref TensorOperations.tensorcontract!) is implemented.
`tensoradd!` and `tensortrace!` fall through to `TensorOperations`' own backends
rather than being routed here: TAPP can express both -- a trace is a repeated
label, an add is a contraction against a rank-0 operand -- but each is a
correctness surface that should be added with tests rather than acquired by
accident.

This backend is opt-in and is deliberately **not** registered with
`select_backend`, so nothing starts using it because it was loaded.

!!! note "Threading is off by default"
    The engine runs single-threaded unless `TENSORCONTRACT_THREADS` is set, and
    that variable is read once per process. See
    [`TensorPrimitives.thread_setting`](@ref).
"""
struct TAPPBackend <: TensorOperations.AbstractBackend end

# `op` is applied elementwise by StridedViews, so for the scalar element types
# this package supports, `adjoint` is conjugation and `transpose` is a no-op.
_isconj(::typeof(identity)) = false
_isconj(::typeof(conj)) = true
_isconj(::typeof(adjoint)) = true
_isconj(::typeof(transpose)) = false

"""
    _describe(x, conjugate) -> (pointer, extents, strides, conj)

Everything TAPP needs about one operand. The conjugation flag is the caller's
`conjA`/`conjB` combined with whatever the view already carries -- exclusive-or,
not replacement, because `TensorOperations` composes them the same way when it
absorbs a flag into a `StridedView`.
"""
function _describe(x::AbstractArray{T}, conjugate::Bool) where {T}
    sv = x isa StridedView ? x : StridedView(x)
    # The base pointer must address the element at the all-zero index, which is
    # what `offset` is measured from. Strides may be negative, in which case that
    # element is not the lowest address in the buffer -- the engine handles this,
    # so it must not be "fixed" here.
    ptr = pointer(parent(sv)) + StridedViews.offset(sv) * sizeof(T)
    return (
        convert(Ptr{T}, ptr),
        collect(Int64, size(sv)),
        collect(Int64, strides(sv)),
        conjugate ⊻ _isconj(sv.op),
    )
end

"""
    _labels(pA, pB, pAB, nA, nB, nC) -> (idxA, idxB, idxC)

Translate `TensorOperations`' permutation specification into TAPP index labels.

`pA = (codA, contA)` and `pB = (contB, codB)` name the modes of `A` and `B` that
are open and contracted, with `contA` and `contB` in matching order. The
intermediate result has the open `A` modes followed by the open `B` modes, and
`pAB` permutes that into `C`'s mode order.

So the labels are: `1:ncont` for the contracted modes, then the open `A` modes,
then the open `B` modes; `C` takes the same labels through `pAB`. Every label
array is indexed by *mode position within its own tensor*, which is what TAPP
wants and what makes this a relabelling rather than a permutation.
"""
function _labels(pA::Index2Tuple, pB::Index2Tuple, pAB::Index2Tuple,
                 nA::Int, nB::Int, nC::Int)
    codA, contA = pA
    contB, codB = pB
    ncont = length(contA)
    length(contB) == ncont || throw(ArgumentError(
        "pA and pB disagree on the number of contracted modes: $ncont vs $(length(contB))"
    ))
    noa, nob = length(codA), length(codB)
    (noa + ncont == nA && nob + ncont == nB) || throw(ArgumentError(
        "pA/pB do not cover every mode of A ($nA) and B ($nB)"
    ))
    noa + nob == nC || throw(ArgumentError(
        "the open modes ($noa + $nob) do not match C's rank ($nC)"
    ))

    idxA = zeros(Int64, nA)
    idxB = zeros(Int64, nB)
    idxC = zeros(Int64, nC)
    for (k, m) in enumerate(contA)
        idxA[m] = k
    end
    for (j, m) in enumerate(codA)
        idxA[m] = ncont + j
    end
    for (k, m) in enumerate(contB)
        idxB[m] = k
    end
    for (j, m) in enumerate(codB)
        idxB[m] = ncont + noa + j
    end
    # C's mode i holds intermediate index `perm[i]`, whose label is `ncont + it`.
    for (i, q) in enumerate((pAB[1]..., pAB[2]...))
        idxC[i] = ncont + q
    end
    return idxA, idxB, idxC
end

"""
The fallback for anything the engine cannot compute.

Without it, `TensorOperations`' generic method reports `Unknown backend
TAPPBackend()`, which is misleading in both directions: the backend is known, and
the element type is the problem. Two cases reach here — an element type outside
[`LibTAPP.TAPPElement`](@ref) (`Float16` and `BFloat16` are declared by the ABI and
refused by the library; `BigFloat` and friends are not in the ABI at all), and
operands whose element types differ from each other, which this engine rejects
because it plans one element type per product.
"""
function TensorOperations.tensorcontract!(
        C::AbstractArray, A::AbstractArray, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray, pB::Index2Tuple, conjB::Bool, pAB::Index2Tuple,
        α::Number, β::Number,
        ::TAPPBackend, allocator = TensorOperations.DefaultAllocator()
    )
    types = (eltype(C), eltype(A), eltype(B))
    throw(ArgumentError("""
    TAPPBackend cannot contract element types $(types).
    This engine supports Float32, Float64, ComplexF32 and ComplexF64, and requires \
    all operands to share one element type. Drop `backend = TAPPBackend()` to use \
    TensorOperations' own backends instead."""))
end

function TensorOperations.tensorcontract!(
        C::AbstractArray{T},
        A::AbstractArray{T}, pA::Index2Tuple, conjA::Bool,
        B::AbstractArray{T}, pB::Index2Tuple, conjB::Bool,
        pAB::Index2Tuple,
        α::Number, β::Number,
        ::TAPPBackend, allocator = TensorOperations.DefaultAllocator()
    ) where {T <: LibTAPP.TAPPElement}
    pa, exta, stra, ca = _describe(A, conjA)
    pb, extb, strb, cb = _describe(B, conjB)
    pd, extd, strd, cd = _describe(C, false)
    cd && throw(ArgumentError("the output array must not be a conjugated view"))

    idxA, idxB, idxC = _labels(pA, pB, pAB, ndims(A), ndims(B), ndims(C))

    infoA = LibTAPP.TensorInfo(T, exta, stra)
    infoB = LibTAPP.TensorInfo(T, extb, strb)
    infoD = LibTAPP.TensorInfo(T, extd, strd)

    op(c) = c ? LibTAPP.TAPP_CONJUGATE : LibTAPP.TAPP_IDENTITY

    plan = LibTAPP.Product(
        _handle(),
        op(ca), infoA, idxA,
        op(cb), infoB, idxB,
        # C and D are the same tensor: TensorOperations computes
        # `C = β*C + α*op(A)*op(B)` in place.
        LibTAPP.TAPP_IDENTITY, infoD, idxC,
        LibTAPP.TAPP_IDENTITY, infoD, idxC,
    )

    alpha = Ref(convert(T, α))
    beta = Ref(convert(T, β))
    # `β == 0` means "overwrite, do not read C", which is not the same as
    # multiplying C by zero: a zero times an uninitialised NaN is a NaN. TAPP
    # spells that as a null C operand, which it accepts only with `beta == 0`.
    cptr = iszero(β) ? Ptr{T}(C_NULL) : pd

    GC.@preserve A B C alpha beta begin
        LibTAPP.execute!(plan, _executor(), alpha, pa, pb, beta, cptr, pd)
    end

    # Destroy eagerly rather than waiting for a finalizer: a contraction in a hot
    # loop would otherwise accumulate handles until the next GC.
    LibTAPP.destroy!(plan)
    LibTAPP.destroy!(infoA)
    LibTAPP.destroy!(infoB)
    LibTAPP.destroy!(infoD)

    return C
end
