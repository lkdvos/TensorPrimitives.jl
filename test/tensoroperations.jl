using TensorOperations
using TensorOperations: StridedNative, StridedView
using TensorPrimitives
using LinearAlgebra
using Random
using Test

Random.seed!(1234567)

const eltypes = (Float32, Float64, ComplexF32, ComplexF64)
const tapp = TAPPBackend()
const reference = StridedNative()

# Poison the output with `NaN` whenever `β == 0`, so that a kernel computing `0 * C`
# instead of ignoring `C` is caught. TAPP spells "ignore C" as a null operand, and this
# is the test that pins it.
poison!(C) = fill!(C, convert(eltype(C), NaN))

@info "linked TAPP provider" name = TensorPrimitives.implementation_name() version =
    TensorPrimitives.implementation_version()

@testset "tensorcontract! (eltype = $T)" for T in eltypes
    # A matrix product, the multi-mode case, and the output permuted relative to the
    # natural (open-A, open-B) order, which is what exercises `pAB`.
    @testset "shapes" begin
        A = randn(T, (6, 7))
        B = randn(T, (7, 5))
        C = randn(T, (6, 5))
        Cref = copy(C)
        @test tensorcontract!(
            C, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1,), (2,)), 1, 0, tapp
        ) ≈ tensorcontract!(
            Cref, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1,), (2,)), 1, 0,
            reference
        )

        A2 = randn(T, (4, 3, 5))
        B2 = randn(T, (5, 3, 6))
        @tensor want[j, i] := A2[i, k, l] * B2[l, k, j]
        @tensor backend = tapp got[j, i] := A2[i, k, l] * B2[l, k, j]
        @test got ≈ want
    end

    @testset "conjugation" begin
        # Through the flags rather than by pre-conjugating, so the flags are what is
        # tested. The engine folds conjugation into packing, so it must change nothing else.
        A = randn(T, (5, 4))
        B = randn(T, (4, 6))
        for conjA in (false, true), conjB in (false, true)
            C = randn(T, (5, 6))
            Cref = copy(C)
            @test tensorcontract!(
                C, A, ((1,), (2,)), conjA, B, ((1,), (2,)), conjB, ((1,), (2,)), 1, 0, tapp
            ) ≈ tensorcontract!(
                Cref, A, ((1,), (2,)), conjA, B, ((1,), (2,)), conjB, ((1,), (2,)), 1, 0,
                reference
            )
        end

        # An `Adjoint` input already carries a conjugation, which has to combine with the
        # caller's flag rather than replace it.
        Aadj = adjoint(randn(T, (4, 5)))
        B = randn(T, (4, 6))
        for conjA in (false, true)
            C = randn(T, (5, 6))
            Cref = copy(C)
            @test tensorcontract!(
                C, Aadj, ((1,), (2,)), conjA, B, ((1,), (2,)), false, ((1,), (2,)), 1, 0,
                tapp
            ) ≈ tensorcontract!(
                Cref, Aadj, ((1,), (2,)), conjA, B, ((1,), (2,)), false, ((1,), (2,)), 1, 0,
                reference
            )
        end
    end

    @testset "alpha and beta" begin
        A = randn(T, (4, 5))
        B = randn(T, (5, 3))
        for (α, β) in ((1, 0), (randn(T), 0), (randn(T), randn(T)), (randn(T), 1))
            C = randn(T, (4, 3))
            Cref = copy(C)
            # `β == 0` must mean "overwrite", not "multiply by zero".
            iszero(β) && (poison!(C); poison!(Cref))
            @test tensorcontract!(
                C, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1,), (2,)), α, β, tapp
            ) ≈ tensorcontract!(
                Cref, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1,), (2,)), α, β,
                reference
            )
            iszero(β) && @test !any(isnan, C)
        end
    end

    @testset "non-contiguous views" begin
        # Where a strides-in-elements or base-pointer bug would show, and nothing else
        # would: a leading stride > 1 and a non-zero offset.
        A = view(randn(T, (12, 9)), 2:2:12, 1:2:9)
        B = view(randn(T, (9, 11)), 1:2:9, 3:7)
        C = randn(T, (6, 5))
        Cref = copy(C)
        @test tensorcontract!(
            C, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1,), (2,)), 1, 0, tapp
        ) ≈ tensorcontract!(
            Cref, A, ((1,), (2,)), false, B, ((1,), (2,)), false, ((1,), (2,)), 1, 0,
            reference
        )
    end

    @testset "degenerate index patterns" begin
        # No contracted modes at all: an outer product.
        A = randn(T, (4, 3))
        B = randn(T, (5, 2))
        @tensor want[i, j, k, l] := A[i, j] * B[k, l]
        @tensor backend = tapp got[i, j, k, l] := A[i, j] * B[k, l]
        @test got ≈ want

        # Every mode contracted: a scalar output, which is a rank-0 tensor here.
        M = randn(T, (5, 6))
        N = randn(T, (5, 6))
        want = fill(zero(T))
        got = fill(zero(T))
        tensorcontract!(
            want, M, ((), (1, 2)), false, N, ((1, 2), ()), false, ((), ()), 1, 0, reference
        )
        tensorcontract!(
            got, M, ((), (1, 2)), false, N, ((1, 2), ()), false, ((), ()), 1, 0, tapp
        )
        @test got[] ≈ want[]
    end
end

# `tensoradd!` and `tensortrace!` are delegated rather than computed by the engine, and
# delegation is not a no-op: without those methods the generic fallback in `interface.jl`
# throws "Unknown backend" for anything that is not a pure contraction, so a permutation
# -- which is a `tensoradd!` -- and every trace failed. See A64 upstream.
@testset "delegated operations (eltype = $T)" for T in eltypes
    @testset "tensoradd!" begin
        A = randn(T, (3, 5, 4, 6))
        p = ((3, 1), (4, 2))
        for conjA in (false, true), (α, β) in ((1, 0), (randn(T), randn(T)))
            C = randn(T, (4, 3, 6, 5))
            Cref = copy(C)
            iszero(β) && (poison!(C); poison!(Cref))
            @test tensoradd!(C, A, p, conjA, α, β, tapp) ≈
                tensoradd!(Cref, A, p, conjA, α, β, reference)
        end
    end

    @testset "tensortrace!" begin
        A = randn(T, (5, 3, 5, 4))
        p = ((2,), (4,))
        q = ((1,), (3,))
        for conjA in (false, true), (α, β) in ((1, 0), (randn(T), randn(T)))
            C = randn(T, (3, 4))
            Cref = copy(C)
            iszero(β) && (poison!(C); poison!(Cref))
            @test tensortrace!(C, A, p, q, conjA, α, β, tapp) ≈
                tensortrace!(Cref, A, p, q, conjA, α, β, reference)
        end
    end

    @testset "a permutation through @tensor" begin
        # The expression that used to throw "Unknown backend".
        A = randn(T, (4, 6))
        @tensor want[j, i] := A[i, j]
        @tensor backend = tapp got[j, i] := A[i, j]
        @test got ≈ want
    end

    @testset "a trace through @tensor" begin
        A = randn(T, (5, 3, 5))
        @tensor want[j] := A[i, j, i]
        @tensor backend = tapp got[j] := A[i, j, i]
        @test got ≈ want
    end

    @testset "a network mixing all three" begin
        # A contraction and a trace in one expression, which is what a user actually
        # writes and what no single-operation test covers.
        A = randn(T, (4, 3, 3))
        B = randn(T, (4, 5))
        @tensor want[j] := A[i, k, k] * B[i, j]
        @tensor backend = tapp got[j] := A[i, k, k] * B[i, j]
        @test got ≈ want
    end
end

@testset "unsupported arguments are refused, not delegated" begin
    # `Float16` is declared by the ABI and refused by the library. The message must name
    # the element type: the generic fallback would say "Unknown backend TAPPBackend()",
    # which points at the wrong thing. And a contraction that never reached the engine
    # must not be able to pass for one that did, so this throws rather than delegating.
    A = randn(Float16, (3, 3))
    e = @test_throws ArgumentError tensorcontract!(
        similar(A), A, ((1,), (2,)), false, A, ((1,), (2,)), false, ((1,), (2,)), 1, 0, tapp
    )
    @test occursin("Float16", e.value.msg)

    # Mixed element types, which this engine rejects because it plans one element type
    # per product.
    Af = randn(Float32, (3, 3))
    Ad = randn(Float64, (3, 3))
    e = @test_throws ArgumentError tensorcontract!(
        zeros(Float64, (3, 3)), Af, ((1,), (2,)), false, Ad, ((1,), (2,)), false,
        ((1,), (2,)), 1, 0, tapp
    )
    @test occursin("Float32", e.value.msg)

    # A conjugated output view: TAPP has no flag for the destination, and silently
    # writing unconjugated values into it would be wrong. Note this has to be built as a
    # `StridedView` -- `conj` of an `Array` materialises a new array and carries no flag,
    # so it is not the case being guarded against.
    C = conj(StridedView(randn(ComplexF64, (3, 3))))
    A = randn(ComplexF64, (3, 3))
    @test_throws ArgumentError tensorcontract!(
        C, A, ((1,), (2,)), false, A, ((1,), (2,)), false, ((1,), (2,)), 1, 0, tapp
    )
end
