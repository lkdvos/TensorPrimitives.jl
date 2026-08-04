using Test
using TensorPrimitives
using TensorPrimitives: LibTAPP
using TensorOperations
using LinearAlgebra: norm

const ELTYPES = (Float32, Float64, ComplexF32, ComplexF64)

# Single precision carries ~7 decimal digits and these contractions sum over up to
# a few hundred terms, so the tolerance is per-eltype rather than one number.
tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 100

@testset "TensorPrimitives" begin

    @testset "identification" begin
        name = TensorPrimitives.implementation_name()
        @test occursin("tensorprimitives", name)
        # The version the *loaded library* reports. This is the check that a JLL
        # updated independently of this wrapper would fail.
        @test TensorPrimitives.implementation_version() isa VersionNumber
        @info "linked provider" name version = TensorPrimitives.implementation_version()
        # Threading is opt-in and the engine caches the setting on first use, so
        # this reports what the *process* will do, not what ENV says now.
        @test TensorPrimitives.thread_setting() >= 1
    end

    @testset "errors" begin
        @test LibTAPP.check_success(LibTAPP.TAPP_SUCCESS)
        @test !LibTAPP.check_success(LibTAPP.TAPP_ERROR_SHAPE)
        # The library explains its own codes; a non-empty string is the contract.
        @test !isempty(LibTAPP.explain_error(LibTAPP.TAPP_ERROR_SHAPE))
        @test LibTAPP.check(LibTAPP.TAPP_SUCCESS) === nothing
        e = @test_throws LibTAPP.TAPPError LibTAPP.check(LibTAPP.TAPP_ERROR_LABELS)
        @test e.value.code == LibTAPP.TAPP_ERROR_LABELS
        @test !isempty(sprint(showerror, e.value))
    end

    @testset "handles" begin
        h = LibTAPP.Handle()
        @test isvalid(h)
        # Two handles must be distinct allocations -- they were all the value 1
        # once, which made destroying both a double free.
        h2 = LibTAPP.Handle()
        @test h.ptr != h2.ptr
        @test h.ptr != 0 && h2.ptr != 0
        LibTAPP.destroy!(h2)
        @test !isvalid(h2)
        # Idempotent, so an explicit destroy followed by the finalizer is safe.
        @test LibTAPP.destroy!(h2) === nothing
        LibTAPP.destroy!(h)

        e = LibTAPP.Executor()
        @test isvalid(e)
        LibTAPP.destroy!(e)
    end

    @testset "tensor info round-trips" begin
        info = LibTAPP.TensorInfo(Float64, Int64[3, 4, 5], Int64[1, 3, 12])
        @test LibTAPP.get_nmodes(info) == 3
        @test LibTAPP.get_extents(info) == Int64[3, 4, 5]
        @test LibTAPP.get_strides(info) == Int64[1, 3, 12]
        LibTAPP.set_extents!(info, Int64[2, 4, 5])
        @test LibTAPP.get_extents(info) == Int64[2, 4, 5]
        LibTAPP.set_strides!(info, Int64[1, 2, 8])
        @test LibTAPP.get_strides(info) == Int64[1, 2, 8]
        LibTAPP.destroy!(info)

        @test_throws ArgumentError LibTAPP.TensorInfo(Float64, Int64[3, 4], Int64[1])
    end

    @testset "attributes are exported and refuse" begin
        # Upstream specifies no keys, so refusal is conformant; failing to export
        # the symbols is not, and that was the actual bug once.
        code, value = LibTAPP.attr_get(0, 0)
        @test code == LibTAPP.TAPP_ERROR_UNSUPPORTED
        # A defined value is always stored first, so a caller ignoring the code
        # does not read its own uninitialised slot.
        @test value == C_NULL
        @test LibTAPP.attr_set(0, 0, C_NULL) == LibTAPP.TAPP_ERROR_UNSUPPORTED
        @test LibTAPP.attr_clear(0, 0) == LibTAPP.TAPP_ERROR_UNSUPPORTED
    end

    # ---------------------------------------------------------------- the backend
    #
    # Every case below is checked against TensorOperations' own default backend on
    # the same inputs, which is an independent implementation rather than a
    # rewritten expectation.

    @testset "matrix product, $T" for T in ELTYPES
        A = randn(T, 6, 7)
        B = randn(T, 7, 5)
        @tensor want[i, j] := A[i, k] * B[k, j]
        @tensor backend = TAPPBackend() got[i, j] := A[i, k] * B[k, j]
        @test norm(got - want) <= tol(T) * norm(want)
    end

    @testset "multi-mode contraction and permutation, $T" for T in ELTYPES
        A = randn(T, 4, 3, 5)
        B = randn(T, 5, 3, 6)
        # Two contracted modes, and the output permuted relative to the natural
        # (open-A, open-B) order -- which is what exercises `pAB`.
        @tensor want[j, i] := A[i, k, l] * B[l, k, j]
        @tensor backend = TAPPBackend() got[j, i] := A[i, k, l] * B[l, k, j]
        @test norm(got - want) <= tol(T) * norm(want)
    end

    @testset "conjugation, $T" for T in (ComplexF32, ComplexF64)
        A = randn(T, 5, 4)
        B = randn(T, 4, 6)
        for (ca, cb) in ((true, false), (false, true), (true, true))
            fa = ca ? conj(A) : A
            fb = cb ? conj(B) : B
            @tensor want[i, j] := fa[i, k] * fb[k, j]
            got = similar(want)
            # Through the documented interface rather than by pre-conjugating, so
            # the conjA/conjB flags are what is being tested. The engine folds
            # conjugation into packing, so this must cost nothing and change
            # nothing else.
            TensorOperations.tensorcontract!(
                got, A, ((1,), (2,)), ca, B, ((1,), (2,)), cb,
                ((1,), (2,)), true, false, TAPPBackend()
            )
            @test norm(got - want) <= tol(T) * norm(want)
        end
    end

    @testset "alpha and beta, $T" for T in ELTYPES
        A = randn(T, 4, 5)
        B = randn(T, 5, 3)
        α = T <: Complex ? T(0.7, -0.3) : T(0.7)
        β = T <: Complex ? T(-0.2, 0.5) : T(-0.2)
        C0 = randn(T, 4, 3)

        want = copy(C0)
        TensorOperations.tensorcontract!(
            want, A, ((1,), (2,)), false, B, ((1,), (2,)), false,
            ((1,), (2,)), α, β
        )

        got = copy(C0)
        TensorOperations.tensorcontract!(
            got, A, ((1,), (2,)), false, B, ((1,), (2,)), false,
            ((1,), (2,)), α, β, TAPPBackend()
        )
        @test norm(got - want) <= tol(T) * norm(want)
    end

    @testset "beta = 0 does not read C, $T" for T in ELTYPES
        # `β == 0` must mean "overwrite", not "multiply by zero": a zero times an
        # uninitialised NaN is a NaN. TAPP spells this as a null C operand.
        A = randn(T, 3, 4)
        B = randn(T, 4, 2)
        @tensor want[i, j] := A[i, k] * B[k, j]
        got = fill(T(NaN), 3, 2)
        TensorOperations.tensorcontract!(
            got, A, ((1,), (2,)), false, B, ((1,), (2,)), false,
            ((1,), (2,)), true, false, TAPPBackend()
        )
        @test !any(isnan, got)
        @test norm(got - want) <= tol(T) * norm(want)
    end

    @testset "non-contiguous views, $T" for T in ELTYPES
        # Where a strides-in-elements bug would show. The views have a leading
        # stride > 1 and a non-zero offset, so both the stride vector and the base
        # pointer have to be right.
        Abig = randn(T, 12, 9)
        Bbig = randn(T, 9, 11)
        A = view(Abig, 2:2:12, 1:2:9)
        B = view(Bbig, 1:2:9, 3:7)
        @tensor want[i, j] := A[i, k] * B[k, j]
        @tensor backend = TAPPBackend() got[i, j] := A[i, k] * B[k, j]
        @test norm(got - want) <= tol(T) * norm(want)
    end

    @testset "no contracted modes: outer product, $T" for T in (Float64, ComplexF64)
        A = randn(T, 4, 3)
        B = randn(T, 5, 2)
        @tensor want[i, j, k, l] := A[i, j] * B[k, l]
        @tensor backend = TAPPBackend() got[i, j, k, l] := A[i, j] * B[k, l]
        @test norm(got - want) <= tol(T) * norm(want)
    end

    @testset "full contraction to a scalar, $T" for T in (Float64, ComplexF64)
        M = randn(T, 5, 6)
        N = randn(T, 5, 6)
        want = fill(zero(T))
        TensorOperations.tensorcontract!(
            want, M, ((), (1, 2)), false, N, ((1, 2), ()), false, ((), ()), true, false
        )
        got = fill(zero(T))
        TensorOperations.tensorcontract!(
            got, M, ((), (1, 2)), false, N, ((1, 2), ()), false, ((), ()),
            true, false, TAPPBackend()
        )
        @test abs(got[] - want[]) <= tol(T) * abs(want[])
    end

    @testset "unsupported element types are refused clearly" begin
        # `Float16` is declared by the ABI and refused by the library. The error
        # must name the element type: TensorOperations' generic fallback would say
        # "Unknown backend TAPPBackend()", which points at the wrong thing.
        A = randn(Float16, 3, 3)
        B = randn(Float16, 3, 3)
        e = @test_throws ArgumentError TensorOperations.tensorcontract!(
            similar(A), A, ((1,), (2,)), false, B, ((1,), (2,)), false,
            ((1,), (2,)), true, false, TAPPBackend()
        )
        @test occursin("Float16", e.value.msg)

        # Mixed element types, which this engine rejects because it plans one
        # element type per product.
        Af = randn(Float32, 3, 3)
        Ad = randn(Float64, 3, 3)
        e = @test_throws ArgumentError TensorOperations.tensorcontract!(
            zeros(Float64, 3, 3), Af, ((1,), (2,)), false, Ad, ((1,), (2,)), false,
            ((1,), (2,)), true, false, TAPPBackend()
        )
        @test occursin("Float32", e.value.msg)
    end
end
