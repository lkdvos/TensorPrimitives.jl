using Test
using TensorPrimitives
using TensorPrimitives: LibTAPP
using LinearAlgebra: norm

const TP = TensorPrimitives

# Single precision carries ~7 decimal digits and these contractions sum over up to
# a few hundred terms, so the tolerance is per-eltype rather than one number.
tol(::Type{T}) where {T} = sqrt(eps(real(T))) * 100

@testset "identification" begin
    name = TP.implementation_name()
    @test occursin("tensorprimitives", name)
    # The version the *loaded library* reports, against the version the committed
    # bindings were generated from. This is the check that a JLL updated without a
    # regeneration would fail -- the whole point of shipping the header as a product.
    @test TP.implementation_version() isa VersionNumber
    @test TP.implementation_version() == TP.header_version()
    @info "linked provider" name version = TP.implementation_version()
    # Threading is opt-in and the engine caches the setting on first use, so this
    # reports what the *process* will do, not what ENV says now.
    @test TP.thread_setting() >= 1
end

@testset "generated enumerators behave as the ABI's plain ints" begin
    # `tapp.h` declares every enum anonymously and every function taking one as
    # `int`, so the enumerators have to compare and convert as integers. CEnum's
    # `@cenum` does; Julia's native `@enum` does not -- `x == 0` against a code the
    # library returned is silently `false`, and `convert(Cint, x)` is a MethodError.
    # Both directions are pinned here so a switch back cannot pass quietly.
    @test LibTAPP.TAPP_SUCCESS == 0
    @test 0 == LibTAPP.TAPP_SUCCESS
    @test convert(Cint, LibTAPP.TAPP_ERROR_SHAPE) === Cint(3)
    # The datatype tag reaches the ABI by value, and a returned code compares to it.
    @test TP.datatype(Float64) == LibTAPP.TAPP_F64
    @test LibTAPP.TAPP_DEFAULT_PREC == -1
end

@testset "errors" begin
    @test TP.check_success(LibTAPP.TAPP_SUCCESS)
    @test !TP.check_success(LibTAPP.TAPP_ERROR_SHAPE)
    # The library explains its own codes; a non-empty string is the contract.
    @test !isempty(TP.explain_error(LibTAPP.TAPP_ERROR_SHAPE))
    @test TP.check(LibTAPP.TAPP_SUCCESS) === nothing
    e = @test_throws TP.TAPPError TP.check(LibTAPP.TAPP_ERROR_LABELS)
    @test e.value.code == LibTAPP.TAPP_ERROR_LABELS
    @test !isempty(sprint(showerror, e.value))
end

@testset "handles" begin
    h = TP.Handle()
    @test isvalid(h)
    # Two handles must be distinct allocations -- they were all the value 1 once,
    # which made destroying both a double free.
    h2 = TP.Handle()
    @test h.ptr != h2.ptr
    @test h.ptr != 0 && h2.ptr != 0
    TP.destroy!(h2)
    @test !isvalid(h2)
    # Idempotent, so an explicit destroy followed by the finalizer is safe.
    @test TP.destroy!(h2) === nothing
    TP.destroy!(h)

    e = TP.Executor()
    @test isvalid(e)
    TP.destroy!(e)
end

@testset "the process context is one handle, reused" begin
    # Public API rather than an internal, because the `TensorOperations` extension
    # lives in another package and reaches for exactly these.
    @test TP.handle() === TP.handle()
    @test TP.executor() === TP.executor()
    @test isvalid(TP.handle())
    @test isvalid(TP.executor())
end

@testset "tensor info round-trips" begin
    info = TP.TensorInfo(Float64, Int64[3, 4, 5], Int64[1, 3, 12])
    @test TP.get_nmodes(info) == 3
    @test TP.get_extents(info) == Int64[3, 4, 5]
    @test TP.get_strides(info) == Int64[1, 3, 12]
    TP.set_extents!(info, Int64[2, 4, 5])
    @test TP.get_extents(info) == Int64[2, 4, 5]
    TP.set_strides!(info, Int64[1, 2, 8])
    @test TP.get_strides(info) == Int64[1, 2, 8]
    TP.set_nmodes!(info, 2)
    @test TP.get_nmodes(info) == 2
    TP.destroy!(info)

    @test_throws ArgumentError TP.TensorInfo(Float64, Int64[3, 4], Int64[1])
end

@testset "status objects" begin
    # This implementation never produces a status -- upstream defines no semantics
    # for the object beyond its destructor, and the header recommends passing null.
    # The destructor exists so portable caller code links.
    @test TP.destroy_status(0) === nothing
end

@testset "plan and execute, $T" for T in (Float32, Float64, ComplexF32, ComplexF64)
    # One contraction through the raw layer, which is the path a caller planning
    # once and executing many times takes. The `TensorOperations` backend has its
    # own tests, upstream, and does not exercise this entry point.
    m, k, n = 3, 4, 2
    A = randn(T, m, k)
    B = randn(T, k, n)
    D = zeros(T, m, n)

    infoA = TP.TensorInfo(T, Int64[m, k], Int64[1, m])
    infoB = TP.TensorInfo(T, Int64[k, n], Int64[1, k])
    infoD = TP.TensorInfo(T, Int64[m, n], Int64[1, m])
    plan = TP.Product(
        TP.handle(),
        LibTAPP.TAPP_IDENTITY, infoA, Int64[2, 1],
        LibTAPP.TAPP_IDENTITY, infoB, Int64[1, 3],
        LibTAPP.TAPP_IDENTITY, infoD, Int64[2, 3],
        LibTAPP.TAPP_IDENTITY, infoD, Int64[2, 3],
    )

    alpha = Ref(one(T))
    beta = Ref(zero(T))
    GC.@preserve A B D begin
        # A null `C` with `beta == 0` means "overwrite, do not read", which is not
        # the same as multiplying an uninitialised value by zero.
        TP.execute!(
            plan, TP.executor(), alpha, pointer(A), pointer(B),
            beta, Ptr{T}(C_NULL), pointer(D)
        )
    end
    want = A * B
    @test norm(D - want) <= tol(T) * norm(want)

    TP.destroy!(plan)
    TP.destroy!(infoA)
    TP.destroy!(infoB)
    TP.destroy!(infoD)
end

@testset "batched execution, $T" for T in (Float64, ComplexF64)
    # `TAPP_execute_batched_product` is not reachable through the backend, so
    # without this it is a wrapper nothing calls -- and the claim that the bindings
    # are *complete* would be untested exactly where an unused signature is most
    # likely to be wrong.
    m, k, n, nbatch = 3, 4, 2, 3
    As = [randn(T, m, k) for _ in 1:nbatch]
    Bs = [randn(T, k, n) for _ in 1:nbatch]
    Ds = [zeros(T, m, n) for _ in 1:nbatch]

    # Column-major strides, and labels for D[i,j] = sum_k A[i,k] B[k,j].
    infoA = TP.TensorInfo(T, Int64[m, k], Int64[1, m])
    infoB = TP.TensorInfo(T, Int64[k, n], Int64[1, k])
    infoD = TP.TensorInfo(T, Int64[m, n], Int64[1, m])
    plan = TP.Product(
        TP.handle(),
        LibTAPP.TAPP_IDENTITY, infoA, Int64[2, 1],
        LibTAPP.TAPP_IDENTITY, infoB, Int64[1, 3],
        LibTAPP.TAPP_IDENTITY, infoD, Int64[2, 3],
        LibTAPP.TAPP_IDENTITY, infoD, Int64[2, 3],
    )

    alpha = Ref(one(T))
    beta = Ref(zero(T))
    GC.@preserve As Bs Ds begin
        TP.execute_batched!(
            plan, TP.executor(), alpha,
            [pointer(A) for A in As], [pointer(B) for B in Bs],
            beta, [Ptr{T}(C_NULL) for _ in 1:nbatch], [pointer(D) for D in Ds]
        )
    end

    for b in 1:nbatch
        want = As[b] * Bs[b]
        @test norm(Ds[b] - want) <= tol(T) * norm(want)
    end
    # Each batch must have used its *own* buffers, which a loop that ignored the
    # pointer arrays would not.
    @test Ds[1] != Ds[2]

    @test_throws ArgumentError TP.execute_batched!(
        plan, TP.executor(), alpha,
        [pointer(As[1])], Ptr{T}[], beta, Ptr{T}[], Ptr{T}[]
    )

    TP.destroy!(plan)
end

@testset "attributes are exported and refuse" begin
    # Upstream specifies no keys, so refusal is conformant; failing to export the
    # symbols is not, and that was the actual bug once.
    code, value = TP.attr_get(0, 0)
    @test code == LibTAPP.TAPP_ERROR_UNSUPPORTED
    # A defined value is always stored first, so a caller ignoring the code does not
    # read its own uninitialised slot.
    @test value == C_NULL
    @test TP.attr_set(0, 0, C_NULL) == LibTAPP.TAPP_ERROR_UNSUPPORTED
    @test TP.attr_clear(0, 0) == LibTAPP.TAPP_ERROR_UNSUPPORTED
end
