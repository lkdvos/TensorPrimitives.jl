using SafeTestsets

@time @safetestset "wrapper" begin
    include("wrapper.jl")
end

@time @safetestset "TensorOperations backend" begin
    include("tensoroperations.jl")
end

@time @safetestset "Aqua" begin
    include("aqua.jl")
end
