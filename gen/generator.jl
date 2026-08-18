# Regenerate `../src/lib/` from the `tapp.h` that `tensorprimitives_tapp_jll`
# ships. See README.md in this directory.

using Pkg

# `TAPP_INCLUDE_DIR` points the generator at a locally installed prefix instead of
# the JLL -- the `install.sh --prefix=` layout, which is how you regenerate against
# an engine you just built. Unset, the header comes from the JLL, which is the
# arrangement that matters: it is a `FileProduct`, so the declarations and the
# library implementing them come from one artifact and cannot disagree.
const OVERRIDE_INCLUDE = get(ENV, "TAPP_INCLUDE_DIR", "")

if isempty(OVERRIDE_INCLUDE)
    Pkg.activate(@__DIR__)
    Pkg.instantiate()
end

using Clang.Generators
using Clang.Generators.JLLEnvs
import Runic

cd(@__DIR__)

const OUTDIR = normpath(joinpath(@__DIR__, "..", "src", "lib"))
isdir(OUTDIR) || mkpath(OUTDIR)

function jll_include_dir()
    @eval using tensorprimitives_tapp_jll
    return normpath(joinpath(@eval(tensorprimitives_tapp_jll.artifact_dir), "include"))
end

const INCLUDE_DIR = isempty(OVERRIDE_INCLUDE) ? jll_include_dir() :
    normpath(OVERRIDE_INCLUDE)
const TAPP_H = joinpath(INCLUDE_DIR, "tapp.h")
isfile(TAPP_H) || error("no tapp.h in $(INCLUDE_DIR)")

# Every platform the JLL builds for, as `arch-os[-libc]` prefixes. Clang's own
# environment triples carry OS version suffixes (`x86_64-apple-darwin20`), so
# these are matched by prefix against `JLLEnvs.JLL_ENV_TRIPLES` below rather than
# spelled out here, which would pin us to whichever suffix Clang shipped today.
const PLATFORM_PREFIXES = [
    "x86_64-linux-gnu",
    "i686-linux-gnu",
    "aarch64-linux-gnu",
    "armv6l-linux-gnueabihf",
    "armv7l-linux-gnueabihf",
    "powerpc64le-linux-gnu",
    "riscv64-linux-gnu",
    "x86_64-linux-musl",
    "i686-linux-musl",
    "aarch64-linux-musl",
    "armv6l-linux-musleabihf",
    "armv7l-linux-musleabihf",
    "x86_64-apple-darwin",
    "aarch64-apple-darwin",
    "x86_64-unknown-freebsd",
    "aarch64-unknown-freebsd",
    "x86_64-w64-mingw32",
]

"""
Resolve each wanted platform to a triple Clang has a JLL environment for.
Returns `(resolved, unresolved)`; nothing is silently dropped.
"""
function resolve_targets(prefixes)
    resolved = String[]
    unresolved = String[]
    for p in prefixes
        hits = filter(t -> startswith(t, p), JLLEnvs.JLL_ENV_TRIPLES)
        if isempty(hits)
            push!(unresolved, p)
        else
            # Longest match wins, so `x86_64-linux-gnu` does not shadow itself
            # against a hypothetical `x86_64-linux-gnu-something`.
            push!(resolved, first(sort(hits; by = length)))
        end
    end
    return resolved, unresolved
end

const TARGETS, MISSING_TARGETS = resolve_targets(PLATFORM_PREFIXES)

isempty(MISSING_TARGETS) ||
    @warn "Clang has no JLL environment for these JLL platforms; they will be \
           covered by the pointer-width file chosen for them at run time, which \
           is asserted below for every platform that *could* be generated" MISSING_TARGETS

options = load_options(joinpath(@__DIR__, "generator.toml"))

# ------------------------------------------------------------------ generation

mktempdir() do scratch
    outputs = Dict{String, String}()   # triple => generated source
    for target in TARGETS
        @info "processing $target"
        path = joinpath(scratch, "$target.jl")
        options["general"]["output_file_path"] = path
        args = get_default_args(target)
        push!(args, "-I$INCLUDE_DIR")
        ctx = create_context([TAPP_H], args, options)
        build!(ctx)
        Runic.format_file(path; inplace = true)
        outputs[target] = read(path, String)
    end

    # Group by content. `tapp.h` declares no struct and no type whose width is not
    # expressible as one of Julia's own C aliases -- the handles are `intptr_t`, the
    # extents and strides `int64_t`, every enumerator an anonymous `enum` -- so one
    # file is expected to cover every platform. This is where that stops being an
    # assumption. (TBLIS.jl needs one file per triple because its header has structs,
    # whose field offsets the generator bakes in as integers.)
    groups = Dict{String, Vector{String}}()
    for (target, src) in outputs
        push!(get!(groups, src, String[]), target)
    end

    if length(groups) != 1
        error(
            """
            The generated bindings are not platform-independent: $(length(groups)) distinct \
            outputs over $(length(TARGETS)) targets, grouped as \
            $(sort(collect(values(groups)); by = first)).

            `src/lib/libtapp.jl` is a single file on the strength of them being identical, so \
            it cannot represent this. Something entered the header whose translation depends \
            on the target -- a struct, a `long`, a sized `enum`, a `char` whose signedness \
            matters. Emit one file per triple and add a `Base.BUILD_TRIPLET` ladder to \
            `src/LibTAPP.jl`, the way TBLIS.jl does."""
        )
    end

    src, verified = only(groups)
    dest = joinpath(OUTDIR, "libtapp.jl")
    open(dest, "w") do io
        write(io, "# This file is generated by `gen/generator.jl`. Do not edit.\n")
        write(io, "#\n")
        write(io, "# Generated from the `tapp.h` that `tensorprimitives_tapp_jll` ships, and\n")
        write(io, "# verified byte-identical for every target Clang has an environment for:\n")
        for t in sort(verified)
            write(io, "#   $t\n")
        end
        if !isempty(MISSING_TARGETS)
            write(io, "#\n")
            write(io, "# Not checkable here, because Clang ships no environment for them:\n")
            for t in sort(MISSING_TARGETS)
                write(io, "#   $t\n")
            end
        end
        write(io, "\n")
        write(io, src)
    end
    Runic.format_file(dest; inplace = true)
    @info "wrote $dest" verified_targets = length(verified) unchecked = MISSING_TARGETS
end
