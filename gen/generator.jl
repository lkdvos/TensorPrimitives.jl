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

const ALL_TARGETS, MISSING_TARGETS = resolve_targets(PLATFORM_PREFIXES)

# `TAPP_GEN_TARGETS` restricts the sweep to a comma-separated subset, matched as
# prefixes. The point of sweeping every platform is to *prove* that one file covers
# them all, and that proof needs the whole set -- but each target pulls a
# GCCBootstrap sysroot, which is ~18 GB across all thirteen, so CI cannot afford it
# and runs a subset purely as a drift check. Only a full sweep is allowed to write
# the record in `verified_targets.txt`, so a restricted run cannot quietly narrow it.
const TARGET_FILTER = get(ENV, "TAPP_GEN_TARGETS", "")
const RESTRICTED = !isempty(TARGET_FILTER)

const TARGETS = if RESTRICTED
    wanted = strip.(split(TARGET_FILTER, ","))
    picked = filter(t -> any(w -> startswith(t, w), wanted), ALL_TARGETS)
    isempty(picked) && error("TAPP_GEN_TARGETS=$TARGET_FILTER matched none of $ALL_TARGETS")
    picked
else
    ALL_TARGETS
end

if RESTRICTED
    @warn "Restricted sweep: this checks the committed bindings for drift but does \
           NOT establish that one file covers every platform. Run with \
           TAPP_GEN_TARGETS unset to renew that claim." targets = TARGETS
end

isempty(MISSING_TARGETS) ||
    @warn "Clang has no JLL environment for these JLL platforms, so they are covered \
           by inference from the others rather than by generation" MISSING_TARGETS

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
        write(io, "# Generated from the `tapp.h` that `tensorprimitives_tapp_jll` ships. One file\n")
        write(io, "# covers every platform; `gen/verified_targets.txt` records the targets it was\n")
        write(io, "# checked against, and `gen/README.md` says why it holds.\n")
        write(io, "#\n")
        write(io, "# The target list deliberately lives outside this file, so that a restricted\n")
        write(io, "# sweep still reproduces it byte for byte and a drift check stays meaningful.\n")
        write(io, "\n")
        write(io, src)
    end
    Runic.format_file(dest; inplace = true)

    if RESTRICTED
        @info "wrote $dest (restricted sweep; verified_targets.txt left alone)" checked = length(verified)
    else
        record = joinpath(@__DIR__, "verified_targets.txt")
        open(record, "w") do io
            write(io, "# Written by gen/generator.jl on a full sweep. Every target below produced\n")
            write(io, "# byte-identical bindings, which is why ../src/lib/libtapp.jl is one file.\n")
            write(io, "\n")
            write(io, "# Generated and compared:\n")
            for t in sort(verified)
                write(io, "$t\n")
            end
            if !isempty(MISSING_TARGETS)
                write(io, "\n")
                write(io, "# Shipped by the JLL, but Clang has no environment for them, so these are\n")
                write(io, "# covered by inference from the above rather than by generation:\n")
                for t in sort(MISSING_TARGETS)
                    write(io, "# $t\n")
                end
            end
        end
        @info "wrote $dest and $record" verified = length(verified) inferred = MISSING_TARGETS
    end
end
