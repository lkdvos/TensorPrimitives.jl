# Generating the raw bindings

`../src/lib/` is generated from the `tapp.h` that `tensorprimitives_tapp_jll`
ships, and is committed. To regenerate:

```bash
julia --project generator.jl
```

Then check the result is what the pinned toolchain produces:

```bash
git diff --exit-code ../src/lib
```

A non-empty diff after a header change is the expected outcome; a non-empty diff
without one means the `Clang` or `Runic` version moved, and `Project.toml` here
pins both so that cannot happen silently.

## Why there is one file for seventeen platforms

The JLL ships 17 platforms and `src/lib/` holds one file. `tapp.h` declares no
struct, and no type whose translation depends on the target: the handle typedefs are
`intptr_t`, which Clang renders as `Cptrdiff_t` — an alias Julia resolves per
platform — extents and strides are `int64_t`, and every enumerator is an anonymous
`enum`. So the generated source is platform-independent rather than merely happening
to match on the machine it was run on. TBLIS.jl needs one file per triple because its
header has structs, whose field offsets the generator bakes in as integers.

`generator.jl` does not assume that: it generates for **every** JLL platform Clang
has an environment for, groups the outputs by content, and fails unless they are
all identical. So the single committed file is a measured result, and the day a
`long`, a sized `enum` or a struct enters the header, the generator says so
instead of quietly shipping bindings that are wrong for some platform.
`verified_targets.txt` records which targets that was established over, including
the four the JLL ships that Clang has no environment for.

## Two ways to run it

| | command | what it establishes |
|---|---|---|
| **Full sweep** | `julia --project generator.jl` | that one file covers every platform. Rewrites `verified_targets.txt`. Pulls a GCCBootstrap sysroot per target — ~18 GB in total, so expect a long first run |
| **Restricted** | `TAPP_GEN_TARGETS=x86_64-linux-gnu,i686-linux-gnu,x86_64-w64-mingw32 julia --project generator.jl` | only that the committed bindings have not drifted from the header. Leaves `verified_targets.txt` alone |

CI runs the restricted form, because a runner cannot afford the sysroots. The target
list lives in `verified_targets.txt` rather than in the generated file's own header
for exactly this reason: a restricted sweep then reproduces `src/lib/libtapp.jl` byte
for byte, so the cheap check stays meaningful, and only a full sweep can widen or
narrow the record.
