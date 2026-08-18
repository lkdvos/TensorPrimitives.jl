"""
    TensorPrimitives.LibTAPP

The TAPP C ABI, one Julia function per exported symbol, **generated** from the
`tapp.h` that `tensorprimitives_tapp_jll` ships as a `FileProduct`.

Nothing in here is hand-written, and nothing in here decides anything: the layer
above, [`TensorPrimitives`](@ref), adds handle lifetimes, error checking and the
element-type mapping. Reach into this module when you want the ABI exactly as the
header declares it — every enumerator, including the aliases and the two datatypes
(`TAPP_F16`, `TAPP_BF16`) that the ABI declares and this library refuses.

The bindings used to be a hand transcription, which made three copies of the
interface: the header, the Rust side's `abi_layout.rs`, and this. Regenerate with
`gen/generator.jl`; CI fails if the committed output is not what the pinned
generator produces.
"""
module LibTAPP

using tensorprimitives_tapp_jll: libtensorprimitives_tapp

# One file covers every platform. `tapp.h` declares no struct, and its handle
# typedefs are `intptr_t`, which Clang translates to `Cptrdiff_t` -- an alias Julia
# resolves per platform -- so the generated source is platform-independent rather
# than merely happening to match. `gen/generator.jl` asserts that across every
# target Clang has an environment for, and the file's own header lists them.
include("lib/libtapp.jl")

end # module LibTAPP
