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

## Why there are fewer files than platforms

The JLL ships 17 platforms. `tapp.h` declares no type whose width varies other
than through the pointer size — the handle typedefs are `intptr_t`, extents and
strides are `int64_t`, and every enumerator is an anonymous `enum` — so the
generated bindings differ between 64-bit and 32-bit targets and nowhere else.

`generator.jl` does not assume that: it generates for **every** JLL platform
Clang has an environment for, groups the outputs by content, and fails if the
grouping is not exactly the pointer-width split. So the two committed files are
a measured result, and the day a `long` or a sized `enum` enters the header, the
generator says so instead of quietly shipping bindings that are wrong for some
platform.
