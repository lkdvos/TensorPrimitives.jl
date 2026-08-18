# Changelog

Notable changes to `TensorPrimitives.jl`. This package versions **independently of
the engine it wraps**: it is bounded by a `[compat]` entry on
`tensorprimitives_tapp_jll`, and `implementation_version()` reports the version of
the shared library actually loaded, which is the number to quote when describing
performance.

## 0.1.0 — unreleased

First release. Split out of
[`tensorprimitives-rs`](https://github.com/lkdvos/tensorprimitives-rs), where it
lived as `julia/TensorPrimitives` and shared that repository's version.

### Added

* **A complete wrapper of the TAPP C ABI** over `tensorprimitives_tapp_jll`: one
  function per exported symbol, handles as distinct Julia types with finalizers and
  an idempotent `destroy!`, error codes as `TAPPError` with the library's own
  explanation text, and `implementation_name`/`implementation_version` so a
  header/library mismatch is discoverable rather than latent.
* **Generated raw bindings** in `src/lib/`, produced by `gen/generator.jl` from the
  `tapp.h` the JLL ships as a `FileProduct`, with the generator toolchain pinned.
  CI regenerates and fails on a diff, so the bindings cannot drift from the header.
  Before this they were a hand transcription — a third copy of the interface, after
  the header itself and the Rust side's `abi_layout.rs`. Two things fell out of
  generating rather than assuming: **one file covers all 17 platforms**, because the
  header declares no struct and its `intptr_t` handles land on `Cptrdiff_t`, which
  Julia resolves per platform (the generator proves this and fails if it stops being
  true); and the enumerators are CEnum's `@cenum` rather than Julia's native `@enum`,
  because every enum in the header is anonymous and every function taking one declares
  it `int`, so native enum values cannot be passed to the functions they exist for and
  compare silently `false` against a code the library returned.
* **A documented process context.** `handle()` and `executor()` create one
  `TAPP_handle` and one `TAPP_executor` per process, on first use rather than in
  `__init__`, so loading the package touches no library state. They are public
  because the `TensorOperations` extension lives in another package and cannot
  reach into internals.

### Changed

* **The `TensorOperations` backend is no longer here.** It is an extension *in
  `TensorOperations.jl`*, which is where that ecosystem put the equivalent TBLIS
  bridge in v5.8.0, and the backend type is `TensorOperations.TAPPBackend`. So this
  package has one dependency and no knowledge of `TensorOperations`, and a user
  gets the backend by loading both.
* **`tensoradd!` and `tensortrace!` now work** through the backend, by delegation.
  The in-tree version implemented only `tensorcontract!` and was documented as
  falling through to `TensorOperations`' own backends for the other two; it did not
  — `TensorOperations` throws `Unknown backend` for any operation a selected backend
  does not implement, so `@tensor backend = TAPPBackend() C[i, j] := A[j, i]` had
  never worked. See A64 upstream.

### Known limitations

* **Threading is not reachable from Julia.** `TAPP_execute_product` ignores its
  executor argument, so the only control is `TENSORCONTRACT_THREADS`, which the
  engine reads once per process. `thread_setting()` reports what the process will
  do. Results are bitwise identical at every thread count, so this is never a
  correctness decision.
* **No performance number is measured through Julia.** The engine's own numbers are
  in the upstream `CHANGELOG.md`, per component, and none of them was taken through
  this wrapper.
