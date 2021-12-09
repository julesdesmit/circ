# zsharp (nee zok07) interpreter quickstart

**WARNING** this interpreter is still experimental! When things break, please
tell me about them :)

## building

1. see `scripts/dependencies_*` for info on installing deps. Note that M1 macs
   will not yet work with the homebrew instructions because the coin-or build
   from homebrew doesn't work.

2. circ uses some experimental APIs, so you'll need rust nightly!

3. To build the zok interpreter cli, `cargo build --release --example zoki`

## running

After building as above, `target/release/examples/zoki` will have been
generated. This executable takes one argument, the name of a .zok file.
Absolute and relative paths are both OK:

    target/release/examples/zoki /tmp/foo.zok
    target/release/examples/zoki ../../path/to/somewhere/else.zok

You may want to set the `RUST_LOG` environment variable to see more info
about the typechecking and interpreting process:

    RUST_LOG=debug target/release/examples/zoki /tmp/foo.zok
