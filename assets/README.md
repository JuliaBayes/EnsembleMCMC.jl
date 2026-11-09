# Logo

`logo.svg` is the source artwork. The transparent PNG exports have square sizes
of 96, 192, 256, 512, and 1024 pixels. The artwork uses the
[Julia palette](https://github.com/JuliaLang/julia-logo-graphics) and the
repository's [Apache 2.0 license](../LICENSE.md).

Three coloured walkers explore a shared probability landscape. The dots below
them represent samples. The logo has no border or background.

The README and documentation display the SVG at 128 pixels. The documentation
build copies it into `docs/src/assets/`.

Regenerate a PNG with [librsvg](https://gitlab.gnome.org/GNOME/librsvg):

```sh
rsvg-convert --width 512 --height 512 --output assets/logo-512.png assets/logo.svg
```
