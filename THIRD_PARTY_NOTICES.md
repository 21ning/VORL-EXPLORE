# Third-party acknowledgements

Animation uses the installed [Pogema](https://github.com/Cognitive-AI-Systems/pogema)
1.1.1 `AnimationMonitor` and SVG primitives. The playback adapter adds team-shared
map visibility; it does not vendor or copy the Pogema renderer. CairoSVG rasterizes
native frames and Pillow encodes the GIF. These dependencies retain their own
upstream licenses.

The EPOM encoder topology in `vorl/policy.py` is adapted from the supplied
reference project and its upstream [When to Switch implementation](https://github.com/Cognitive-AI-Systems/when-to-switch).
The original Sample Factory model core is used as an installed dependency, not
copied into this repository. The external EPOM checkpoint is downloaded from
the upstream release and is not included in Git.

Upstream When to Switch license notice:

MIT License

Copyright (c) 2023 Alexey Skrynnik, Anton Andreychuk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

This notice applies to the attributed third-party material. No project-wide
license is assigned to the remaining code by this notice.
