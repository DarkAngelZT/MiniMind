# MiniMind MNN vendor manifest

- Upstream version: MNN 3.6.1
- Source archive SHA-256: `0DA429162604201F5861CC1D780C3D782488A11D36DA37D64129911B73E96773`
- Upstream license: `LICENSE.txt`

Included:

- CPU backend for x86/x64, ARM and ARM82
- Express, Module and tensor graph APIs
- shape and geometry transforms
- complete MNN Train gradient rules
- `NN::Linear`, convolution, normalization and dropout support
- SGD, ADAM, parameter serialization and inference transformer support
- generated schemas plus the required FlatBuffers and half headers

Excluded:

- GPU, NPU and vendor accelerator backends
- converter frontends and protobuf
- Python, Java/JNI, apps, demos, datasets, upstream tests and benchmarks
- OpenCV/audio helper APIs, LLM and diffusion applications

The vendored defaults build one static CPU library with Express and Train.
MiniMind may still override the `MNN_*` CMake cache options before calling
`add_subdirectory(3rd_party/MNN)`.
