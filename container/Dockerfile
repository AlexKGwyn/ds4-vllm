# DeepSeek-V4-Flash — patched vLLM engine for AMD gfx1151 (Strix Halo), TP=2.
#
# This rebuilds the hand-patched serving engine as a THIN layer over kyuz0's
# TheRock/ROCm gfx1151 base image — the exact base the patches were developed
# against. The base already contains the full ROCm 7 + PyTorch + vLLM build
# (vLLM commit 470229c); this file applies the DeepSeek-V4 / DSpark patch
# (patches/vllm-upstream.patch, 37 files), overlays the 18 new files from
# ./rootfs, builds the OdinLink userspace (the RCCL net plugin and the odl_ar2
# all-reduce) and the hand-written decode kernels, and rebuilds ROCr. Third-party sources are FETCHED at pinned
# commits, not vendored (network needed on first build).
#
# Build (context is the REPO ROOT):
#   container/build.sh               # -> ds4-vllm-patched:local
#   podman build -t ds4-vllm-patched:local -f container/Dockerfile .
#
# The FROM line is pinned to the exact base digest the patches were built
# against. If that digest is no longer pullable, swap it for :latest and see
# "Base image drift" in ../README.md before trusting the result.

ARG DS4_BASE="docker.io/kyuz0/vllm-therock-gfx1151@sha256:25fd294fde9f729d1e75f109022ab4496c78190c0a6dc0142440529f7af20e4d"

# Rebuild the base image's exact ROCr source with AMD's complete event-age fix
# series plus the DS4 infinite-wait policy. The bundled June runtime can pin
# one CPU core in AsyncEventsLoop and another in an indefinite active wait.
FROM ${DS4_BASE} AS rocr-idle-fix
USER root
RUN dnf install -y git elfutils-libelf-devel numactl-devel && dnf clean all
COPY container/patches/rocr-force-block-indefinite-active-wait.patch /tmp/
RUN git clone --filter=blob:none --no-checkout \
        https://github.com/ROCm/rocm-systems.git /tmp/rocm-systems && \
    git -C /tmp/rocm-systems sparse-checkout init --cone && \
    git -C /tmp/rocm-systems sparse-checkout set projects/rocr-runtime && \
    git -C /tmp/rocm-systems checkout \
        d34cbb640961dcd0c0aa6c1c3c1cfa5d4becfa31 && \
    for commit in \
        c06ea68a592f6fcf08ada1e10e7b5b54b21aeae7 \
        933596e921d72e050fc982faebec0fea4cb2d64e \
        78b874d46014128675bae911934323361eaee03d; do \
      GIT_COMMITTER_DATE="$(git -C /tmp/rocm-systems show -s --format=%cI "$commit")" \
        git -C /tmp/rocm-systems \
          -c user.name="DS4 reproducible build" \
          -c user.email="ds4-build@invalid" \
          cherry-pick "$commit"; \
    done && \
    git -C /tmp/rocm-systems apply \
      /tmp/rocr-force-block-indefinite-active-wait.patch
RUN cmake \
        -S /tmp/rocm-systems/projects/rocr-runtime \
        -B /tmp/rocm-build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON && \
    cmake --build /tmp/rocm-build --parallel 16

# Build the OdinLink userspace -- the cluster fabric:
# libodl_tb5 (the stream API over /dev/odl_tb5_*), the RCCL net plugin RCCL
# dlopens via NCCL_NET_PLUGIN, and the odl_ar2 decode all-reduce that
# rootfs' odl_ar2.py ctypes-loads under DS4_ODL_AR2=1. Fetched at the same
# pin as odinlink/build-odinlink.sh, with the same local patch. The kernel
# driver (odl_tb5.ko) is a per-kernel HOST build — see odinlink/README.md.
FROM ${DS4_BASE} AS odinlink-build
USER root
RUN dnf install -y git cmake gcc && dnf clean all
COPY odinlink/odinlink-local.patch odinlink/ar2/odl_ar2.hip /tmp/
RUN git clone https://github.com/Geramy/OdinLink-Five /tmp/odinlink && \
    git -C /tmp/odinlink checkout 4534f5852253503e61f8540a175d7e32cb072c3b && \
    git -C /tmp/odinlink apply /tmp/odinlink-local.patch && \
    cd /tmp/odinlink && \
    cmake -S . -B build \
      -DBUILD_KERNEL_MODULE=OFF -DBUILD_DAEMON=OFF -DBUILD_TRAY=OFF \
      -DBUILD_NCCL=OFF -DBUILD_VERBS=OFF && \
    cmake --build build --target odl_tb5 --parallel 16 && \
    /opt/rocm/llvm/bin/clang -O3 -DNDEBUG -fPIC \
      -I third_party/rccl -I lib/include \
      -shared -Wl,-soname,librccl_net_odl_tb5.so \
      -Wl,-rpath,/usr/local/lib/odinlink \
      -o build/librccl_net_odl_tb5.so rccl/src/odl_tb5_plugin.c \
      build/lib/libodl_tb5.so.0.1.0 && \
    /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 -fPIC -shared \
      -I lib/include -Wl,-rpath,/usr/local/lib/odinlink \
      -o build/libodl_ar2.so /tmp/odl_ar2.hip \
      -L build/lib -lodl_tb5

FROM ${DS4_BASE}

# Apply the vLLM patch-set to the base's own sources (37 files), then overlay
# the 18 new files from rootfs/ (which mirrors "/"). The patch is the same
# reviewable diff in patches/; container/verify-patches.sh proves it applies.
COPY container/patches/vllm-upstream.patch /tmp/
RUN cd / && git apply -p1 --whitespace=nowarn /tmp/vllm-upstream.patch && rm /tmp/vllm-upstream.patch
COPY --chown=root:root container/rootfs/ /

# Replace PyTorch's bundled ROCr SONAME with the ABI-compatible build above.
# The immutable base image remains the rollback path.
COPY --from=rocr-idle-fix --chown=root:root \
  /tmp/rocm-build/rocr/lib/libhsa-runtime64.so.1.21.0 \
  /opt/venv/lib/python3.12/site-packages/_rocm_sdk_core/lib/libhsa-runtime64.so.1

# OdinLink userspace. The odl env profile points
# NCCL_NET_PLUGIN at the plugin and odl_ar2.py finds libodl_ar2.so next to
# itself in site-packages; both resolve libodl_tb5.so.0 via their rpath.
COPY --from=odinlink-build --chown=root:root \
  /tmp/odinlink/build/lib/libodl_tb5.so.0.1.0 \
  /tmp/odinlink/build/librccl_net_odl_tb5.so \
  /usr/local/lib/odinlink/
RUN ln -s libodl_tb5.so.0.1.0 /usr/local/lib/odinlink/libodl_tb5.so.0 && \
    ln -s libodl_tb5.so.0 /usr/local/lib/odinlink/libodl_tb5.so
COPY --from=odinlink-build --chown=root:root \
  /tmp/odinlink/build/libodl_ar2.so \
  /opt/venv/lib/python3.12/site-packages/libodl_ar2.so

# Build the decode kernels that ds4_moe_hip.py / ds4_hip_gemv.py ctypes-load.
# rootfs/ carries only the python wrappers: without these libraries each wrapper
# records a `disabled_reason` and the engine quietly uses the stock path, so a
# missing build here costs decode speed without failing anything.
COPY --chown=root:root container/native/ /tmp/ds4-native/
# rdma-core: libibverbs runtime for the optional `transport: ib` path (RCCL's
# built-in IB net + ib_ar2) and the headers ib_ar2.hip compiles against.
RUN dnf install -y rdma-core rdma-core-devel && dnf clean all
RUN /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 -shared -fPIC \
      -o /opt/venv/lib/python3.12/site-packages/libds4moe.so \
      /tmp/ds4-native/ds4_moe_mxfp4.cpp && \
    /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 -shared -fPIC \
      -o /opt/venv/lib/python3.12/site-packages/libds4fp8gemv.so \
      /tmp/ds4-native/ds4_fp8_gemv_hip.cpp && \
    /opt/rocm/bin/hipcc -O2 --offload-arch=gfx1151 -shared -fPIC \
      -o /opt/venv/lib/python3.12/site-packages/libib_ar2.so \
      /tmp/ds4-native/ib_ar2.hip -libverbs -lpthread && \
    /opt/rocm/bin/hipcc -O3 --offload-arch=gfx1151 -shared -fPIC \
      -o /opt/venv/lib/python3.12/site-packages/libds4idx.so \
      /tmp/ds4-native/ds4_idx_score.hip && \
    rm -rf /tmp/ds4-native

# Drop stale bytecode for the packages we touched so the first import recompiles
# from the patched sources. (COPY already writes fresh mtimes, which invalidates
# the .pyc; this is belt-and-suspenders.)
RUN find /opt/venv/lib/python3.12/site-packages/vllm \
         /opt/venv/lib/python3.12/site-packages/aiter \
         -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true

# com.github.containers.toolbox is what `toolbox create` checks for; the pinned
# base digest does not carry it, and box2 uses toolbox rather than distrobox.
LABEL com.github.containers.toolbox="true" \
      org.opencontainers.image.title="ds4-vllm-patched" \
      org.opencontainers.image.description="DeepSeek-V4-Flash vLLM patched engine (AMD gfx1151, TP=2, TB4-RDMA)" \
      ds4.base="docker.io/kyuz0/vllm-therock-gfx1151@sha256:25fd294f" \
      ds4.vllm_commit="470229c" \
      ds4.snapshot_equiv="localhost/ds4-vllm-patched:20260728"
