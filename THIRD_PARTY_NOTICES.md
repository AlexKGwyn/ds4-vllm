# Third-party code and licenses

This project builds on the projects below. Their sources are **fetched at
pinned revisions at build time, not redistributed here** — what this
repository itself ships is at most a derivative patch against them (each
patch is licensed like the code it modifies). Original code in this
repository (the `host/` orchestration, `container/` build tooling, the new
engine files under `container/rootfs/`, the `odinlink/` build/bringup scripts,
and docs) is licensed under [Apache-2.0](LICENSE). The local
`odinlink/nhi-throttle-mod/` kernel module is GPL-2.0
(`MODULE_LICENSE("GPL")`), as kernel modules must be.

| Component | Upstream | License | What this repo ships |
|---|---|---|---|
| **vLLM** | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) @ `470229c` | Apache-2.0 | A derivative patch ([`container/patches/vllm-upstream.patch`](container/patches/vllm-upstream.patch), 36 files) applied to the base image's own vLLM sources at build time, plus 15 new files under `container/rootfs/`. Change inventory: [`MANIFEST.md`](container/patches/MANIFEST.md). |
| **OdinLink-Five** | [github.com/Geramy/OdinLink-Five](https://github.com/Geramy/OdinLink-Five) @ `4534f58` | GPL-2.0 (kernel driver) / MIT (userspace) | Nothing redistributed — fetched at the pin by `odinlink/build-odinlink.sh` (host driver + lib) and by the `odinlink-build` image stage (net plugin), with this repo's [`odinlink/odinlink-local.patch`](odinlink/odinlink-local.patch) (derivative, licensed like the code it modifies) applied. The `odinlink/ar2/` all-reduce and the host tooling are this repo's own code. |
| **amd-strix-halo-vllm-toolboxes** | [github.com/kyuz0/amd-strix-halo-vllm-toolboxes](https://github.com/kyuz0/amd-strix-halo-vllm-toolboxes) | MIT | Nothing redistributed — the container build pulls the base image by pinned digest (`container/Dockerfile`). |
| **ROCR-Runtime** | [github.com/ROCm/ROCR-Runtime](https://github.com/ROCm/ROCR-Runtime) | MIT (NCSA-style) | One patch (`container/patches/rocr-force-block-indefinite-active-wait.patch`) applied to the base image's bundled runtime at image build time. |

**Model weights** (`deepseek-ai/DeepSeek-V4-Flash-0731`) are not included and
are governed by their own license on Hugging Face.
