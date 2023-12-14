#!/bin/bash --
###############################################################################
#
# Installation of Text-Generation-Webui with Cuda 11 and Code dated Sep 18 2023
#
# Note: The cuda 11 above refers to the cuda libs embedded inside the PY module
# packages. The root filesystem does not need to have CUDA libraries installed.
# The Rootfs only needs to have the latest nvidia-driver and supporting rpms.
#
###############################################################################

# Facebook's Llama2 Base model and it's specialized offshoots (CodeLlama, ...)
# work best with a back-dated version of the text-generation-webui from github.
# Newer text-generation-webui version are under heavy development for the next
# generation of Mistral and other mode advanced model configurations so runtime
# stability, model compatibility and GPU VRam bloat are contunually changing.
# The exact choice of GUI code from github, python modules and huggingface
# quantized (GPTQ) model(s) have been tested and confirmed to be stable with
# 10GB of GPU VRam utilization on 16GB VRam systems.

export CHROOT=/var/lib/warewulf/chroots/rocky-8/rootfs
export NFSAPPS=/opt/ai_apps
export CUDAROOT=/opt/cuda
export NSIGHTROOT=/opt/nvidia

export CUDA_REPOSITORY="http://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo"
export TGW_GITHUB="https://github.com/oobabooga/text-generation-webui.git"
export TGW_HASHREF="8466cf229ab29ace6e336a96f81f4eda44ca94fa"

# export APPTAINER_BIND="--bind /tmp:/tmp:rw,/opt:/opt:rw"
# The --bind /tmp:/tmp,/opt,/opt flag was moved into a variable called APPTAINER_BIND but
# that triggered an unusual side-effect where /root/--bind was substituted in front of the
# flag. /root/ == $PWD at the time.  Very odd so just hard code mounts for now.

echo "LLaMa2 / CodeLLaMa installation started $(date)"

# Command line to wrap chroot filesystem with a full apptainer container for native build/install 
apptainer_shell="apptainer exec --writable --containall --disable-cache --cleanenv --no-home --hostname rocky --allow-setuid --bind /tmp:/tmp,/opt:/opt ${CHROOT}/."

# Crucial llama_cpp_python module needs gcc-c++ or it will not build and many packages fail install
${apptainer_shell} bash -e -x -c "dnf -y install gcc-c++"

${apptainer_shell} bash -e -x -c "dnf -y install wget curl libpciaccess nvme-cli numad numatop yum-utils"

# Prevent dnf/yum from checking free disk space for future runs of dnf install on ramdisk
${apptainer_shell} bash -e -x -c "dnf -y config-manager --setopt=diskspacecheck=false --save"
${apptainer_shell} bash -e -x -c "dnf -y config-manager --add-repo ${CUDA_REPOSITORY}"

${apptainer_shell} bash -e -x -c "dnf -y install epel-release"
${apptainer_shell} bash -e -x -c "dnf -y install kernel-core kernel-modules kernel-headers"

PACKAGES="dnf-plugin-nvidia nvidia-driver nvidia-driver-devel nvidia-driver-cuda nvidia-persistenced"
${apptainer_shell} bash -e -x -c "dnf -y install ${PACKAGES}"

# Power management may be helpful or at worst a no-op.
${apptainer_shell} bash -e -x -c "systemctl set-default multi-user"
${apptainer_shell} bash -e -x -c "systemctl enable nvidia-powerd"
 
dnf -y --installroot="${CHROOT}" clean all

# Python interpreter build requirements, may be uninstalled after PY build
PACKAGES="tk-devel tcl-devel xz-devel gdbm-devel libffi-devel openssl-devel bzip2-devel libuuid-devel readline-devel sqlite-devel ncurses-devel"
${apptainer_shell} bash -e -x -c "dnf -y install ${PACKAGES}"
${apptainer_shell} bash -e -x -c 'export PYVER=3.11.4; export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; mkdir -pv ${INSTPATH}/src; cd ${INSTPATH}/src; wget https://www.python.org/ftp/python/3.11.4/Python-${PYVER}.tgz; tar zxvf Python-${PYVER}.tgz; cd Python-${PYVER}; ./configure --prefix=${INSTPATH}/python_${PYVER} --enable-optimizations; make -j 4 ; make install; cd ${INSTPATH}; rm -rf src'
${apptainer_shell} bash -e -x -c "rpm -e --nodeps ${PACKAGES}"

${apptainer_shell} bash -e -x -c 'dnf -y install git git-lfs'
${apptainer_shell} bash -e -x -c 'export PYVER=3.11.4; export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; cd ${INSTPATH}; git clone '"${TGW_GITHUB}"'; cd text-generation-webui; git checkout '"${TGW_HASHREF}"

cat <<EOF >"${NFSAPPS}/webui_llama2/run_python.sh"
#!/bin/bash --
export PYVER=3.11.4
export CODELLAMA_INSTALL_ROOT="\$(dirname \$(realpath -L "\$0"))"
export PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/bin:/bin:/usr/bin:/usr/local/bin"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cusparse/lib:\${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH}"

exec "\${CODELLAMA_INSTALL_ROOT}/python_3.11.4/bin/python3.11" "\$@"
EOF

chmod 755 "${NFSAPPS}/webui_llama2/run_python.sh"

PACKAGES="beautifulsoup4==4.12.2 soupsieve==2.5"
${apptainer_shell} bash -e -x -c 'export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; ${INSTPATH}/run_python.sh -mpip install --no-cache-dir '"${PACKAGES}"
PACKAGES="bs4==0.0.1 lit==16.0.6 ffmpy==0.3.1 pathtools==0.1.2"
${apptainer_shell} bash -e -x -c 'export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; ${INSTPATH}/run_python.sh -mpip install --no-cache-dir '"${PACKAGES}"

PACKAGES="nltk==3.8.1 numpy==1.24.0 scikit_learn==1.3.2 scipy==1.11.2 sentencepiece==0.1.99"
PACKAGES="${PACKAGES} torch==2.0.1 torchvision==0.15.2 transformers==4.33.2 click==8.1.7 joblib==1.3.2"
PACKAGES="${PACKAGES} regex==2023.8.8 threadpoolctl==3.2.0 sympy==1.12 networkx==3.1 Jinja2==3.1.2"
PACKAGES="${PACKAGES} nvidia_cuda_nvrtc_cu11==11.7.99 nvidia_cuda_runtime_cu11==11.7.99"
PACKAGES="${PACKAGES} nvidia_cuda_cupti_cu11==11.7.101 nvidia_cudnn_cu11==8.5.0.96"
PACKAGES="${PACKAGES} nvidia_cublas_cu11==11.10.3.66 nvidia_cufft_cu11==10.9.0.58"
PACKAGES="${PACKAGES} nvidia_curand_cu11==10.2.10.91 nvidia_cusolver_cu11==11.4.0.1"
PACKAGES="${PACKAGES} nvidia_cusparse_cu11==11.7.4.91 nvidia_nccl_cu11==2.14.3 Pillow==10.0.1"
PACKAGES="${PACKAGES} nvidia_nvtx_cu11==11.7.91 triton==2.0.0 wheel==0.42.0 cmake==3.27.5"
PACKAGES="${PACKAGES} tokenizers==0.13.3 safetensors==0.3.2 mpmath==1.3.0 MarkupSafe==2.1.3"
${apptainer_shell} bash -e -x -c 'export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; ${INSTPATH}/run_python.sh -mpip install --no-cache-dir '"${PACKAGES}"

PACKAGES="huggingface_hub==0.17.2 filelock==3.12.4 fsspec==2023.6.0 requests==2.31.0 tqdm==4.66.1 PyYAML==6.0.1 typing_extensions==4.8.0"
PACKAGES="${PACKAGES} packaging==23.1 charset_normalizer==3.2.0 idna==3.4 urllib3==1.26.16 certifi==2023.7.22"
${apptainer_shell} bash -e -x -c 'export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; ${INSTPATH}/run_python.sh -mpip install --no-cache-dir '"${PACKAGES}"

PACKAGES="fastparquet==2023.8.0 ninja==1.11.1 pandas==2.1.0 cramjam==2.7.0"
PACKAGES="${PACKAGES} python_dateutil==2.8.2 tzdata==2023.3"
${apptainer_shell} bash -e -x -c 'export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; ${INSTPATH}/run_python.sh -mpip install --no-cache-dir '"${PACKAGES}"

PACKAGES="absl_py==1.4.0 accelerate==0.23.0 aiofiles==23.1.0 aiohttp==3.8.5 aiosignal==1.3.1"
PACKAGES="${PACKAGES} altair==5.1.1 anyio==4.0.0 appdirs==1.4.4 async_timeout==4.0.3 attrs==23.1.0"
PACKAGES="${PACKAGES} bitsandbytes==0.41.1 blinker==1.7.0 cachetools==5.3.1 colorama==0.4.6"
PACKAGES="${PACKAGES} coloredlogs==15.0.1 contourpy==1.1.1 cycler==0.11.0 datasets==2.14.5 dill==0.3.7"
PACKAGES="${PACKAGES} diskcache==5.6.3 distro==1.8.0 docker_pycreds==0.4.0 einops==0.6.1 exllamav2==0.0.2"
PACKAGES="${PACKAGES} fastapi==0.95.2 flask==3.0.0 flask_cloudflared==0.0.14 fonttools==4.42.1"
PACKAGES="${PACKAGES} frozenlist==1.4.0 gitdb==4.0.10 GitPython==3.1.36 google_auth==2.23.0"
PACKAGES="${PACKAGES} google_auth_oauthlib==1.0.0 gradio==3.33.1 gradio_client==0.2.5 grpcio==1.58.0"
PACKAGES="${PACKAGES} h11==0.14.0 httpcore==0.18.0 httpx==0.25.0 humanfriendly==10.0 itsdangerous==2.1.2"
PACKAGES="${PACKAGES} jsonschema==4.19.0 jsonschema_specifications==2023.7.1 kiwisolver==1.4.5"
PACKAGES="${PACKAGES} linkify_it_py==2.0.2 llama_cpp_python==0.1.85 Markdown==3.4.4 markdown_it_py==2.2.0"
PACKAGES="${PACKAGES} matplotlib==3.8.0 mdit_py_plugins==0.3.3 mdurl==0.1.2 multidict==6.0.4"
PACKAGES="${PACKAGES} multiprocess==0.70.15 oauthlib==3.2.2 optimum==1.13.1 orjson==3.9.7 peft==0.5.0"
PACKAGES="${PACKAGES} pip_search==0.0.12 protobuf==4.24.3 psutil==5.9.5 pyarrow==13.0.0 pyasn1==0.5.0"
PACKAGES="${PACKAGES} pyasn1_modules==0.3.0 py_cpuinfo==9.0.0 pydantic==1.10.12 pydub==0.25.1 Pygments==2.16.1"
PACKAGES="${PACKAGES} pyparsing==3.1.1 python_multipart==0.0.6 referencing==0.30.2 requests_oauthlib==1.3.1"
PACKAGES="${PACKAGES} rich==13.5.3 rouge==1.0.1 rpds_py==0.10.3 rsa==4.9 scikit_build==0.17.6"
PACKAGES="${PACKAGES} semantic_version==2.10.0 sentence-transformers==2.2.2 sentry_sdk==1.31.0"
PACKAGES="${PACKAGES} setproctitle==1.3.2 smmap==5.0.1 sniffio==1.3.0 SpeechRecognition==3.10.0"
PACKAGES="${PACKAGES} sse_starlette==1.6.5 starlette==0.27.0 tensorboard==2.14.0 tensorboard_data_server==0.7.1"
PACKAGES="${PACKAGES} tiktoken==0.5.1 toolz==0.12.0 torchaudio==2.0.2 torch_grammar==0.3.3 uc_micro_py==1.0.2"
PACKAGES="${PACKAGES} uvicorn==0.23.2 wandb==0.15.10 websockets==11.0.3 werkzeug==3.0.1 xxhash==3.3.0"
PACKAGES="${PACKAGES} yarl==1.9.2"
${apptainer_shell} bash -e -x -c 'export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; ${INSTPATH}/run_python.sh -mpip install --no-cache-dir '"${PACKAGES}"

PACKAGES="https://github.com/jllllll/ctransformers-cuBLAS-wheels/releases/download/AVX2/ctransformers-0.2.27+cu117-py3-none-any.whl"
# Note that in these urls the (pairs of) versions of python was changed from cp310 to cp311
PACKAGES="${PACKAGES} https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.4.2/auto_gptq-0.4.2+cu117-cp311-cp311-linux_x86_64.whl"
PACKAGES="${PACKAGES} https://github.com/jllllll/exllama/releases/download/0.0.17/exllama-0.0.17+cu117-cp311-cp311-linux_x86_64.whl"
PACKAGES="${PACKAGES} https://github.com/jllllll/GPTQ-for-LLaMa-CUDA/releases/download/0.1.0/gptq_for_llama-0.1.0+cu117-cp311-cp311-linux_x86_64.whl"
PACKAGES="${PACKAGES} https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.85+cu117-cp311-cp311-linux_x86_64.whl"
${apptainer_shell} bash -e -x -c 'export INSTPATH="'"${NFSAPPS}"'"/webui_llama2; ${INSTPATH}/run_python.sh -mpip install --no-cache-dir '"${PACKAGES}"


${apptainer_shell} bash -e -x -c 'mkdir -pv "'"${NFSAPPS}"'"/webui_llama2/models'

# ========================================================================================
#
# This is a 13 Billion floating point x 8-Bit model that was downsampled (Quantized) from
# a 16-bit model on the fly during loading.  Note the --load-in-8bit flag on the command
# line.  Experience dictates that a drop fropm 32 to 16 or 16 to 8 does a good job of
# preserving accuracy when loaded on the fly.  Dropping to 4bit on the fly can make the
# language processing parts of the model very faulty.
#
# This occupies 13.4 GB of VRam leaving plenty of room for query processing but not much
# else. This is the best option for a dedicated 16GB GPU used for code and code-review work.
#
MODEL_REPO="https://huggingface.co/WizardLM/WizardCoder-Python-13B-V1.0"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/models; git lfs clone '"${MODEL_REPO}"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/text-generation-webui/models; ln -sfv ../../models/WizardCoder-Python-7B-V1.0 .'

cat <<EOF >"${NFSAPPS}/webui_llama2/run_wizard_13b_8bit_code.sh"
#!/bin/bash --
export PYVER=3.11.4
export CODELLAMA_INSTALL_ROOT="\$(dirname \$(realpath -L "\$0"))"
export PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/bin:/bin:/usr/bin:/usr/local/bin"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cusparse/lib:\${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH}"

# Will get KeyError: exception if not run from inside t-g-w/ directory
cd "\${CODELLAMA_INSTALL_ROOT}/text-generation-webui"
exec "\${CODELLAMA_INSTALL_ROOT}/python_3.11.4/bin/python3.11" ./server.py --load-in-8bit --listen  --api --extensions openai --model WizardCoder-Python-13B-V1.0 "\$@"
EOF

chmod 755 "${NFSAPPS}/webui_llama2/run_wizard_13b_8bit_code.sh"

# ========================================================================================
#
# This is a 7 Billion floating point x 16-Bit model. This model also uses 13.2 GB of VRam
# but does not use any downsampling (quantizition) during loading. This is the largest
# unaltered model that will fit and run on a 16GB GPU.
#
MODEL_REPO="https://huggingface.co/WizardLM/WizardCoder-Python-7B-V1.0"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/models; git lfs clone '"${MODEL_REPO}"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/text-generation-webui/models; ln -sfv ../../models/WizardCoder-Python-7B-V1.0 .'

cat <<EOF >"${NFSAPPS}/webui_llama2/run_wizard_7b_16bit_code.sh"
#!/bin/bash --
export PYVER=3.11.4
export CODELLAMA_INSTALL_ROOT="\$(dirname \$(realpath -L "\$0"))"
export PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/bin:/bin:/usr/bin:/usr/local/bin"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cusparse/lib:\${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH}"

# Will get KeyError: exception if not run from inside t-g-w/ directory
cd "\${CODELLAMA_INSTALL_ROOT}/text-generation-webui"
exec "\${CODELLAMA_INSTALL_ROOT}/python_3.11.4/bin/python3.11" ./server.py --listen  --api --extensions openai --model WizardCoder-Python-7B-V1.0 "\$@"
EOF

chmod 755 "${NFSAPPS}/webui_llama2/run_wizard_7b_16bit_code.sh"

# ========================================================================================
#
# This is a 13 Billion floating point x 4-bit model that will use about 9.2 GB of VRam.
# It will also use about 500M of VRam when under load answering small questions.
# If you plan to run other models on the same GPU for other purposes then this might
# be a good option for code-generation dominant GPU work.
#
# /opt/tools/webui_llama2/models/WizardCoder-Python-13B-V1.0-GPTQ/config.json
#     "quantization_config": {
#         "bits": 4,
#         . . .
#         "quant_method": "gptq"

MODEL_REPO="https://huggingface.co/TheBloke/WizardCoder-Python-13B-V1.0-GPTQ"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/models; git lfs clone '"${MODEL_REPO}"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/text-generation-webui/models; ln -sfv ../../models/WizardCoder-Python-13B-V1.0-GPTQ .'

cat <<EOF >"${NFSAPPS}/webui_llama2/run_wizard_13b_4bit_code.sh"
#!/bin/bash --
export PYVER=3.11.4
export CODELLAMA_INSTALL_ROOT="\$(dirname \$(realpath -L "\$0"))"
export PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/bin:/bin:/usr/bin:/usr/local/bin"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cusparse/lib:\${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH}"

# Will get KeyError: exception if not run from inside t-g-w/ directory
cd "\${CODELLAMA_INSTALL_ROOT}/text-generation-webui"
exec "\${CODELLAMA_INSTALL_ROOT}/python_3.11.4/bin/python3.11" ./server.py --listen --api --extensions openai --model WizardCoder-Python-13B-V1.0-GPTQ "\$@"
EOF

chmod 755 "${NFSAPPS}/webui_llama2/run_wizard_13b_4bit_code.sh"

# ========================================================================================
#
# This is a 7 Billion floating point x 4-bit model that will use about 6.0 GB of VRam.
# It will also use about 500M of VRam when under load answering small questions.
# If you plan to run other models on the same GPU for other purposes then this might
# be a good option for code-generation dominant GPU work.

# NOTE: This model generate an error message at startup about turning off injected attention
# support so   --no_inject_fused_attention   was added to the coammand line.
#
#
MODEL_REPO="https://huggingface.co/TheBloke/WizardCoder-Python-7B-V1.0-GPTQ"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/models; git lfs clone '"${MODEL_REPO}"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/text-generation-webui/models; ln -sfv ../../models/WizardCoder-Python-7B-V1.0-GPTQ .'

cat <<EOF >"${NFSAPPS}/webui_llama2/run_wizard_7b_4bit_code.sh"
#!/bin/bash --
export PYVER=3.11.4
export CODELLAMA_INSTALL_ROOT="\$(dirname \$(realpath -L "\$0"))"
export PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/bin:/bin:/usr/bin:/usr/local/bin"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cusparse/lib:\${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH}"

# Will get KeyError: exception if not run from inside t-g-w/ directory
cd "\${CODELLAMA_INSTALL_ROOT}/text-generation-webui"
exec "\${CODELLAMA_INSTALL_ROOT}/python_3.11.4/bin/python3.11" ./server.py --no_inject_fused_attention --api --extensions openai --listen --model WizardCoder-Python-7B-V1.0-GPTQ "\$@"
EOF

chmod 755 "${NFSAPPS}/webui_llama2/run_wizard_7b_4bit_code.sh"

# ========================================================================================
#
# This is a 1 Billion floating point x 32-Bit model. If the code requests you have are
# relatively simple and you want them to run as fast as possible then this is a good option.
# The underlying english language processing may not be great at resolving ambiguous context
# so be as specific as possible.  Like you are running up to a stranger on the street with
# a star-trek t-shirt and need to give them all the request info on the first shot. Be very
# very specific and ask for code in small chunks... Give an exampe of a program main, given
# an example of a class, given an exampe of a function, give an example of another function...
#
# Uses about 2.5GB of video ram.

MODEL_REPO="https://huggingface.co/WizardLM/WizardCoder-1B-V1.0"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/models; git lfs clone '"${MODEL_REPO}"
${apptainer_shell} bash -e -x -c 'cd "'"${NFSAPPS}"'"/webui_llama2/text-generation-webui/models; ln -sfv ../../models/WizardCoder-Python-7B-V1.0 .'

cat <<EOF >"${NFSAPPS}/webui_llama2/run_wizard_1b_32bit_code.sh"
#!/bin/bash --
export PYVER=3.11.4
export CODELLAMA_INSTALL_ROOT="\$(dirname \$(realpath -L "\$0"))"
export PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/bin:/bin:/usr/bin:/usr/local/bin"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cusparse/lib:\${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH}"

# Will get KeyError: exception if not run from inside t-g-w/ directory
cd "\${CODELLAMA_INSTALL_ROOT}/text-generation-webui"
exec "\${CODELLAMA_INSTALL_ROOT}/python_3.11.4/bin/python3.11" ./server.py --listen --api --extensions openai --model WizardCoder-1B-V1.0 "\$@"
EOF

chmod 755 "${NFSAPPS}/webui_llama2/run_wizard_1b_32bit_code.sh"

# ========================================================================================
#
# The --model-menu flag will present hte user with a list of models found in the t-g-u/models
# directory and let the user pick a model at launch time. This requires a little background
# knowledge about the model because it may not fit on your system without a --load-in-8bit
# quantization or other args.  Your mileage may vary =)
#
cat <<EOF >"${NFSAPPS}/webui_llama2/run_webui_server.sh"
#!/bin/bash --
export PYVER=3.11.4
export CODELLAMA_INSTALL_ROOT="\$(dirname \$(realpath -L "\$0"))"
export PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/bin:/bin:/usr/bin:/usr/local/bin"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cudnn/lib"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cusparse/lib:\${LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH="\${CODELLAMA_INSTALL_ROOT}/python_\${PYVER}/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:\${LD_LIBRARY_PATH}"

# Will get KeyError: exception if not run from inside t-g-w/ directory
cd "\${CODELLAMA_INSTALL_ROOT}/text-generation-webui"
exec "\${CODELLAMA_INSTALL_ROOT}/python_3.11.4/bin/python3.11" ./server.py --listen  --api --extensions openai --model-menu "\$@"
EOF

chmod 755 "${NFSAPPS}/webui_llama2/run_webui_server.sh"

# ========================================================================================

${apptainer_shell} bash -e -x -c 'chown -Rh test:users "'"${NFSAPPS}"'"/'

# ========================================================================================
# Container rebuild is crucial, other commands may not be needed but should be harmless
wwctl container build rocky-8
wwctl configure --all
wwctl overlay build
wwctl server restart

echo "LLaMa2 / CodeLLaMa installation completed $(date)"
