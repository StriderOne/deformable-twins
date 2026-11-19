FROM nvcr.io/nvidia/isaac-lab:3.0.0-kitless@sha256:d47169952b55effb46dba8f79761d381532b521561ecba38a3019adc7648884b

ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,display

USER isaaclab
WORKDIR /workspace/isaaclab

CMD ["isaaclab", "random_agent", "--task", "Isaac-Cartpole-Direct", "physics=newton_mjwarp", "--num_envs", "1", "--viz", "newton_gl"]
