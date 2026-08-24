FROM dolfinx/dolfinx:stable

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglu1-mesa \
    && rm -rf /var/lib/apt/lists/* \
    && python3 -m pip install --no-cache-dir pandas matplotlib gmsh
WORKDIR /work
ENV MPLCONFIGDIR=/tmp/matplotlib
CMD ["python3", "run_all.py"]
