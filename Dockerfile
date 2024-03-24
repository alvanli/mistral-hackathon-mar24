# syntax=docker/dockerfile:1
FROM python/python:3.9.19-slim-bullseye as base
RUN python3 -m pip install -r requiremets.txt

WORKDIR /exp

CMD ["bash"]