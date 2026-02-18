FROM bitsandbytes_nv:latest 


RUN pip install --upgrade pip && \
    pip install --index-url https://pypi.org/simple --no-cache-dir nvflare && \
    pip install --index-url https://pypi.org/simple --no-cache-dir datasets && \
    pip install --index-url https://pypi.org/simple --no-cache-dir peft && \
    pip install --index-url https://pypi.org/simple --no-cache-dir trl && \
    pip install --index-url https://pypi.org/simple --no-cache-dir pympler

RUN pip install --index-url https://pypi.org/simple --no-cache-dir tensorboard

ENV HF_HOME=/files/model
ENV TRANSFORMERS_CACHE=/files/model

RUN pip install --index-url https://pypi.org/simple --no-cache-dir matplotlib
RUN pip install --index-url https://pypi.org/simple --no-cache-dir -U bitsandbytes>=0.46.1
