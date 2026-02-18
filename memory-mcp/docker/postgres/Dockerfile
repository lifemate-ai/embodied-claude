# PostgreSQL 16 + pgvector + pgroonga + MeCab
# ARM64 (Apple Silicon) native build
FROM postgres:16-bookworm

ARG DEBIAN_FRONTEND=noninteractive

# Build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    file \
    git \
    libmecab-dev \
    libmsgpack-dev \
    libzmq3-dev \
    mecab \
    mecab-ipadic-utf8 \
    pkg-config \
    postgresql-server-dev-16 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# ── pgvector ──
ARG PGVECTOR_VERSION=0.8.0
RUN cd /tmp \
    && git clone --branch v${PGVECTOR_VERSION} --depth 1 \
       https://github.com/pgvector/pgvector.git \
    && cd pgvector \
    && make \
    && make install \
    && rm -rf /tmp/pgvector

# ── Groonga + pgroonga ──
ARG GROONGA_VERSION=14.1.2
ARG PGROONGA_VERSION=3.2.5
RUN cd /tmp \
    && wget -q https://packages.groonga.org/source/groonga/groonga-${GROONGA_VERSION}.tar.gz \
    && tar xf groonga-${GROONGA_VERSION}.tar.gz \
    && cd groonga-${GROONGA_VERSION} \
    && cmake -S . -B build \
       -DCMAKE_INSTALL_PREFIX=/usr/local \
       -DGRN_WITH_MECAB=ON \
       -DGRN_WITH_MRUBY=OFF \
    && cmake --build build -j$(nproc) \
    && cmake --install build \
    && ldconfig \
    && rm -rf /tmp/groonga*

RUN cd /tmp \
    && wget -q https://github.com/pgroonga/pgroonga/releases/download/${PGROONGA_VERSION}/pgroonga-${PGROONGA_VERSION}.tar.gz \
    && tar xf pgroonga-${PGROONGA_VERSION}.tar.gz \
    && cd pgroonga-${PGROONGA_VERSION} \
    && make \
    && make install \
    && rm -rf /tmp/pgroonga*

# ── MeCab NEologd dictionary (optional) ──
ARG INSTALL_NEOLOGD=true
RUN if [ "$INSTALL_NEOLOGD" = "true" ]; then \
    cd /tmp \
    && git clone --depth 1 https://github.com/neologd/mecab-ipadic-neologd.git \
    && cd mecab-ipadic-neologd \
    && ./bin/install-mecab-ipadic-neologd -n -y \
    && rm -rf /tmp/mecab-ipadic-neologd; \
    fi

# ── Japanese locale ──
RUN apt-get update && apt-get install -y --no-install-recommends locales \
    && sed -i '/ja_JP.UTF-8/s/^# //' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/*

ENV LANG=ja_JP.UTF-8
ENV LC_ALL=ja_JP.UTF-8

# ── Remove build dependencies ──
RUN apt-get purge -y --auto-remove \
    build-essential cmake git wget \
    postgresql-server-dev-16 \
    && rm -rf /var/lib/apt/lists/*
