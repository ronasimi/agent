# ==========================================
# FILE: Dockerfile
# ==========================================
FROM archlinux:latest

RUN pacman -Syu --noconfirm --needed \
      python \
      python-pip \
      git \
      bubblewrap \
      poppler \
      poppler-glib \
      libnotify \
      graphviz \
      nmap \
      iputils \
      iproute2 \
      mtr \
      bind \
      sqlite \
      systemd \
      kitty-terminfo \
      nss \
      alsa-lib \
      atk \
      at-spi2-core \
      cups \
      libdrm \
      mesa \
      libxcomposite \
      libxdamage \
      libxrandr \
      libxkbcommon \
      pango \
      cairo \
      gdk-pixbuf2 \
      fontconfig \
      ttf-dejavu \
      && pacman -Scc --noconfirm && \
      rm -rf /var/cache/pacman/pkg/* /var/lib/pacman/sync/*

RUN useradd --uid 1000 --create-home --shell /bin/bash agent && \
    mkdir -p /app/config /app/workspace /app/memory && \
    chown -R agent:agent /app

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

COPY --chown=agent:agent . /app

USER agent

# Playwright is retained for explicit screenshot requests.
RUN playwright install chromium

CMD ["python", "/app/agent.py"]
