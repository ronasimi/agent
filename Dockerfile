# ==========================================
# FILE: Dockerfile
# ==========================================
FROM archlinux:latest

# Force pacman to initialize and sync official Arch repositories
RUN pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    python \
    python-pip \
    poppler \
    poppler-glib \
    libnotify \
    graphviz \
    nmap \
    iputils \
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
    && pacman -Scc --noconfirm \
    && rm -rf /var/cache/pacman/pkg/* \
    && rm -rf /var/lib/pacman/sync/*

WORKDIR /app

# Copy requirement list and install
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Install Playwright's headless Chromium browser
RUN playwright install chromium

# Default launch path (Interactive detached CLI mode)
CMD ["python", "/app/agent.py"]
