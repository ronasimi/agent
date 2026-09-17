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
    && pacman -Scc --noconfirm

WORKDIR /app

# Upgrade pip and install standard harness requirements
RUN pip install --no-cache-dir --break-system-packages \
    prompt_toolkit \
    pyyaml \
    requests \
    beautifulsoup4 \
    ddgs \
    wikipedia \
    pdf2image \
    zeroconf \
    ollama \
    graphviz \
    playwright \
    markdown \
    weasyprint \
    pygments

# Install Playwright's headless Chromium browser
RUN playwright install chromium

# Default launch path (Interactive detached CLI mode)
CMD ["python", "/app/agent.py"]
