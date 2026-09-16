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
    && pacman -Scc --noconfirm

WORKDIR /app

# Upgrade pip and install standard harness requirements
RUN pip install --no-cache-dir --break-system-packages \
    prompt_toolkit \
    pyyaml \
    requests \
    beautifulsoup4 \
    duckduckgo-search \
    wikipedia \
    pdf2image \
    zeroconf \
    ollama \
    graphviz

# Default launch path (Interactive detached CLI mode)
CMD ["python", "/app/agent.py"]
