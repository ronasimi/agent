FROM archlinux:base

WORKDIR /app

# 1. Enable the multilib repository (required for many BlackArch tools)
RUN echo -e "\n[multilib]\nInclude = /etc/pacman.d/mirrorlist" >> /etc/pacman.conf

# 2. Initialize keyring, update system, and install prerequisites
RUN pacman-key --init && pacman-key --populate archlinux && \
    pacman -Syu --noconfirm && \
    pacman -S --noconfirm \
    python python-pip base-devel curl git wget sqlite gnupg

# 3. Add the BlackArch Linux repository
RUN curl -O https://blackarch.org/strap.sh && \
    chmod +x strap.sh && \
    ./strap.sh

# 4. Install pentesting tools and standard utilities
RUN pacman -S --noconfirm \
    nmap sqlmap hydra john tcpdump wireshark-cli bind \
    openbsd-netcat iproute2 iputils traceroute && \
    pacman -Scc --noconfirm
# 5. Set up Python agent dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir --break-system-packages -r requirements.txt

# Create mount points
RUN mkdir -p /app/workspace /app/memory /app/config /app/tools

# Copy modular tools and main script
COPY tools/ ./tools
COPY agent.py .

CMD ["python", "-u", "agent.py"]
