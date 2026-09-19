(function (root, factory) {
  const exported = factory();
  if (typeof module === 'object' && module.exports) module.exports = exported;
  if (root) root.CommandHistoryBuffer = exported.CommandHistoryBuffer;
})(typeof globalThis !== 'undefined' ? globalThis : this, function () {
  class CommandHistoryBuffer {
    constructor({ storage = null, key = 'agent.webui.commandHistory.v1', limit = 100 } = {}) {
      this.storage = storage;
      this.key = key;
      this.limit = Math.max(1, Number(limit) || 100);
      this.items = [];
      this.index = 0;
      this.draft = '';
      this.load();
    }

    load() {
      let rows = [];
      try {
        const parsed = JSON.parse(this.storage?.getItem(this.key) || '[]');
        if (Array.isArray(parsed)) rows = parsed;
      } catch (_) {}
      this.items = rows
        .filter((value) => typeof value === 'string' && value.trim())
        .map((value) => value.trim())
        .slice(-this.limit);
      this.reset();
      return this.items.slice();
    }

    persist() {
      try {
        this.storage?.setItem(this.key, JSON.stringify(this.items.slice(-this.limit)));
      } catch (_) {}
    }

    push(value) {
      const text = String(value ?? '').trim();
      if (!text) return false;
      if (this.items[this.items.length - 1] !== text) this.items.push(text);
      if (this.items.length > this.limit) this.items = this.items.slice(-this.limit);
      this.persist();
      this.reset();
      return true;
    }

    reset() {
      this.index = this.items.length;
      this.draft = '';
    }

    isBrowsing() {
      return this.index !== this.items.length;
    }

    move(delta, currentValue = '') {
      if (!this.items.length) return null;
      if (!this.isBrowsing()) this.draft = String(currentValue ?? '');
      const next = Math.max(0, Math.min(this.items.length, this.index + Number(delta || 0)));
      if (next === this.index) return null;
      this.index = next;
      return this.index === this.items.length ? this.draft : this.items[this.index];
    }

    restoreDraft() {
      const value = this.isBrowsing() ? this.draft : null;
      this.reset();
      return value;
    }
  }

  return { CommandHistoryBuffer };
});
