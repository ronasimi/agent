(() => {
  const root = document.documentElement;
  const fontSpec = '16px "Material Design Icons"';

  function markReady(ready) {
    root.classList.toggle('mdi-ready', Boolean(ready));
    root.classList.toggle('mdi-fallback', !ready);
  }

  async function verifyMdiFont() {
    if (!document.fonts || typeof document.fonts.load !== 'function') {
      markReady(false);
      return false;
    }
    try {
      const timeout = new Promise(resolve => setTimeout(() => resolve([]), 1800));
      const loaded = await Promise.race([document.fonts.load(fontSpec), timeout]);
      const ready = Array.isArray(loaded) && loaded.length > 0;
      markReady(ready);
      return ready;
    } catch {
      markReady(false);
      return false;
    }
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', verifyMdiFont, {once: true});
  } else {
    verifyMdiFont();
  }
  window.addEventListener('load', () => {
    verifyMdiFont();
    setTimeout(verifyMdiFont, 2000);
  }, {once: true});
})();
