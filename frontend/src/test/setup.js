import '@testing-library/jest-dom/vitest';
import { afterEach } from 'vitest';
import { cleanup } from '@testing-library/react';

if (!globalThis.ResizeObserver) {
  globalThis.ResizeObserver = class ResizeObserverMock {
    constructor(callback) {
      this.callback = callback;
    }

    observe(target) {
      this.callback([
        {
          target,
          contentRect: { width: 800, height: 300 },
        },
      ]);
    }

    unobserve() {}

    disconnect() {}
  };
}

afterEach(() => {
  cleanup();
  localStorage.clear();
});
