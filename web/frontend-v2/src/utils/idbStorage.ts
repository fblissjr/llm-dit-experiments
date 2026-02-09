/**
 * IndexedDB Storage Adapter for Zustand Persist
 *
 * Implements zustand's StateStorage interface using IndexedDB instead of
 * localStorage. Provides ~50MB+ quota (vs ~5-10MB), async writes (no main
 * thread blocking), and native binary support.
 *
 * Includes one-time migration from localStorage for existing users.
 */

import type { StateStorage } from 'zustand/middleware';

const DB_NAME = 'llm-dit-studio';
const STORE_NAME = 'zustand';
const DB_VERSION = 1;

/** Cached database connection (opened once, reused across calls). */
let dbPromise: Promise<IDBDatabase> | null = null;

/**
 * Open (or reuse) the IndexedDB connection.
 *
 * Creates the object store on first open. Caches the connection promise
 * so concurrent calls share the same connection.
 */
function openDB(): Promise<IDBDatabase> {
  if (dbPromise) return dbPromise;

  dbPromise = new Promise<IDBDatabase>((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);

    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE_NAME)) {
        db.createObjectStore(STORE_NAME);
      }
    };

    request.onsuccess = () => resolve(request.result);

    request.onerror = () => {
      // Clear cache so next attempt retries
      dbPromise = null;
      reject(request.error);
    };
  });

  return dbPromise;
}

/**
 * IndexedDB-backed StateStorage for zustand persist middleware.
 *
 * Falls back to localStorage on IndexedDB failure (e.g., private browsing
 * in some browsers, or if IndexedDB is disabled).
 */
export const idbStorage: StateStorage = {
  getItem: async (name: string): Promise<string | null> => {
    try {
      const db = await openDB();
      return new Promise<string | null>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readonly');
        const store = tx.objectStore(STORE_NAME);
        const request = store.get(name);

        request.onsuccess = () => resolve(request.result ?? null);
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.warn('[idbStorage] getItem failed, falling back to localStorage:', error);
      return localStorage.getItem(name);
    }
  },

  setItem: async (name: string, value: string): Promise<void> => {
    try {
      const db = await openDB();
      return new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const request = store.put(value, name);

        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.warn('[idbStorage] setItem failed, falling back to localStorage:', error);
      try {
        localStorage.setItem(name, value);
      } catch {
        // localStorage also failed -- data not persisted this time
      }
    }
  },

  removeItem: async (name: string): Promise<void> => {
    try {
      const db = await openDB();
      return new Promise<void>((resolve, reject) => {
        const tx = db.transaction(STORE_NAME, 'readwrite');
        const store = tx.objectStore(STORE_NAME);
        const request = store.delete(name);

        request.onsuccess = () => resolve();
        request.onerror = () => reject(request.error);
      });
    } catch (error) {
      console.warn('[idbStorage] removeItem failed:', error);
    }
  },
};

/**
 * One-time migration from localStorage to IndexedDB.
 *
 * Reads the given key from localStorage, writes it to IndexedDB, then
 * removes it from localStorage. Idempotent -- if the key doesn't exist
 * in localStorage, this is a no-op.
 *
 * Call this at module load (before the store hydrates) so existing users
 * don't lose their history.
 */
export async function migrateFromLocalStorage(key: string): Promise<void> {
  try {
    const existing = localStorage.getItem(key);
    if (!existing) return;

    await idbStorage.setItem(key, existing);
    localStorage.removeItem(key);
    console.log(`[idbStorage] Migrated '${key}' from localStorage to IndexedDB`);
  } catch (error) {
    // Migration failed -- localStorage data stays put, will be read by
    // the fallback path in getItem on next load
    console.warn('[idbStorage] Migration failed (data preserved in localStorage):', error);
  }
}
