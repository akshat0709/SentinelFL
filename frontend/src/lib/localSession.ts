/**
 * Token for the offline "Local Researcher" session.
 *
 * The local session lives only in React state, but API helpers read auth headers outside
 * React. This module makes the local token available to them (memory + sessionStorage),
 * so backend calls in offline mode carry `Authorization: Bearer local-...`, which the
 * backend accepts as the local demo identity.
 */
const KEY = 'sentinelfl_local_session_token';
let memoryToken: string | null = null;

export const localSession = {
  set(token: string): void {
    memoryToken = token;
    try { sessionStorage.setItem(KEY, token); } catch { /* not available (tests / SSR) */ }
  },
  get(): string | null {
    if (memoryToken) return memoryToken;
    try { return sessionStorage.getItem(KEY); } catch { return null; }
  },
  clear(): void {
    memoryToken = null;
    try { sessionStorage.removeItem(KEY); } catch { /* ignore */ }
  },
};
