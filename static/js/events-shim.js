// Minimal Node-style EventEmitter for the LiveAvatar UMD bundle, which
// expects a global `events$1.EventEmitter`.
(function (global) {
  class EventEmitter {
    constructor() { this._l = new Map(); this._max = 10; }
    setMaxListeners(n) { this._max = n; return this; }
    getMaxListeners() { return this._max; }
    on(ev, fn) { (this._l.get(ev) || this._l.set(ev, []).get(ev)).push(fn); return this; }
    addListener(ev, fn) { return this.on(ev, fn); }
    once(ev, fn) {
      const w = (...a) => { this.off(ev, w); fn(...a); };
      w._orig = fn;
      return this.on(ev, w);
    }
    off(ev, fn) {
      const arr = this._l.get(ev);
      if (arr) {
        const i = arr.findIndex((f) => f === fn || f._orig === fn);
        if (i >= 0) arr.splice(i, 1);
      }
      return this;
    }
    removeListener(ev, fn) { return this.off(ev, fn); }
    removeAllListeners(ev) { ev ? this._l.delete(ev) : this._l.clear(); return this; }
    listeners(ev) { return (this._l.get(ev) || []).slice(); }
    listenerCount(ev) { return (this._l.get(ev) || []).length; }
    eventNames() { return [...this._l.keys()]; }
    emit(ev, ...a) {
      const arr = this._l.get(ev);
      if (!arr || !arr.length) return false;
      arr.slice().forEach((fn) => fn(...a));
      return true;
    }
  }
  global.events$1 = { EventEmitter };
})(window);
