// Mic capture + PCM playback for the local speech-to-speech pipeline.
// Wire format both directions: 16 kHz mono int16 little-endian.

const S2S_RATE = 16000;

const CAPTURE_WORKLET = `
class CaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("capture-processor", CaptureProcessor);
`;

export class AudioEngine {
  constructor({ onLevel } = {}) {
    this.onLevel = onLevel || (() => {});
    this.ctx = null;
    this.micStream = null;
    this.onChunk = null;      // (ArrayBuffer int16 @16k) => void
    this.playHead = 0;        // scheduled playback time
    this.muted = false;
    this._resampleRemainder = new Float32Array(0);
    this._sources = new Set();
  }

  async start(onChunk) {
    this.onChunk = onChunk;
    this.ctx = new (window.AudioContext || window.webkitAudioContext)();
    await this.ctx.resume();

    const blob = new Blob([CAPTURE_WORKLET], { type: "application/javascript" });
    await this.ctx.audioWorklet.addModule(URL.createObjectURL(blob));

    this.micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });
    const src = this.ctx.createMediaStreamSource(this.micStream);
    const node = new AudioWorkletNode(this.ctx, "capture-processor");
    src.connect(node);

    const inRate = this.ctx.sampleRate;
    node.port.onmessage = (e) => {
      if (this.muted) return;
      const f32 = e.data;
      // level meter (input)
      let peak = 0;
      for (let i = 0; i < f32.length; i += 8) peak = Math.max(peak, Math.abs(f32[i]));
      this.onLevel(peak, "in");
      const ds = this._resampleTo16k(f32, inRate);
      if (ds.length && this.onChunk) this.onChunk(this._toInt16(ds).buffer);
    };
  }

  _resampleTo16k(f32, inRate) {
    if (inRate === S2S_RATE) return f32;
    const joined = new Float32Array(this._resampleRemainder.length + f32.length);
    joined.set(this._resampleRemainder);
    joined.set(f32, this._resampleRemainder.length);
    const ratio = inRate / S2S_RATE;
    const outLen = Math.floor(joined.length / ratio);
    const out = new Float32Array(outLen);
    for (let i = 0; i < outLen; i++) {
      const pos = i * ratio;
      const i0 = Math.floor(pos);
      const i1 = Math.min(i0 + 1, joined.length - 1);
      out[i] = joined[i0] + (joined[i1] - joined[i0]) * (pos - i0);
    }
    const consumed = Math.floor(outLen * ratio);
    this._resampleRemainder = joined.slice(consumed);
    return out;
  }

  _toInt16(f32) {
    const out = new Int16Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
      const s = Math.max(-1, Math.min(1, f32[i]));
      out[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return out;
  }

  // Schedule an incoming int16 @16k buffer for gapless playback.
  playChunk(arrayBuffer) {
    if (!this.ctx) return;
    const i16 = new Int16Array(arrayBuffer);
    if (!i16.length) return;
    const f32 = new Float32Array(i16.length);
    let peak = 0;
    for (let i = 0; i < i16.length; i++) {
      f32[i] = i16[i] / 0x8000;
      if ((i & 7) === 0) peak = Math.max(peak, Math.abs(f32[i]));
    }
    this.onLevel(peak, "out");
    const buf = this.ctx.createBuffer(1, f32.length, S2S_RATE);
    buf.getChannelData(0).set(f32);
    const src = this.ctx.createBufferSource();
    src.buffer = buf;
    src.connect(this.ctx.destination);
    const now = this.ctx.currentTime;
    if (this.playHead < now + 0.05) this.playHead = now + 0.05;
    src.start(this.playHead);
    this.playHead += buf.duration;
    this._sources.add(src);
    src.onended = () => this._sources.delete(src);
  }

  // Cut whatever is still queued to play (the "skip reply" control).
  stopPlayback() {
    for (const src of this._sources) {
      try { src.stop(); } catch {}
    }
    this._sources.clear();
    if (this.ctx) this.playHead = this.ctx.currentTime;
  }

  isPlaying() {
    return this.ctx && this.playHead > this.ctx.currentTime + 0.06;
  }

  setMuted(m) { this.muted = m; }

  stop() {
    if (this.micStream) this.micStream.getTracks().forEach((t) => t.stop());
    if (this.ctx) this.ctx.close();
    this.ctx = null;
    this.micStream = null;
  }
}
