import WaveSurfer from "wavesurfer.js";

export type WaveformController = Pick<WaveSurfer, "destroy" | "playPause">;

export function createWaveform(
  container: HTMLElement,
  url: string,
  bearerToken: string,
): WaveformController {
  const headers = bearerToken ? { Authorization: `Bearer ${bearerToken}` } : undefined;
  return WaveSurfer.create({
    container,
    url,
    height: 88,
    waveColor: "#7ca6ed",
    progressColor: "#f6c453",
    cursorColor: "#ffffff",
    normalize: true,
    fetchParams: headers ? { headers } : undefined,
  });
}
